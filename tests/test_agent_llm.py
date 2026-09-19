import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from src.SRgraphrag.retrieval.agent_llm import AgentLLMAdapter


class Client:
    base_url = "https://example.invalid/v1"

    def __init__(self):
        self.calls = []
        self.options = []
        self.chat = SimpleNamespace(completions=self)

    def with_options(self, **kwargs):
        self.options.append(kwargs)
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"action":"stop","arguments":{}}'), finish_reason="stop")], usage=SimpleNamespace(prompt_tokens=12, completion_tokens=5))


class AdapterTests(unittest.TestCase):
    def test_cache_ignores_deadline_not_model_or_token_cap(self):
        client = Client()
        llm = SimpleNamespace(openai_client=client, llm_config=SimpleNamespace(generate_params={"model": "deepseek-chat", "max_completion_tokens": 400}))
        with TemporaryDirectory() as directory:
            adapter = AgentLLMAdapter(llm, Path(directory) / "cache.sqlite")
            def messages(seconds, tokens=100):
                return [{"role":"user", "content":json.dumps({"llm_limits":{"timeout_seconds":seconds,"max_completion_tokens":tokens},"index_version":"v1"})}]
            self.assertFalse(adapter(messages(30))[2])
            self.assertTrue(adapter(messages(20))[2])
            self.assertFalse(adapter(messages(20, 50))[2])
            self.assertEqual(client.options[0]["max_retries"], 0)
            self.assertGreater(client.options[0]["timeout"], 0)
            self.assertLessEqual(client.options[0]["timeout"], 30.0)
            self.assertEqual(client.calls[0]["max_tokens"], 100)
            self.assertNotIn("max_completion_tokens", client.calls[0])
            self.assertNotIn("timeout_seconds", client.calls[0]["messages"][-1]["content"])

    def test_unsupported_backend_is_explicit(self):
        with self.assertRaises(ValueError):
            AgentLLMAdapter(object(), "unused.sqlite")([])

    def test_prompt_change_does_not_reuse_old_response_cache(self):
        from src.SRgraphrag.prompts.templates.agent_graph_search import SYSTEM_PROMPT
        client = Client()
        llm = SimpleNamespace(openai_client=client, llm_config=SimpleNamespace(generate_params={"model": "deepseek-chat"}))
        state = {"llm_limits": {"timeout_seconds": 30, "max_completion_tokens": 100}}
        with TemporaryDirectory() as directory:
            adapter = AgentLLMAdapter(llm, Path(directory) / "cache.sqlite")
            def messages(prompt):
                return [{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps(state)}]
            self.assertFalse(adapter(messages("old prompt without complete envelope examples"))[2])
            self.assertFalse(adapter(messages(SYSTEM_PROMPT))[2])
            self.assertTrue(adapter(messages(SYSTEM_PROMPT))[2])
            self.assertEqual(len(client.calls), 2)
