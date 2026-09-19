"""Offline judge protocol checks with in-memory HTTP responses."""

import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.SRgraphrag.retrieval.judge import judge_answerability_and_bridge, resolve_judge_metadata


def response(content):
    return {"choices": [{"message": {"content": json.dumps(content)}}]}


class FakeProgress:
    def update(self, amount):
        pass

    def close(self):
        pass


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.prompt = Path(self.directory.name) / "prompt.json"
        self.prompt.write_text(json.dumps({"system_prompt": "Judge", "user_prefix": "Evidence:\n"}), encoding="utf-8")
        self.calls = []

    def fake_modules(self, responses):
        calls = self.calls

        class FakeResponse:
            status = 200

            def __init__(self, outcome):
                self.delay, self.outcome = outcome if isinstance(outcome, tuple) else (0, outcome)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def text(self):
                await asyncio.sleep(self.delay)
                if isinstance(self.outcome, Exception):
                    raise self.outcome
                return json.dumps(self.outcome)

        class FakeSession:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            def post(self, url, *, headers, json):
                query = json["messages"][-1]["content"].split("query: ", 1)[1].split("\n", 1)[0]
                calls.append({"url": url, "payload": json, "query": query})
                return FakeResponse(responses[query])

        class FakeClientError(Exception):
            pass

        return {
            "aiohttp": SimpleNamespace(
                ClientSession=FakeSession,
                ClientTimeout=lambda **kwargs: None,
                TCPConnector=lambda **kwargs: None,
                ClientError=FakeClientError,
            ),
            "tqdm": SimpleNamespace(tqdm=lambda **kwargs: FakeProgress()),
        }

    def judge(self, responses, **kwargs):
        queries = list(responses)
        docs = [[f"{query}-evidence"] for query in queries]
        with patch.dict(sys.modules, self.fake_modules(responses)), patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "offline-test-key",
            "DEEPSEEK_BASE_URL": "https://example.invalid/v1/",
            "DEEPSEEK_MODEL": "unused-environment-model",
        }):
            return judge_answerability_and_bridge(
                queries, docs, prompt_json=str(self.prompt),
                judge_model_name="actual-test-judge", **kwargs,
            )

    def test_empty_batch_needs_no_dependencies_credentials_or_prompt(self):
        with patch.dict(sys.modules, {"aiohttp": None}), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(judge_answerability_and_bridge([], [], prompt_json="missing.json"), [])

    def test_mismatched_lengths_fail_before_loading_backends(self):
        with patch.dict(sys.modules, {"aiohttp": None}):
            with self.assertRaises(ValueError):
                judge_answerability_and_bridge(["q"], [])

    def test_later_failure_does_not_overwrite_successful_batch_first_item(self):
        rows = self.judge({
            "first": response({"can_answer": True, "evidence_ids": ["D1"]}),
            "failure": (0.02, ValueError("unexpected fake response error")),
            "third": (0.01, response({"can_answer": True, "evidence_ids": ["D1"]})),
        }, judge_concurrency=3, judge_batch_size=3)
        self.assertTrue(rows[0]["can_answer"])
        self.assertEqual(rows[0]["evidence_docs"], ["first-evidence"])
        self.assertEqual(rows[1]["_error"], "judge_exception")
        self.assertTrue(rows[2]["can_answer"])
        self.assertEqual(rows[2]["evidence_docs"], ["third-evidence"])
        self.assertTrue(all(row.get("_error") != "none_result" for row in rows))

    def test_non_object_json_is_a_parse_failure_not_an_indexed_task_exception(self):
        rows = self.judge({"list": response([]), "null": response(None), "number": response(42)})
        self.assertEqual([row["_error"] for row in rows], ["parsed_not_dict"] * 3)
        self.assertTrue(all(not row["can_answer"] and not row["bridge_possible"] for row in rows))

    def test_bad_envelope_and_bad_content_have_distinct_errors(self):
        rows = self.judge({"envelope": [], "shape": {"choices": []}, "text": {
            "choices": [{"message": {"content": "not JSON"}}],
        }})
        self.assertEqual([row["_error"] for row in rows], ["raw_not_dict", "bad_response_shape", "non_json_output"])

    def test_model_endpoint_prompt_and_metadata_reflect_actual_request(self):
        rows = self.judge({"q": response({
            "can_answer": False, "bridge_possible": True, "bridge_question": "bridge?",
            "evidence_ids": ["D1"], "bridge_evidence_ids": ["D1"],
        })}, judge_concurrency=2, judge_batch_size=4)
        call = self.calls[0]
        self.assertEqual(call["url"], "https://example.invalid/v1/chat/completions")
        self.assertEqual(call["payload"]["model"], "actual-test-judge")
        self.assertEqual(call["payload"]["temperature"], 0.0)
        self.assertEqual(call["payload"]["messages"][0]["content"], "Judge")
        metadata = rows[0]["_judge_metadata"]
        self.assertEqual(metadata["judge_model"], call["payload"]["model"])
        self.assertEqual(metadata["judge_endpoint"], call["url"])
        self.assertEqual(metadata["judge_prompt_sha256"], hashlib.sha256(self.prompt.read_bytes()).hexdigest())
        self.assertEqual(metadata["judge_concurrency"], 2)
        self.assertEqual(metadata["judge_batch_size"], 4)
        self.assertNotIn("offline-test-key", json.dumps(metadata))
        self.assertTrue(rows[0]["bridge_possible"])
        self.assertEqual(rows[0]["bridge_question"], "bridge?")

    def test_successful_answer_clears_bridge_request(self):
        rows = self.judge({"q": response({
            "can_answer": True, "bridge_possible": True, "bridge_question": "ignored",
            "bridge_evidence_ids": ["D1"],
        })})
        self.assertFalse(rows[0]["bridge_possible"])
        self.assertEqual(rows[0]["bridge_question"], "")
        self.assertEqual(rows[0]["bridge_evidence_ids"], [])

    def test_metadata_helper_preserves_legacy_base_url(self):
        with patch.dict(os.environ, {"DEEPSEEK_BASE_URL": "https://example.invalid/"}):
            metadata = resolve_judge_metadata("model", 0, 0, str(self.prompt))
        self.assertEqual(metadata["judge_endpoint"], "https://example.invalid/v1/chat/completions")
        self.assertEqual(metadata["judge_concurrency"], 1)
        self.assertEqual(metadata["judge_batch_size"], 1)


if __name__ == "__main__":
    unittest.main()
