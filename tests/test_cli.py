"""CLI smoke checks that do not load model backends or call external services."""

from pathlib import Path
import contextlib
import importlib
import io
import json
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CliSmokeTests(unittest.TestCase):
    def test_help_without_model_dependencies(self):
        # Disable site packages and reject model imports explicitly, even if the
        # calling environment exposes them through PYTHONPATH.
        bootstrap = """
import importlib.abc
import runpy
import sys

class RejectModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'torch', 'transformers', 'numpy', 'igraph', 'openai', 'litellm',
            'vllm', 'gritlm', 'sentence_transformers',
        }:
            raise ImportError('CLI help imported model dependency: ' + fullname)

sys.meta_path.insert(0, RejectModelImports())
sys.argv = ['main.py', '--help']
runpy.run_path('main.py', run_name='__main__')
"""
        result = subprocess.run(
            [sys.executable, "-S", "-c", bootstrap],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in (
            "--dataset", "--mode", "--judge_llm_name", "--subject_level_Entity_Cap",
            "--graph_search_mode", "--agent_apply_to", "--agent_max_tokens",
        ):
            self.assertIn(option, result.stdout)

    def test_default_strategy_and_budget(self):
        args = main.build_parser().parse_args([])
        self.assertEqual(args.graph_search_mode, "ppr")
        self.assertEqual(args.agent_apply_to, "round2")
        self.assertEqual(args.agent_fallback, "ppr")
        self.assertIsNone(args.num_to_retrieve)
        self.assertEqual(main.agent_budget_from_args(args), {
            "max_steps": 8,
            "max_tool_calls": 16,
            "max_expansions": 64,
            "max_path_length": 4,
            "max_neighbors": 8,
            "max_tokens": None,
            "max_seconds": 60.0,
        })

    def test_cumulative_token_limit_can_be_explicitly_disabled_or_enabled(self):
        for value, expected in (("none", None), ("NONE", None), ("12000", 12000)):
            with self.subTest(value=value):
                args = main.build_parser().parse_args(["--agent_max_tokens", value])
                self.assertEqual(main.agent_budget_from_args(args)["max_tokens"], expected)

    def test_zero_queries_does_not_initialize_models(self):
        result = subprocess.run(
            [sys.executable, "-S", str(PROJECT_ROOT / "main.py"), "--test_n", "0"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_budgets_fail_during_argument_parsing(self):
        for arguments in (
            ["--agent_max_steps", "0"],
            ["--agent_max_tokens", "-1"],
            ["--agent_max_tokens", "0"],
            ["--agent_max_tokens", "nan"],
            ["--agent_max_seconds", "nan"],
            ["--agent_max_seconds", "inf"],
            ["--agent_apply_to", "round1"],
            ["--test_n", "-1"],
            ["--num_to_retrieve", "0"],
        ):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main.build_parser().parse_args(arguments)
                self.assertEqual(error.exception.code, 2)

    def test_retrieve_and_qa_receive_identical_retrieval_configuration(self):
        calls = {}

        class StubRetriever:
            def retrieve(self, **kwargs):
                calls["retrieve"] = kwargs
                return []

            def rag_qa(self, **kwargs):
                calls["qa"] = kwargs
                return [], [], []

        options = {
            "dataset_name": "sample",
            "queries": ["question"],
            "gold_docs": [["evidence"]],
            "gold_answers": [["answer"]],
            "subject_cap": 2,
            "result_save_root": "results",
            "save_dir": "outputs",
            "llm_name": "test-llm",
            "num_to_retrieve": 11,
            "judge_model_name": "test-judge",
            "judge_concurrency": 3,
            "judge_batch_size": 7,
            "eval_k_list": (2, 5),
            "graph_search_mode": "hybrid",
            "agent_apply_to": "round2",
            "agent_budget": {"max_steps": 3},
            "agent_fallback": "dpr",
        }
        with patch.object(main, "save_rag_qa_results_to_json") as save:
            for mode in ("retrieve", "qa"):
                main.run_pipeline(StubRetriever(), mode=mode, **options)
        self.assertEqual(calls["qa"].pop("gold_answers"), options["gold_answers"])
        self.assertEqual(calls["qa"], calls["retrieve"])
        for key in (
            "subject_cap", "num_to_retrieve", "judge_model_name", "judge_concurrency",
            "judge_batch_size", "eval_k_list", "result_save_root", "graph_search_mode",
            "agent_apply_to", "agent_budget", "agent_fallback",
        ):
            self.assertEqual(calls["qa"][key], options[key])
        self.assertTrue(calls["qa"]["keep_miss_distribution"])
        self.assertEqual(save.call_args.kwargs["extra_info"]["judge_model_name"], "test-judge")

    def test_replay_reads_only_first_round_retrieval_fields(self):
        row = {
            "query": "q",
            "gold_docs": ["secret-gold"],
            "missing_gold_at5": ["secret-gold"],
            "round1": {"final_docs": ["evidence"], "final_top5": ["evidence"],
                       "final_scores": [0.8], "gold_answers": ["secret-gold"]},
            "judge1": {"can_answer": True, "evidence_docs": ["evidence"],
                       "gold_docs": ["secret-gold"], "_judge_metadata": {
                           "judge_model": "frozen-judge", "gold_answers": ["secret-gold"],
                       }},
            "round2": {"final_docs": ["future-evidence"]},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            replay = main.load_round1_replay(str(path))
            self.assertEqual(set(replay[0]), {"query", "round1", "judge1"})
            self.assertNotIn("secret-gold", json.dumps(replay))
            self.assertNotIn("future-evidence", json.dumps(replay))
            calls = []
            retriever = SimpleNamespace(
                retrieve=lambda **kwargs: calls.append(kwargs),
                rag_qa=lambda **kwargs: (calls.append(kwargs) or ([], [], [])),
            )
            with patch.object(main, "save_rag_qa_results_to_json") as save:
                for mode in ("retrieve", "qa"):
                    main.run_pipeline(
                        retriever, mode=mode, dataset_name="sample", queries=["q"],
                        gold_docs=None, gold_answers=None, subject_cap=4,
                        result_save_root=None, save_dir=directory, llm_name="test",
                        round1_replay_path=str(path),
                    )
            self.assertEqual(calls[0]["round1_replay"], replay)
            self.assertEqual(calls[1]["round1_replay"], replay)
            self.assertEqual(save.call_args.kwargs["extra_info"]["round1_replay_path"], str(path))

    def test_malformed_replay_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.jsonl"
            for text in ("not json\n", "[]\n", '{"query": "q"}\n'):
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text), self.assertRaises(ValueError):
                    main.load_round1_replay(str(path))


class LazyBackendTests(unittest.TestCase):
    def test_backend_packages_import_without_optional_dependencies(self):
        code = """
import importlib
import importlib.abc
import sys
class RejectBackends(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'openai', 'vllm', 'gritlm', 'boto3', 'transformers'}:
            raise ImportError('unselected backend imported: ' + fullname)
sys.meta_path.insert(0, RejectBackends())
for name in ('llm', 'embedding_model', 'information_extraction'):
    importlib.import_module('src.SRgraphrag.' + name)
"""
        result = subprocess.run(
            [sys.executable, "-S", "-c", code], cwd=PROJECT_ROOT,
            text=True, capture_output=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_embedding_factory_imports_only_selected_backend(self):
        factory = importlib.import_module("src.SRgraphrag.embedding_model")
        fake_backend = type("FakeEmbedding", (), {})
        with patch.object(factory, "import_module", return_value=SimpleNamespace(NVEmbedV2EmbeddingModel=fake_backend)) as loader:
            self.assertIs(factory._get_embedding_model_class("nvidia/NV-Embed-v2"), fake_backend)
            loader.assert_called_once_with(".NVEmbedV2", factory.__name__)
        # Do not let the fake export escape into tests run later in this process.
        factory.__dict__.pop("NVEmbedV2EmbeddingModel", None)

    def test_llm_factory_imports_only_selected_backend(self):
        factory = importlib.import_module("src.SRgraphrag.llm")
        fake_backend = SimpleNamespace(from_experiment_config=lambda config: ("selected", config.llm_name))
        config = SimpleNamespace(llm_name="test-online", llm_base_url="https://example.invalid")
        with patch.dict(factory.os.environ, {}, clear=True), patch.object(factory, "import_module", return_value=SimpleNamespace(CacheOpenAI=fake_backend)) as loader:
            self.assertEqual(factory._get_llm_class(config), ("selected", "test-online"))
            loader.assert_called_once_with(".openai_gpt", factory.__name__)
        factory.__dict__.pop("CacheOpenAI", None)


if __name__ == "__main__":
    unittest.main()
