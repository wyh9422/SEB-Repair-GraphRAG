"""Execute the actual orchestration methods with no model imports or requests."""

import ast
from dataclasses import dataclass
import logging
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from src.SRgraphrag.retrieval.evidence import align_scores_to_top5
from src.SRgraphrag.retrieval.metrics import all_recall_at_k, avg_recall_at_k, hit_at_1, missing_list, mrr_first_hit


@dataclass
class QuerySolution:
    question: str
    docs: list
    doc_scores: list = None


def method(name):
    source = Path(__file__).resolve().parents[1] / "src/SRgraphrag/SRgraphrag.py"
    cls = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef))
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    unit = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    namespace = dict(globals(), logger=logging.getLogger("contract-test"),
                     resolve_judge_metadata=lambda model, *args: {"judge_model": model})
    exec(compile(ast.fix_missing_locations(unit), str(source), "exec"), namespace)
    return namespace[name]


class Owner:
    retrieve = method("retrieve")
    rag_qa = method("rag_qa")

    def __init__(self):
        self.global_config = SimpleNamespace(retrieval_top_k=7)
        self.ready_to_retrieve = True
        self.all_retrieval_time = 0
        self.calls = []
        self.apply_repair = True

    def get_query_embeddings(self, queries):
        self.calls.append(("embedding", queries))

    def retrieve_full_once(self, **kwargs):
        self.calls.append(("round", kwargs))
        repair = kwargs["query"] == "bridge"
        docs = list("XYABC") if repair else list("ABCDEFG")
        scores = [0.5, 0.4, 0.3, 0.2, 0.1] if repair else [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        return {"final_top5": docs[:5], "final_docs": docs, "final_scores": scores, "repair_applied": self.apply_repair}

    def judge_answerability_and_bridge(self, **kwargs):
        self.calls.append(("judge", kwargs))
        return [{"can_answer": False, "bridge_possible": True, "bridge_question": "bridge", "evidence_docs": ["A"], "_judge_metadata": {"judge_model": "actual-judge"}} for q in kwargs["queries"]]

    def qa(self, queries):
        self.calls.append(("qa", queries))
        return queries, [], []


class PipelineContractTests(unittest.TestCase):
    def setUp(self):
        module = ModuleType("tqdm")
        module.tqdm = lambda iterable, **kwargs: iterable
        self.patch = patch.dict("sys.modules", {"tqdm": module})
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_empty_queries_do_not_call_any_backend(self):
        owner = Owner()
        self.assertEqual(owner.retrieve([]), [])
        self.assertEqual(owner.rag_qa([]), ([], [], []))
        self.assertEqual(owner.rag_qa([], gold_answers=[]), ([], [], [], {}, {}))
        self.assertEqual(owner.calls, [])

    def test_final_scores_follow_second_round_docs_and_first_round_tail(self):
        owner = Owner()
        result = owner.retrieve(["question"])[0]
        self.assertEqual(result.docs, list("XYABCFG"))
        self.assertEqual(result.doc_scores, [0.5, 0.4, 0.3, 0.2, 0.1, 0.4, 0.3])

    def test_agent_is_only_passed_to_second_round_with_original_question(self):
        owner = Owner()
        owner.retrieve(["question"], graph_search_mode="hybrid", agent_budget={"max_steps": 4})
        rounds = [payload for event, payload in owner.calls if event == "round"]
        self.assertNotIn("graph_search_mode", rounds[0])
        self.assertEqual(rounds[1]["graph_search_mode"], "hybrid")
        self.assertEqual(rounds[1]["original_query"], "question")
        self.assertEqual(rounds[1]["round_index"], 2)

    def test_failed_repair_keeps_first_round_and_scores(self):
        owner = Owner()
        owner.apply_repair = False
        result = owner.retrieve(["question"], graph_search_mode="agent", agent_fallback="none")[0]
        self.assertEqual(result.docs, list("ABCDEFG"))
        self.assertEqual(result.doc_scores[:5], [0.9, 0.8, 0.7, 0.6, 0.5])

    def test_qa_forwards_all_retrieval_options(self):
        owner = Owner()
        options = {"num_to_retrieve": 12, "subject_cap": 2, "judge_model_name": "selected", "graph_search_mode": "agent", "agent_budget": {"max_steps": 4}, "agent_fallback": "none", "result_save_root": "result-root"}
        captured = []
        owner.retrieve = lambda **kwargs: captured.append(kwargs) or []
        owner.rag_qa(["q"], **options)
        for key, value in options.items():
            self.assertEqual(captured[0][key], value)

    def test_prebuilt_solutions_skip_retrieval(self):
        owner = Owner()
        queries = [QuerySolution("q", ["d"])]
        self.assertEqual(owner.rag_qa(queries)[0], queries)
        self.assertEqual([event for event, _ in owner.calls], ["qa"])

    def test_replay_uses_frozen_round_and_actual_judge_metadata(self):
        owner = Owner()
        owner.retrieve(["question"])
        frozen = owner.last_retrieval_trace
        owner.calls = []
        _, metrics = owner.retrieve(["question"], gold_docs=[["X"]], round1_replay=frozen, judge_model_name="different-request")
        self.assertEqual(len([event for event, _ in owner.calls if event == "round"]), 1)
        self.assertNotIn("judge", [event for event, _ in owner.calls])
        self.assertEqual(metrics["judge_model"], "actual-judge")
        self.assertEqual(metrics["requested_judge_metadata"]["judge_model"], "different-request")

    def test_rejects_mismatched_gold_or_replay(self):
        owner = Owner()
        with self.assertRaises(ValueError):
            owner.retrieve(["question"], gold_docs=[])
        with self.assertRaises(ValueError):
            owner.rag_qa(["question"], gold_answers=[])
        with self.assertRaises(ValueError):
            owner.retrieve(["question"], round1_replay=[])
