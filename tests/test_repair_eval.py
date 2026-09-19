"""Offline repair metrics, trace alignment and cost accounting regression tests."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "SRgraphrag"))
from evaluation.repair_eval import aggregate_usage, evaluate_repair, load_gold, load_trace


def row(query, first, final, second=None, mode="agent"):
    return {
        "query": query, "round1": {"query": query, "final_top5": first, "graph_search": None},
        "judge1": None, "round2": second, "final_top5": final,
        "final_docs": final, "final_scores": [None] * len(final),
        "score_round": 2 if second else 1, "graph_search_mode": mode,
    }


def agent_round(path_docs=(), usage=None, committed=True, fallback=False):
    return {
        "graph_search_mode": "agent", "repair_applied": True,
        "path_protected_docs": list(path_docs),
        "graph_search": {
            "selected_paths": [[{"fact_id": "fixture-fact"}]] if committed else [],
            "stop_reason": "fallback_ppr_complete" if fallback else "committed",
            "fallback_reason": "tool_budget" if fallback else None,
            "usage": usage if usage is not None else {"llm_calls": 2, "tool_calls": 2, "elapsed_seconds": 1},
            "trace": [{"event": "fallback", "used": "ppr_complete", "reason": "tool_budget"}] if fallback else [],
        },
    }


class RepairEvalTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            row("repaired", ["A"], ["A", "B"], agent_round(["B"])),
            row("damaged", ["A", "B"], ["A", "C"], agent_round(["B"])),
            row("failed", ["C"], ["C"], agent_round(committed=False, fallback=True, usage={
                "agent": {"llm_calls": 3, "tool_calls": 2, "elapsed_seconds": 2, "total_tokens": 30},
                "fallback": {"ppr_seconds": 0.5},
            })),
            row("unlabeled", [], []),
        ]
        self.gold = [{"query": item["query"], "gold_docs": ["A", "B"] if index < 3 else None}
                     for index, item in enumerate(self.rows)]

    def test_quality_and_repair_denominators_keep_failures_and_exclude_only_empty_gold(self):
        result = evaluate_repair(self.rows, self.gold)
        self.assertEqual(result["total_queries"], 4)
        self.assertEqual(result["empty_gold_queries"], 1)
        for stage in ("round1", "final"):
            self.assertEqual(result["quality"][stage]["avg_recall@5"], 0.5)
            self.assertEqual(result["quality"][stage]["full_recall@5"], 1 / 3)
            self.assertEqual(result["quality"][stage]["denominator"], 3)
        self.assertEqual(result["repair"]["miss1_to_miss0"], {"numerator": 1, "denominator": 1, "rate": 1})
        self.assertEqual(result["repair"]["miss0_damage"]["rate"], 1)
        self.assertEqual(result["coverage"]["agent_executed"]["denominator"], 4)
        self.assertEqual(result["usage"]["llm_calls"]["sum"], 7)
        self.assertEqual(len(result["per_query"]), 4)

    def test_path_retention_requires_committed_paths_and_nonempty_sources(self):
        result = evaluate_repair(self.rows, self.gold)
        self.assertEqual(result["coverage"]["committed_path_set_retention"], {"numerator": 1, "denominator": 2, "rate": 0.5})
        incomplete = row("no-source", ["A"], ["A"], agent_round())
        missing = evaluate_repair([incomplete], [["A"]])
        self.assertEqual(missing["coverage"]["commits_missing_source_docs"], 1)
        self.assertEqual(missing["coverage"]["committed_path_set_retention"]["rate"], 0)
        no_paths = row("no-path", ["A"], ["A"], agent_round(["A"], committed=False))
        self.assertEqual(evaluate_repair([no_paths], [["A"]])["coverage"]["committed_path_set_retention"]["denominator"], 0)

    def test_fallback_reason_does_not_alone_mean_a_fallback_executed(self):
        second = agent_round(committed=False)
        second["repair_applied"] = False
        second["graph_search"].update(stop_reason="agent_stop", fallback_reason="agent_stop")
        result = evaluate_repair([row("stopped", ["A"], ["A"], second)], [["A", "B"]])
        self.assertEqual(result["coverage"]["round2_fallback"]["numerator"], 0)
        result = evaluate_repair(self.rows, self.gold)
        self.assertEqual(result["coverage"]["round2_fallback"]["denominator"], 3)
        self.assertEqual(result["fallback_reasons_by_round"], {"tool_budget": 1})

    def test_configuration_failure_is_requested_but_not_executed(self):
        second = agent_round(committed=False, usage={"llm_calls": 0})
        result = evaluate_repair([row("config", ["A"], ["A"], second)], [["A"]])
        self.assertEqual(result["coverage"]["agent_requested"]["numerator"], 1)
        self.assertEqual(result["coverage"]["agent_executed"]["numerator"], 0)

    def test_nested_usage_sums_components_without_adding_token_budget_to_tokens(self):
        result = aggregate_usage({
            "agent": {"llm_calls": 2, "prompt_tokens": 10, "completion_tokens": 5,
                      "estimated_tokens": 7, "budget_tokens": 22, "elapsed_seconds": 3},
            "fallback": {"ppr_seconds": 0.2},
        })
        self.assertEqual(result["counters"]["total_tokens"], 15)
        self.assertEqual(result["counters"]["budget_tokens"], 22)
        self.assertEqual(result["measured_graph_seconds"], 3.2)

    def test_hybrid_prior_reuse_is_not_double_counted_and_distinct_runs_are(self):
        usage = {"agent": {"llm_calls": 2, "elapsed_seconds": 2, "hybrid_ppr": {"ppr_seconds": 1}},
                 "fallback": {"ppr_seconds": 1}, "fallback_reused_hybrid_ppr": True}
        self.assertEqual(aggregate_usage(usage)["measured_graph_seconds"], 3)
        usage["fallback_reused_hybrid_ppr"] = False
        self.assertEqual(aggregate_usage(usage)["measured_graph_seconds"], 4)
        del usage["fallback_reused_hybrid_ppr"]
        legacy = aggregate_usage(usage, fallback_events=[{"used": "ppr_complete"}])
        self.assertEqual(legacy["measured_graph_seconds"], 3)
        self.assertIn("hybrid_prior_reuse_inferred_from_legacy_trace", legacy["warnings"])

    def test_wrapper_totals_are_not_added_again_to_their_children(self):
        result = aggregate_usage({"llm_calls": 3, "elapsed_seconds": 7,
                                  "agent": {"llm_calls": 3, "elapsed_seconds": 5},
                                  "fallback": {"ppr_seconds": 1}})
        self.assertEqual(result["counters"]["llm_calls"], 3)
        self.assertEqual(result["measured_graph_seconds"], 7)

    def test_cached_tokens_are_distinguished_from_uncached_response_tokens(self):
        second = agent_round(["A"], {"llm_calls": 2, "cache_hits": 1, "total_tokens": 30})
        second["graph_search"]["trace"] = [
            {"event": "llm_response", "tokens_reported": True, "cache_hit": True, "tokens": {"total_tokens": 10}},
            {"event": "llm_response", "tokens_reported": True, "cache_hit": False, "tokens": {"total_tokens": 20}},
        ]
        result = evaluate_repair([row("cached", ["A"], ["A"], second)], [["A"]])
        self.assertEqual(result["usage"]["total_tokens"]["sum"], 30)
        self.assertEqual(result["usage"]["uncached_reported_tokens"]["sum"], 20)
        self.assertEqual(result["usage"]["noncache_llm_attempts"]["sum"], 1)

    def test_missing_cache_status_does_not_invent_uncached_token_cost(self):
        second = agent_round(["A"], {"llm_calls": 1, "total_tokens": 10})
        second["graph_search"]["trace"] = [
            {"event": "llm_response", "tokens_reported": True, "cache_hit": None, "tokens": {"total_tokens": 10}}
        ]
        result = evaluate_repair([row("unknown-cache", ["A"], ["A"], second)], [["A"]])
        self.assertIsNone(result["usage"]["uncached_reported_tokens"]["sum"])

    def test_missing_usage_and_empty_denominators_are_null_not_zero(self):
        result = evaluate_repair([row("empty", [], [])], [[]])
        self.assertIsNone(result["quality"]["final"]["avg_recall@5"])
        self.assertIsNone(result["repair"]["miss1_to_miss0"]["rate"])
        self.assertIsNone(result["usage"]["llm_calls"]["sum"])
        self.assertEqual(result["usage"]["llm_calls"]["missing_queries"], 1)
        json.dumps(evaluate_repair([], []), allow_nan=False)

    def test_duplicate_gold_is_not_counted_twice_and_empty_retrieval_is_a_failure(self):
        result = evaluate_repair([row("dup", ["A"], [])], [["A", "A", "B"]])
        self.assertEqual(result["quality"]["round1"]["avg_recall@5"], 0.5)
        self.assertEqual(result["quality"]["final"]["avg_recall@5"], 0)
        self.assertEqual(result["repair"]["miss1_to_miss0"]["denominator"], 1)

    def test_rank_cutoff_precedes_removal_of_padding_documents(self):
        result = evaluate_repair([row("padding", ["A"], ["", "B", "C", "D", "E", "A"])], [["A"]])
        self.assertEqual(result["quality"]["final"]["avg_recall@5"], 0)

    def test_query_order_length_and_mixed_gold_formats_are_rejected(self):
        for gold in (self.gold[:-1], list(reversed(self.gold)), [[], self.gold[1], [], []]):
            with self.assertRaises(ValueError):
                evaluate_repair(self.rows, gold)
        with self.assertRaisesRegex(ValueError, "Baseline query order"):
            evaluate_repair(self.rows, self.gold, list(reversed(self.rows)))
        positional = evaluate_repair(self.rows, [["A", "B"]] * 4)
        self.assertEqual(positional["gold_alignment"], "positional_length_checked")

    def test_nonfinite_values_are_rejected_even_in_optional_scores(self):
        invalid = row("nan", ["A"], ["A"])
        invalid["final_scores"] = [float("nan")]
        with self.assertRaises(ValueError):
            evaluate_repair([invalid], [["A"]])
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "bad.jsonl"
            target.write_text('{"value": Infinity}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_trace(target)
            with self.assertRaises(ValueError):
                load_gold(target)

    def test_paired_round1_baseline_uses_frozen_first_round_and_fixed_seed(self):
        rows = [row("one", [], ["A"]), row("two", [], ["B"])]
        gold = [["A"], ["B"]]
        first = evaluate_repair(rows, gold, baseline_mode="round1", bootstrap_samples=50, seed=7)
        second = evaluate_repair(rows, gold, baseline_mode="round1", bootstrap_samples=50, seed=7)
        comparison = first["paired_comparison"]
        self.assertEqual(comparison, second["paired_comparison"])
        self.assertEqual(comparison["metrics"]["avg_recall@5"]["bootstrap_ci95"], [1, 1])
        self.assertEqual(comparison["metrics"]["full_recall@5"]["wins"], 2)
        paired = evaluate_repair(self.rows, self.gold, self.rows, bootstrap_samples=20)
        self.assertEqual(paired["paired_comparison"]["paired_queries"], 3)
        self.assertEqual(paired["paired_comparison"]["metrics"]["avg_recall@5"]["mean_delta"], 0)

    def test_cli_is_standard_library_only_and_rejects_order_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            gold = Path(directory) / "gold.json"
            trace.write_text("".join(json.dumps(item) + "\n" for item in self.rows), encoding="utf-8")
            gold.write_text(json.dumps(self.gold), encoding="utf-8")
            command = [sys.executable, "-S", str(PROJECT_ROOT / "scripts" / "evaluate_repair.py"),
                       "--trace", str(trace), "--gold", str(gold), "--baseline-mode", "round1", "--bootstrap-samples", "10"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["paired_comparison"]["paired_queries"], 3)
            gold.write_text(json.dumps(list(reversed(self.gold))), encoding="utf-8")
            rejected = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("order mismatch", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
