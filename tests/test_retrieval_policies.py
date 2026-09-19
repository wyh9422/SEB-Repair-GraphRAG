"""Offline policy tests; importing the leaf package avoids loading model backends."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "SRgraphrag"))

from retrieval.evidence import (
    align_scores_to_top5,
    apply_evidence_injection_top5,
    apply_guard_top5_protect_evidence,
    dedup_preserve,
)
from retrieval.facts import build_subject_cap_instruction, count_unique_subjects
from retrieval.metrics import (
    all_recall_at_k,
    avg_recall_at_k,
    hit_at_1,
    missing_list,
    mrr_first_hit,
)


class EvidencePolicyTests(unittest.TestCase):
    def test_missing_evidence_then_dense_anchor_preserves_evidence(self):
        ranked = ["A", "B", "C", "D", "E", "F"]
        evidence = ["B", "X"]
        injected, log = apply_evidence_injection_top5(ranked[:5], evidence, ranked)
        self.assertEqual(injected, ["A", "B", "C", "D", "X"])
        self.assertEqual(log["evidence_missing"], ["X"])
        self.assertEqual(log["replaced_cnt"], 1)
        guarded, guard_log = apply_guard_top5_protect_evidence(injected, "Z", set(evidence))
        self.assertEqual(guarded, ["A", "B", "C", "Z", "X"])
        self.assertEqual(guard_log["guard_replaced_idx"], 3)
        self.assertEqual(ranked, ["A", "B", "C", "D", "E", "F"])
        self.assertEqual(evidence, ["B", "X"])

    def test_multiple_missing_evidence_uses_tail_first(self):
        result, log = apply_evidence_injection_top5(["A", "B", "C", "D", "E"], ["X", "Y"], [])
        self.assertEqual(result, ["A", "B", "C", "Y", "X"])
        self.assertEqual(log["replaced_cnt"], 2)

    def test_already_covered_evidence_does_not_replace(self):
        result, log = apply_evidence_injection_top5(["A", "B", "C", "D", "E"], ["B", "D"], [])
        self.assertEqual(result, ["A", "B", "C", "D", "E"])
        self.assertFalse(log["evidence_used"])
        self.assertEqual(log["replaced_cnt"], 0)

    def test_five_protected_passages_block_dense_anchor_replacement(self):
        evidence = ["A", "B", "C", "D", "E"]
        result, log = apply_guard_top5_protect_evidence(evidence, "X", set(evidence))
        self.assertEqual(result, evidence)
        self.assertFalse(log["guard_triggered"])

    def test_dense_anchor_already_present_is_noop(self):
        docs = ["A", "B", "C", "D", "E"]
        result, log = apply_guard_top5_protect_evidence(docs, "C", set())
        self.assertEqual(result, docs)
        self.assertFalse(log["guard_triggered"])

    def test_empty_candidates_are_padded(self):
        result, log = apply_evidence_injection_top5([], [], [])
        self.assertEqual(result, [""] * 5)
        self.assertFalse(log["evidence_enabled"])

    def test_duplicate_evidence_is_injected_once(self):
        result, log = apply_evidence_injection_top5(["A", "B", "C", "D", "E"], ["X", "", "X"], [])
        self.assertEqual(result.count("X"), 1)
        self.assertEqual(log["evidence_missing_cnt"], 1)

    def test_duplicate_candidates_are_refilled_from_base_ranking(self):
        result, _ = apply_evidence_injection_top5(["A", "A", "B"], None, ["A", "B", "C", "D", "E"])
        self.assertEqual(result, ["A", "B", "C", "D", "E"])

    def test_evidence_budget_is_five(self):
        result, _ = apply_evidence_injection_top5([], ["A", "B", "C", "D", "E", "F"], [])
        self.assertEqual(set(result), {"A", "B", "C", "D", "E"})

    def test_score_alignment_uses_document_identity(self):
        scores = align_scores_to_top5(["B", "X", "A", ""], ["A", "B", "A"], [0.9, 0.7, 0.1])
        self.assertEqual(scores[:1], [0.7])
        self.assertEqual(scores[2], 0.9)
        self.assertIsNone(scores[1])
        self.assertIsNone(scores[3])

    def test_score_alignment_accepts_missing_scores(self):
        self.assertEqual(align_scores_to_top5(["A"], ["A"], None), [None])

    def test_unknown_and_nonfinite_scores_are_json_null(self):
        import json
        scores = align_scores_to_top5(["A", "B", "C"], ["A", "B", "C"], [None, float("nan"), float("-inf")])
        self.assertEqual(json.dumps(scores, allow_nan=False), "[null, null, null]")

    def test_dedup_keeps_first_nonempty_occurrence(self):
        self.assertEqual(dedup_preserve(["B", "", "A", "B", "C", "A"]), ["B", "A", "C"])


class FactPolicyTests(unittest.TestCase):
    def test_subject_budget_is_clamped_to_legacy_bounds(self):
        self.assertEqual(build_subject_cap_instruction(-1), build_subject_cap_instruction(1))
        self.assertEqual(build_subject_cap_instruction(99), build_subject_cap_instruction(5))
        for count, word in enumerate(["one", "two", "three", "four", "five"], 1):
            prompt = build_subject_cap_instruction(count)
            self.assertIn("at most " + word + " key subjects", prompt)
            self.assertIn("does not exceed " + word, prompt)
            self.assertIn("You must only use facts from the candidate list", prompt)

    def test_subject_count_ignores_malformed_candidates(self):
        self.assertEqual(count_unique_subjects([["a", "r", "b"], ["a", "r2", "c"], ("d", "r", "e"), [], "bad"]), 2)
        self.assertEqual(count_unique_subjects(None), 0)


class RetrievalMetricTests(unittest.TestCase):
    def test_partial_coverage_is_distinct_from_full_recall(self):
        golds = [["A", "B"], ["C"], []]
        retrieved = [["A", "X"], ["C", "Y"], ["Z"]]
        self.assertEqual(avg_recall_at_k(golds, retrieved), 0.75)
        self.assertEqual(all_recall_at_k(golds, retrieved), 0.5)

    def test_empty_gold_sets_produce_zero(self):
        self.assertEqual(avg_recall_at_k([[]], [["A"]]), 0.0)
        self.assertEqual(all_recall_at_k([], []), 0.0)

    def test_missing_evidence_preserves_gold_order(self):
        self.assertEqual(missing_list(["C", "A", "B"], ["A", "X"]), ["C", "B"])

    def test_hit_and_reciprocal_rank(self):
        self.assertEqual(hit_at_1(["A"], ["X", "A"]), 0)
        self.assertEqual(hit_at_1(["A"], ["A", "X"]), 1)
        self.assertEqual(mrr_first_hit(["A"], ["X", "A"]), 0.5)
        self.assertEqual(mrr_first_hit([], ["A"]), 0.0)
        self.assertEqual(hit_at_1(["A"], []), 0)


if __name__ == "__main__":
    unittest.main()
