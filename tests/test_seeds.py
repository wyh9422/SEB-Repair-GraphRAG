from hashlib import md5
import unittest

from src.SRgraphrag.retrieval.seeds import build_fact_seeds


def eid(value):
    return "entity-" + md5(value.encode()).hexdigest()


class SeedTests(unittest.TestCase):
    def test_occurrence_average_and_document_frequency(self):
        ids, weights, missing = build_fact_seeds(
            [("a", "r", "b"), ("a", "r", "c")], [0, 1], [0.8, 0.4],
            [eid(x) for x in "abc"], {eid("a"): {"d1", "d2"}}, 0,
        )
        self.assertAlmostEqual(weights[eid("a")], 0.3)
        self.assertEqual(weights[eid("b")], 0.8)
        self.assertEqual(len(ids), 2)
        self.assertEqual(missing, [])

    def test_missing_entities_and_zero_scores_never_generate_nan(self):
        _, weights, missing = build_fact_seeds([("a", "r", "b")], [0], [0], [eid("a")], {}, 0)
        self.assertEqual(weights, {})
        self.assertEqual(missing, [eid("b")])

    def test_top_k_and_invalid_score(self):
        _, weights, _ = build_fact_seeds([("a", "r", "b")], [0], [float("nan")], [eid("a"), eid("b")], {}, 1)
        self.assertEqual(weights, {})
        _, weights, _ = build_fact_seeds([("a", "r", "b")], [0], [1], [eid("a"), eid("b")], {}, 1)
        self.assertEqual(len(weights), 1)
