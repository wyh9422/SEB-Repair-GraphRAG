"""CPU integration checks; explicitly skipped without the scientific runtime."""

import importlib.util
import json
import unittest
from types import SimpleNamespace

RUNTIME = importlib.util.find_spec("numpy") is not None and importlib.util.find_spec("igraph") is not None


@unittest.skipUnless(RUNTIME, "numpy/igraph required; run on the validation server")
class RuntimeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import igraph as ig
        from src.SRgraphrag.SRgraphrag import SRgraphrag
        from src.SRgraphrag.graph.relation_index import RelationIndex
        from src.SRgraphrag.graph.schema import entity_id, fact_id, passage_id
        cls.np, cls.ig, cls.engine_type = np, ig, SRgraphrag
        cls.entity_id, cls.fact_id, cls.passage_id = staticmethod(entity_id), staticmethod(fact_id), staticmethod(passage_id)
        cls.records = [
            {"passage": "A is related to B.", "extracted_triples": [["a", "related to", "b"]]},
            {"passage": "B is located in C.", "extracted_triples": [["b", "located in", "c"]]},
        ]
        cls.index = RelationIndex.from_openie(cls.records)

    def make_owner(self):
        np = self.np
        owner = self.engine_type.__new__(self.engine_type)
        owner.global_config = SimpleNamespace(passage_node_weight=0.1, linking_top_k=0, damping=0.5, retrieval_top_k=5)
        owner.entity_node_keys = [self.entity_id(x) for x in "abc"]
        owner.passage_node_keys = [self.passage_id(row["passage"]) for row in self.records]
        names = owner.entity_node_keys + owner.passage_node_keys
        owner.graph = self.ig.Graph(n=len(names), edges=[(0, 1), (1, 2), (0, 3), (1, 3), (1, 4), (2, 4)])
        owner.graph.vs["name"] = names
        owner.graph.es["weight"] = [1.0] * 6
        owner.node_name_to_vertex_idx = {name: i for i, name in enumerate(names)}
        owner.passage_node_idxs = [3, 4]
        owner.ent_node_to_chunk_ids = {self.entity_id("a"): {owner.passage_node_keys[0]}, self.entity_id("b"): set(owner.passage_node_keys), self.entity_id("c"): {owner.passage_node_keys[1]}}
        owner.ppr_time = owner.rerank_time = owner.all_retrieval_time = 0
        owner._graph_validation_error = None
        owner._relation_index = self.index
        rows = {self.passage_id(row["passage"]): {"content": row["passage"]} for row in self.records}
        owner.chunk_embedding_store = SimpleNamespace(get_row=lambda key: rows[key])
        owner.dense_passage_retrieval = lambda query: (np.array([0, 1]), np.array([0.9, 0.2]))
        owner.get_fact_scores = lambda query: np.array([0.8])
        owner.rerank_facts = lambda *args, **kwargs: ([0], [("a", "related to", "b")], {"facts_before_rerank": [("a", "related to", "b")], "facts_after_rerank": [("a", "related to", "b")]})
        owner.ready_to_retrieve = True
        owner.get_query_embeddings = lambda queries: None
        return owner

    def test_adapter_matches_direct_ppr_and_legacy_reset_formula(self):
        owner = self.make_owner()
        ids, scores = owner.graph_search_with_fact_entities("q", 0, self.np.array([0.8]), [("a", "related to", "b")], [0], 0.1)
        reset = self.np.array([0.8, 0.4, 0, 0.1, 0])
        expected = owner.graph.personalized_pagerank(damping=0.5, directed=False, weights="weight", reset=reset, implementation="prpack")
        passage_scores = self.np.array(expected)[[3, 4]]
        expected_ids = self.np.argsort(passage_scores)[::-1]
        self.np.testing.assert_array_equal(ids, expected_ids)
        self.np.testing.assert_allclose(scores, passage_scores[expected_ids], rtol=1e-12)

    def test_missing_entity_and_zero_filter_do_not_create_nan(self):
        owner = self.make_owner()
        _, scores = owner.graph_search_with_fact_entities("q", 0, self.np.array([0.8]), [("missing", "r", "b")], [0])
        self.assertTrue(self.np.all(self.np.isfinite(scores)))

    def test_invalid_graph_uses_recorded_dense_fallback(self):
        owner = self.make_owner()
        owner._graph_validation_error = "missing_graph_edges"
        result = owner.retrieve_full_once("q", num_to_retrieve=5)
        self.assertEqual(result["final_source"], "dpr_fallback")
        self.assertEqual(result["graph_search"]["fallback_reason"], "missing_graph_edges")
        json.dumps(result, allow_nan=False)

    def test_agent_path_survives_final_evidence_policy_and_hybrid(self):
        for mode in ("agent", "hybrid"):
            with self.subTest(mode=mode):
                owner = self.make_owner()
                a, b, c = [self.entity_id(x) for x in "abc"]
                steps = [
                    {"fact_id": self.fact_id(("a", "related to", "b")), "from_entity_id": a, "to_entity_id": b, "traversal_direction": "out"},
                    {"fact_id": self.fact_id(("b", "located in", "c")), "from_entity_id": b, "to_entity_id": c, "traversal_direction": "out"},
                ]
                actions = iter([
                    {"action": "expand_entity", "arguments": {"entity_id": b, "direction": "both"}},
                    {"action": "commit_paths", "arguments": {"paths": [steps]}},
                ])
                owner._agent_llm = lambda messages: (json.dumps(next(actions)), {"prompt_tokens": 500, "completion_tokens": 100}, False)
                result = owner.retrieve_full_once("repair", original_query="original", round_index=2, num_to_retrieve=5, evidence=[self.records[0]["passage"]], graph_search_mode=mode)
                self.assertEqual(result["final_source"], "agent", result["graph_search"])
                self.assertEqual(len(result["graph_search"]["selected_paths"]), 1)
                self.assertTrue(set(row["passage"] for row in self.records).issubset(result["final_top5"]))
                self.assertTrue(all(score is None for score in result["final_scores"][:2]))
                json.dumps(result, allow_nan=False)

    def test_agent_failure_without_fallback_keeps_first_round(self):
        owner = self.make_owner()
        owner._agent_llm = lambda messages: ('{"action":"stop","arguments":{"reason":"no path"}}', {"prompt_tokens": 100, "completion_tokens": 10}, False)
        owner.judge_answerability_and_bridge = lambda **kwargs: [{"can_answer": False, "bridge_possible": True, "bridge_question": "repair", "evidence_docs": []}]
        result = owner.retrieve(["original"], graph_search_mode="agent", agent_fallback="none")
        self.assertEqual(result[0].docs[:5], owner.last_retrieval_trace[0]["round1"]["final_top5"])
        self.assertFalse(owner.last_retrieval_trace[0]["round2"]["repair_applied"])

    def test_empty_dense_corpus_does_not_encode(self):
        owner = self.make_owner()
        owner.passage_embeddings = self.np.array([])
        ids, scores = self.engine_type.dense_passage_retrieval(owner, "q")
        self.assertEqual(len(ids), 0)
        self.assertEqual(len(scores), 0)

    def test_empty_corpus_does_not_call_embedding_filter_or_judge(self):
        owner = self.make_owner()
        owner.passage_node_keys = []
        def forbidden(*args, **kwargs):
            raise AssertionError("empty corpus called a backend")
        owner.get_query_embeddings = forbidden
        owner.get_fact_scores = forbidden
        owner.dense_passage_retrieval = forbidden
        owner.judge_answerability_and_bridge = forbidden
        result = owner.retrieve(["q"])
        self.assertEqual(result[0].docs, [""] * 5)
        self.assertEqual(result[0].doc_scores, [None] * 5)

    def test_nonfinite_graph_weights_fall_back_explicitly(self):
        owner = self.make_owner()
        owner.graph.es[0]["weight"] = float("nan")
        result = owner.retrieve_full_once("q", num_to_retrieve=5)
        self.assertEqual(result["graph_search"]["fallback_reason"], "invalid_graph_weights")

    def test_nonfinite_dense_fallback_scores_still_serialize_as_null(self):
        owner = self.make_owner()
        owner.dense_passage_retrieval = lambda query: (self.np.array([0, 1]), self.np.array([float("nan"), 0.2]))
        result = owner.retrieve_full_once("q", num_to_retrieve=5)
        self.assertEqual(result["graph_search"]["fallback_reason"], "nonfinite_dense_scores")
        self.assertIsNone(result["graph_search"]["scores"][0])
        json.dumps(result, allow_nan=False)
