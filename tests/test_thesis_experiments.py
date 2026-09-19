"""No-network tests for the experimental controls and durable runner contract."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from SRgraphrag.experiments.controls import choose_ids, mask_predicates, PredicateMaskedLLM, enumerate_paths, static_graph_search, rerank_passages
from SRgraphrag.graph.relation_index import RelationIndex
from SRgraphrag.graph.schema import entity_id, passage_id
from SRgraphrag.retrieval.types import GraphSearchRequest, GraphSearchResult

spec = importlib.util.spec_from_file_location("thesis_runner", ROOT / "scripts/run_thesis_experiments.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ControlsTests(unittest.TestCase):
    def setUp(self):
        self.index = RelationIndex.from_openie([
            {"passage": "A links B. B founded C.", "extracted_triples": [["A", "links", "B"], ["B", "founded", "C"]]},
            {"passage": "Protected", "extracted_triples": []},
            {"passage": "Unrelated X links Y", "extracted_triples": [["X", "links", "Y"]]},
        ])
        self.request = GraphSearchRequest("Who founded C?", "C founder", 2, [], [entity_id("A")], {}, [passage_id("Protected")])

    def test_strict_one_shot_selection(self):
        for text in ('{"ids":["missing"]}', '{"ids":["A","A"]}', '{"ids":[],"extra":1}',
                     '{"ids":["A"],"ids":[]}', '```json\n{"ids":["A"]}\n```'):
            ids, _, _, error = choose_ids(lambda _: text, {}, ["A"], field="ids", maximum=2)
            self.assertEqual(ids, [])
            self.assertIsNotNone(error)
        ids, usage, _, error = choose_ids(lambda _: ('{"ids":["A"]}', {"prompt_tokens": 100, "completion_tokens": 10}, False),
                                         {}, ["A"], field="ids", maximum=2)
        self.assertEqual(ids, ["A"])
        self.assertEqual(usage["total_tokens"], 110)
        self.assertIsNone(error)

    def test_predicate_mask_does_not_modify_index_or_source_text(self):
        source = {"facts": [{"predicate": "founded", "raw_triple_variants": [["B", "founded", "C"]]}], "content": "B founded C"}
        original = deepcopy(source)
        masked = mask_predicates(source)
        self.assertEqual(source, original)
        self.assertNotEqual(masked["facts"][0]["predicate"], "founded")
        self.assertNotIn("founded", str(masked["facts"]))
        self.assertEqual(masked["content"], "B founded C")

    def test_selectable_ids_are_explicit_and_exclude_context_only_ids(self):
        seen = []
        def select(messages):
            seen.append(messages)
            return '{"ids":["protected-only"]}'
        ids, _, event, error = choose_ids(select, {
            "protected_passages": [{"id": "protected-only", "content": "Already retained"}],
            "selectable_ids": ["must-not-override-allowlist"],
        }, ["candidate"], field="ids", maximum=5)
        self.assertEqual(json.loads(seen[0][1]["content"])["selectable_ids"], ["candidate"])
        self.assertIn("already retained", seen[0][0]["content"])
        self.assertEqual(ids, [])
        self.assertEqual(error, "ValueError")
        self.assertEqual(event["selection_error"], "invalid_selection_ids")

    def test_rerank_aliases_map_back_without_exposing_protected_only_ids(self):
        protected = self.request.protected_passage_ids[0]
        owner = SimpleNamespace(chunk_embedding_store=SimpleNamespace(get_row=lambda pid: {"content": "Text " + str(pid==protected)}))
        def select(messages):
            context = json.loads(messages[1]["content"])
            self.assertEqual(context["selectable_ids"], ["D0", "D1"])
            self.assertNotIn(protected, messages[1]["content"])
            self.assertNotIn("candidate-a", messages[1]["content"])
            return '{"passage_ids":["D1"]}'
        result = rerank_passages(owner, self.request, GraphSearchResult(ranked_passage_ids=["candidate-a", "candidate-b"]), select)
        self.assertEqual(result.ranked_passage_ids, ["candidate-b", "candidate-a"])
        self.assertEqual(result.trace[0]["passage_id_map"], {"D0": "candidate-a", "D1": "candidate-b"})
        self.assertIsNone(result.fallback_reason)

    def test_masked_adapter_preserves_limits_and_opaque_ids(self):
        received = []
        wrapped = PredicateMaskedLLM(lambda messages: received.append(messages) or "ok")
        wrapped([{"role": "system", "content": "system"}, {"role": "user", "content": json.dumps({
            "predicate": "links", "fact_id": "f1", "llm_limits": {"max_completion_tokens": 1024}})}])
        state = json.loads(received[0][1]["content"])
        self.assertEqual(state["fact_id"], "f1")
        self.assertEqual(state["llm_limits"]["max_completion_tokens"], 1024)

    def test_paths_are_seed_rooted_and_simple(self):
        paths = enumerate_paths(self.index, set(self.index.facts), [entity_id("A")], 4)
        self.assertTrue(any(len(path) == 2 for path in paths))
        for path in paths:
            self.assertEqual(path[0]["from_entity_id"], entity_id("A"))
            nodes = [path[0]["from_entity_id"]] + [step["to_entity_id"] for step in path]
            self.assertEqual(len(nodes), len(set(nodes)))
            self.assertNotIn(entity_id("X"), nodes)

    def test_static_uses_one_llm_call_and_real_source_commit(self):
        calls = []
        def select(messages):
            calls.append(messages)
            context = json.loads(messages[1]["content"])
            candidate = max(context["paths"], key=lambda item: len(item["steps"]))
            return json.dumps({"path_ids": [candidate["id"]]}), {"prompt_tokens": 100, "completion_tokens": 10}, False
        result = static_graph_search(self.index, self.request, select)
        self.assertEqual(len(calls), 1)
        self.assertTrue(result.selected_paths)
        self.assertEqual(set(result.ranked_passage_ids), {passage_id("Protected"), passage_id("A links B. B founded C.")})
        self.assertLessEqual(result.usage["expanded_edges"], 64)
        self.assertLessEqual(result.usage["tool_calls"], 8)

    def test_static_unknown_selection_fails_closed(self):
        result = static_graph_search(self.index, self.request, lambda _: '{"path_ids":["invented"]}')
        self.assertEqual(result.selected_paths, [])
        self.assertEqual(result.ranked_passage_ids, [])

    def test_atomic_checkpoint_and_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "result.json"
            runner.save(path, {"query": "q", "complete": True})
            self.assertEqual(runner.read(path), {"query": "q", "complete": True})
            self.assertFalse(path.with_suffix(".json.tmp").exists())
            with runner.lock(Path(temp) / "lock"):
                with self.assertRaises(RuntimeError):
                    with runner.lock(Path(temp) / "lock"):
                        pass

    def test_frozen_requests_ignore_budget_but_not_seeds(self):
        before = runner.frozen_request(self.request)
        self.request.budget["max_steps"] = 4
        self.assertEqual(before, runner.frozen_request(self.request))
        self.request.seed_entity_ids.append(entity_id("X"))
        self.assertNotEqual(before, runner.frozen_request(self.request))

    def test_manifest_default_excludes_development(self):
        args = runner.parser().parse_args(["--run-dir", "unused"])
        self.assertEqual((args.start, args.stop), (20, 1000))
        self.assertFalse(args.live)
        self.assertIsNone(runner.BUDGET["max_tokens"])


if __name__ == "__main__":
    unittest.main()
