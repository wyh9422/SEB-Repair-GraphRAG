"""Offline graph Agent tests: real deterministic tools, scripted text-only LLM."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from SRgraphrag.graph.relation_index import RelationIndex
from SRgraphrag.graph.schema import entity_id, fact_id, passage_id
from SRgraphrag.retrieval.agent import AgentGraphSearch
from SRgraphrag.retrieval.agent_actions import ActionError, AgentBudget, parse_action
from SRgraphrag.retrieval.types import GraphSearchRequest


def action(name, **arguments):
    return json.dumps({"action": name, "arguments": arguments})


def step(subject, predicate, obj, reverse=False):
    return {"fact_id": fact_id((subject, predicate, obj)),
            "from_entity_id": entity_id(obj if reverse else subject),
            "to_entity_id": entity_id(subject if reverse else obj),
            "traversal_direction": "in" if reverse else "out"}


class ScriptedLLM:
    def __init__(self, outputs, metadata=True, cached=False):
        self.outputs = iter(outputs)
        self.calls = []
        self.metadata = metadata
        self.cached = cached

    def __call__(self, messages):
        self.calls.append(copy.deepcopy(messages))
        result = next(self.outputs)
        if isinstance(result, Exception):
            raise result
        if self.metadata:
            return result, {"prompt_tokens": 100, "completion_tokens": 20}, self.cached
        return result


class ActionSchemaTests(unittest.TestCase):
    def test_strict_json_shapes_and_unknown_fields(self):
        invalid = ["[]", "null", "42", "```json\n{}\n```", "{}",
                   '{"action":"stop","action":"plan","arguments":{}}',
                   '{"action":"stop","arguments":{},"extra":true}',
                   action("exec", code="print(1)"), action("plan", goal=[]),
                   action("expand_entity", entity_id="x", limit=True),
                   action("expand_entity", entity_id="x", cursor=-1),
                   action("expand_entity", entity_id="x", direction="sideways"),
                   action("expand_entity", entity_id="x", limit=9),
                   action("commit_paths", paths=[]),
                   action("commit_paths", paths=[[]]),
                   action("stop", reason="x", evidence_limit=500)]
        for response in invalid:
            with self.subTest(response=response), self.assertRaises(ActionError):
                parse_action(response, AgentBudget())

    def test_expansion_defaults_canonicalize_repeated_actions(self):
        expected = {"action": "expand_entity", "arguments": {
            "entity_id": "x", "direction": "both", "limit": 8, "cursor": 0}}
        self.assertEqual(parse_action(action("expand_entity", entity_id="x"), AgentBudget()), expected)

    def test_budget_rejects_boolean_nan_negative_and_unknown(self):
        for values in ({"max_steps": True}, {"max_seconds": float("nan")},
                       {"max_seconds": float("inf")}, {"max_tokens": -1},
                       {"max_neighbors": 0}, {"max_path_length": 0},
                       {"max_tokens": 1.5}, {"unbounded": True}):
            with self.subTest(values=values), self.assertRaises(ActionError):
                AgentBudget.from_values(values)

    def test_budget_merge_and_zero_steps(self):
        budget = AgentBudget.from_values({"max_steps": 4}, {"max_steps": 0})
        self.assertEqual(budget.max_steps, 0)
        self.assertEqual(budget.max_tokens, 12000)


class AgentRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            {"passage": "Ada wrote Book.", "extracted_triples": [["Ada", "wrote", "Book"]]},
            {"passage": "Book inspired Film.", "extracted_triples": [["Book", "inspired", "Film"]]},
            {"passage": "Noise is Dust.", "extracted_triples": [["Noise", "is", "Dust"]]},
        ] + [{"passage": "Protected passage " + str(i), "extracted_triples": []} for i in range(5)]
        self.index = RelationIndex.from_openie(self.documents)
        self.path = [step("Ada", "wrote", "Book"), step("Book", "inspired", "Film")]

    def request(self, **overrides):
        values = dict(original_query="Which film traces back to Ada's work?",
                      retrieval_query="What did Book inspire?", round_index=2,
                      seed_fact_ids=[fact_id(("Ada", "wrote", "Book"))],
                      seed_entity_ids=[entity_id("Ada"), entity_id("Noise")],
                      seed_scores={entity_id("Ada"): 0.9, entity_id("Noise"): 0.1},
                      protected_passage_ids=[], budget={})
        values.update(overrides)
        return GraphSearchRequest(**values)

    def success_actions(self):
        return [action("expand_entity", entity_id=entity_id("Ada"), direction="out"),
                action("expand_entity", entity_id=entity_id("Book"), direction="out"),
                action("commit_paths", paths=[self.path])]

    def test_success_commits_source_complete_chain_ignoring_noisy_seed(self):
        llm = ScriptedLLM(self.success_actions())
        result = AgentGraphSearch(self.index, llm).search(self.request())
        self.assertEqual(result.stop_reason, "committed")
        self.assertIsNone(result.fallback_reason)
        self.assertEqual(set(result.ranked_passage_ids), {
            passage_id("Ada wrote Book."), passage_id("Book inspired Film.")})
        self.assertEqual(result.selected_paths, [self.path])
        self.assertEqual(result.scores, [None, None])
        self.assertEqual(result.score_sources, ["agent_path", "agent_path"])
        self.assertEqual(result.usage["llm_calls"], 3)
        self.assertEqual(result.usage["total_tokens"], 360)
        first_context = json.loads(llm.calls[0][1]["content"])
        self.assertEqual(first_context["original_query"], self.request().original_query)
        self.assertEqual(first_context["retrieval_query"], self.request().retrieval_query)
        self.assertIn("source_fingerprint", first_context["index_manifest"])
        self.assertIn("llm_limits", json.loads(llm.calls[0][-1]["content"]))
        self.assertNotIn("gold", first_context)

    def test_comparison_can_commit_two_disconnected_branches(self):
        branches = [[self.path[0]], [step("Noise", "is", "Dust")]]
        outputs = [action("expand_entity", entity_id=entity_id("Ada")),
                   action("expand_entity", entity_id=entity_id("Noise")),
                   action("commit_paths", paths=branches)]
        result = AgentGraphSearch(self.index, ScriptedLLM(outputs)).search(self.request())
        self.assertEqual(result.stop_reason, "committed")
        self.assertEqual(result.selected_paths, branches)

    def test_large_hybrid_region_does_not_exhaust_default_prompt_budget(self):
        records = self.documents + [
            {"passage": "scope " + str(i),
             "extracted_triples": [["scope " + str(i), "links", "leaf " + str(i)]]}
            for i in range(256)]
        index = RelationIndex.from_openie(records)
        llm = ScriptedLLM(self.success_actions())
        request = self.request(candidate_entity_ids=list(index.entities))
        result = AgentGraphSearch(index, llm).search(request)
        self.assertEqual(result.stop_reason, "committed")
        self.assertEqual(len(llm.calls), 3)
        context = json.loads(llm.calls[0][1]["content"])
        self.assertGreater(context["candidate_region"]["entity_count"], 512)
        self.assertNotIn("candidate_entity_ids", context)
        self.assertEqual(result.trace[0]["candidate_entity_ids"], list(index.entities))
        replayed = AgentGraphSearch(index, None).replay(request, result.trace)
        self.assertEqual(replayed.selected_paths, result.selected_paths)

    def test_protected_evidence_survives_path_commit(self):
        protected = [passage_id("Protected passage 0")]
        result = AgentGraphSearch(self.index, ScriptedLLM(self.success_actions())).search(
            self.request(protected_passage_ids=protected))
        self.assertEqual(result.stop_reason, "committed")
        self.assertTrue(set(protected) <= set(result.ranked_passage_ids))
        self.assertEqual(len(result.ranked_passage_ids), 3)

    def test_capacity_conflict_never_returns_partial_path(self):
        protected = [passage_id("Protected passage " + str(i)) for i in range(4)]
        result = AgentGraphSearch(self.index, ScriptedLLM(self.success_actions())).search(
            self.request(protected_passage_ids=protected, budget={"max_retries": 0}))
        self.assertNotEqual(result.stop_reason, "committed")
        self.assertEqual(result.selected_paths, [])
        self.assertEqual(result.ranked_passage_ids, [])
        self.assertEqual(result.usage["tool_failures"], 1)

    def test_observation_without_commit_returns_no_evidence(self):
        outputs = self.success_actions()[:2] + [action("stop", reason="Evidence insufficient")]
        result = AgentGraphSearch(self.index, ScriptedLLM(outputs)).search(self.request())
        self.assertEqual(result.stop_reason, "agent_stop")
        self.assertEqual(result.ranked_passage_ids, [])

    def test_format_error_can_be_repaired_within_budget(self):
        outputs = ["not JSON"] + self.success_actions()
        result = AgentGraphSearch(self.index, ScriptedLLM(outputs)).search(self.request())
        self.assertEqual(result.stop_reason, "committed")
        self.assertEqual(result.usage["invalid_actions"], 1)
        self.assertEqual(result.usage["retries"], 1)
        self.assertEqual(result.usage["llm_calls"], 4)

    def test_illegal_action_never_executes_model_code(self):
        llm = ScriptedLLM([action("python", code="raise Exception()")])
        result = AgentGraphSearch(self.index, llm).search(self.request(budget={"max_retries": 0}))
        self.assertEqual(result.stop_reason, "invalid_action")
        self.assertEqual(result.usage["tool_calls"], 0)
        self.assertEqual(result.selected_paths, [])

    def test_unobserved_fact_cannot_be_committed_directly(self):
        result = AgentGraphSearch(self.index, ScriptedLLM([action("commit_paths", paths=[self.path])])).search(
            self.request(budget={"max_retries": 0}))
        self.assertIsNotNone(result.fallback_reason)
        self.assertEqual(result.selected_paths, [])
        self.assertEqual(result.usage["tool_failures"], 1)

    def test_unknown_entity_returns_structured_failure(self):
        result = AgentGraphSearch(self.index, ScriptedLLM([action("expand_entity", entity_id="invented")])).search(
            self.request(budget={"max_retries": 0}))
        self.assertIsNotNone(result.fallback_reason)
        self.assertEqual(result.selected_paths, [])
        self.assertEqual(result.usage["tool_failures"], 1)

    def test_repeated_action_is_bounded_without_duplicate_tool_execution(self):
        command = action("expand_entity", entity_id=entity_id("Ada"))
        result = AgentGraphSearch(self.index, ScriptedLLM([command] * 3)).search(
            self.request(budget={"max_retries": 1}))
        self.assertEqual(result.stop_reason, "repeated_action")
        self.assertEqual(result.usage["tool_calls"], 1)
        self.assertEqual(result.usage["llm_calls"], 3)

    def test_zero_budgets_do_not_call_the_model(self):
        for budget, reason in [({"max_steps": 0}, "step_budget"),
                               ({"max_tokens": 0}, "token_budget"),
                               ({"max_seconds": 0}, "time_budget"),
                               ({"max_tool_calls": 0}, "tool_budget")]:
            with self.subTest(budget=budget):
                llm = ScriptedLLM([])
                result = AgentGraphSearch(self.index, llm).search(self.request(budget=budget))
                self.assertEqual(result.stop_reason, reason)
                self.assertEqual(llm.calls, [])

    def test_invalid_budget_is_structured_and_no_model_call(self):
        llm = ScriptedLLM([])
        result = AgentGraphSearch(self.index, llm).search(self.request(budget={"max_steps": True}))
        self.assertEqual(result.stop_reason, "invalid_configuration")
        self.assertEqual(llm.calls, [])

    def test_tool_budget_stops_before_another_model_call(self):
        llm = ScriptedLLM(self.success_actions())
        result = AgentGraphSearch(self.index, llm).search(self.request(budget={"max_tool_calls": 1}))
        self.assertEqual(result.stop_reason, "tool_budget")
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(result.selected_paths, [])

    def test_step_budget_keeps_uncommitted_results_empty(self):
        result = AgentGraphSearch(self.index, ScriptedLLM(self.success_actions())).search(
            self.request(budget={"max_steps": 2}))
        self.assertEqual(result.stop_reason, "step_budget")
        self.assertEqual(result.selected_paths, [])

    def test_missing_token_metadata_is_conservatively_counted(self):
        llm = ScriptedLLM([action("stop", reason="No path")], metadata=False)
        result = AgentGraphSearch(self.index, llm).search(self.request())
        self.assertEqual(result.stop_reason, "agent_stop")
        self.assertGreater(result.usage["estimated_tokens"], 0)
        self.assertEqual(result.usage["total_tokens"], 0)
        self.assertGreater(result.usage["budget_tokens"], 0)

    def test_cached_tokens_and_calls_are_separately_recorded(self):
        result = AgentGraphSearch(self.index, ScriptedLLM(self.success_actions(), cached=True)).search(self.request())
        self.assertEqual(result.usage["cache_hits"], 3)
        self.assertEqual(result.usage["llm_calls"], 3)
        self.assertEqual(result.usage["total_tokens"], 360)

    def test_zero_placeholder_token_metadata_uses_conservative_estimate(self):
        def llm(_messages):
            return action("stop", reason="No path"), {"prompt_tokens": 0, "completion_tokens": 0}, False
        result = AgentGraphSearch(self.index, llm).search(self.request())
        self.assertGreater(result.usage["estimated_tokens"], 0)
        self.assertGreater(result.usage["budget_tokens"], 0)

    def test_provider_failure_is_bounded_and_does_not_leak_exception_text(self):
        result = AgentGraphSearch(self.index, ScriptedLLM([RuntimeError("private-api-key")])).search(
            self.request(budget={"max_retries": 0}))
        self.assertEqual(result.stop_reason, "llm_failure")
        self.assertNotIn("private-api-key", json.dumps(result.to_dict()))

    def test_token_overrun_rejects_action_before_tools(self):
        def llm(_messages):
            return self.success_actions()[0], {"prompt_tokens": 13000, "completion_tokens": 1}, False
        result = AgentGraphSearch(self.index, llm).search(self.request())
        self.assertEqual(result.stop_reason, "token_budget")
        self.assertEqual(result.usage["tool_calls"], 0)

    def test_time_overrun_rejects_action_before_tools(self):
        elapsed = [0.0]
        def llm(_messages):
            elapsed[0] = 61.0
            return self.success_actions()[0]
        with patch("SRgraphrag.retrieval.agent.time.monotonic", side_effect=lambda: elapsed[0]):
            result = AgentGraphSearch(self.index, llm).search(self.request())
        self.assertEqual(result.stop_reason, "time_budget")
        self.assertEqual(result.usage["tool_calls"], 0)

    def test_replay_commits_identical_path_without_model(self):
        request = self.request()
        result = AgentGraphSearch(self.index, ScriptedLLM(self.success_actions())).search(request)
        def forbidden(_messages):
            raise AssertionError("Replay must not call the model")
        replayed = AgentGraphSearch(self.index, forbidden).replay(request, result.trace)
        self.assertEqual(replayed.stop_reason, "committed")
        self.assertEqual(replayed.ranked_passage_ids, result.ranked_passage_ids)
        self.assertEqual(replayed.selected_paths, result.selected_paths)
        self.assertEqual(replayed.usage["llm_calls"], 0)

    def test_replay_rejects_tampered_observation_or_changed_index(self):
        request = self.request()
        result = AgentGraphSearch(self.index, ScriptedLLM(self.success_actions())).search(request)
        tampered = copy.deepcopy(result.trace)
        next(item for item in tampered if item.get("event") == "action")["observation"]["ok"] = False
        replayed = AgentGraphSearch(self.index, None).replay(request, tampered)
        self.assertEqual(replayed.stop_reason, "replay_mismatch")
        changed = RelationIndex.from_openie(self.documents + [{"passage": "New", "extracted_triples": []}])
        replayed = AgentGraphSearch(changed, None).replay(request, result.trace)
        self.assertEqual(replayed.stop_reason, "replay_mismatch")

    def test_replay_rejects_command_and_final_summary_tampering(self):
        request = self.request()
        result = AgentGraphSearch(self.index, ScriptedLLM(self.success_actions())).search(request)
        tampered = copy.deepcopy(result.trace)
        next(item for item in tampered if item.get("event") == "llm_response")["response"] = action("stop", reason="Tampered")
        self.assertEqual(AgentGraphSearch(self.index, None).replay(request, tampered).stop_reason, "replay_mismatch")
        tampered = copy.deepcopy(result.trace)
        tampered[-1]["passage_ids"] = []
        self.assertEqual(AgentGraphSearch(self.index, None).replay(request, tampered).stop_reason, "replay_mismatch")

    def test_replay_supports_repaired_format_errors(self):
        request = self.request()
        result = AgentGraphSearch(self.index, ScriptedLLM(["not JSON"] + self.success_actions())).search(request)
        replayed = AgentGraphSearch(self.index, None).replay(request, result.trace)
        self.assertEqual(replayed.stop_reason, "committed")
        self.assertEqual(replayed.selected_paths, result.selected_paths)


if __name__ == "__main__":
    unittest.main()
