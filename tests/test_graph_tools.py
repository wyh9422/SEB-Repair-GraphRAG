"""Closed-world tools, path validation and bounded provenance selection."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "SRgraphrag"))
from graph.relation_index import RelationIndex
from graph.schema import entity_id, fact_id, passage_id
from graph.tools import GraphTools


def step(source, target, relation="links", reverse=False):
    triple = [target, relation, source] if reverse else [source, relation, target]
    return {"fact_id": fact_id(triple), "from_entity_id": entity_id(source),
            "to_entity_id": entity_id(target), "traversal_direction": "in" if reverse else "out"}


class GraphToolsTests(unittest.TestCase):
    def setUp(self):
        records = [
            ("left source", [["A", "links", "B"]]),
            ("right source", [["B", "links", "C"]]),
            ("shared source", [["A", "links", "B"], ["B", "links", "C"]]),
            ("extra source", [["C", "links", "D"]]),
            ("protected source", []),
            ("unrelated source", [["X", "links", "Y"]]),
        ]
        self.index = RelationIndex.from_openie([{"passage": content, "extracted_triples": triples}
                                                for content, triples in records])
        self.tools = GraphTools(self.index, [entity_id("A")])
        self.path = [step("A", "B"), step("B", "C")]

    def expose_path(self, tools=None):
        tools = self.tools if tools is None else tools
        self.assertTrue(tools.expand_entity(entity_id("A"))["ok"])
        self.assertTrue(tools.expand_entity(entity_id("B"))["ok"])

    def test_unknown_or_unexposed_ids_cannot_be_accessed(self):
        result = self.tools.inspect_passage(passage_id("unrelated source"))
        self.assertEqual(result["error"]["code"], "unexposed_id")
        self.assertFalse(self.tools.expand_entity(entity_id("X"))["ok"])
        self.assertFalse(self.tools.validate_path(self.path)["ok"])

    def test_forward_reverse_and_multistep_paths_keep_original_direction(self):
        self.expose_path()
        forward = self.tools.validate_path(self.path)
        reverse = self.tools.validate_path([step("B", "A", reverse=True)])
        self.assertTrue(forward["ok"])
        self.assertTrue(reverse["ok"])
        bad = [dict(step("B", "A", reverse=True), traversal_direction="out")]
        self.assertFalse(self.tools.validate_path(bad)["ok"])
        self.assertEqual(self.index.facts[step("A", "B")["fact_id"]].subject_id, entity_id("A"))

    def test_cycles_disconnections_and_excess_length_are_rejected(self):
        self.expose_path()
        self.tools.expand_entity(entity_id("C"))
        cycle = self.tools.validate_path([step("A", "B"), step("B", "A", reverse=True)])
        disconnected = self.tools.validate_path([step("A", "B"), step("C", "D")])
        self.assertEqual(cycle["error"]["code"], "cyclic_path")
        self.assertEqual(disconnected["error"]["code"], "disconnected_path")
        short = GraphTools(self.index, [entity_id("A")], max_path_length=1)
        self.expose_path(short)
        self.assertEqual(short.validate_path(self.path)["error"]["code"], "path_length_exceeded")

    def test_commit_selects_minimum_shared_source_not_all_provenance(self):
        self.expose_path()
        result = self.tools.commit_paths([self.path])
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["passage_ids"], [passage_id("shared source")])
        self.assertEqual(set(result["fact_passage_ids"].values()), {passage_id("shared source")})
        self.assertEqual(result["selected_paths"], [self.path])
        result["selected_paths"][0][0]["fact_id"] = "invented"
        self.assertNotEqual(self.tools.committed_paths[0][0]["fact_id"], "invented")

    def test_exact_cover_avoids_a_greedy_three_source_solution(self):
        triples = [[f"N{i}", "links", f"N{i + 1}"] for i in range(6)]
        records = [
            {"passage": "greedy covers four", "extracted_triples": triples[:4]},
            {"passage": "optimal odd", "extracted_triples": [triples[i] for i in (0, 1, 4)]},
            {"passage": "optimal even", "extracted_triples": [triples[i] for i in (2, 3, 5)]},
        ]
        index = RelationIndex.from_openie(records)
        tools = GraphTools(index, [entity_id("N0")], evidence_limit=2, max_path_length=6)
        for i in range(6):
            self.assertTrue(tools.expand_entity(entity_id(f"N{i}"), direction="out")["ok"])
        result = tools.commit_paths([[step(f"N{i}", f"N{i + 1}") for i in range(6)]])
        self.assertTrue(result["ok"], result)
        self.assertEqual(set(result["passage_ids"]), {passage_id("optimal odd"), passage_id("optimal even")})

    def test_protected_source_counts_towards_minimum_cover(self):
        protected = passage_id("left source")
        tools = GraphTools(self.index, [entity_id("A")], [protected], evidence_limit=2)
        self.expose_path(tools)
        result = tools.commit_paths([self.path])
        self.assertTrue(result["ok"])
        self.assertEqual(result["passage_ids"][0], protected)
        self.assertEqual(len(result["passage_ids"]), 2)
        for fid, pid in result["fact_passage_ids"].items():
            self.assertIn(pid, self.index.facts[fid].passage_ids)

    def test_capacity_failure_is_explicit_and_cannot_change_protection(self):
        protected = passage_id("protected source")
        tools = GraphTools(self.index, [entity_id("A")], [protected], evidence_limit=1)
        self.expose_path(tools)
        failed = tools.commit_paths([self.path])
        self.assertEqual(failed["error"]["code"], "evidence_capacity_exceeded")
        self.assertEqual(tools.committed_paths, [])
        override = tools.commit_paths([self.path], protected_passage_ids=[])
        self.assertEqual(override["error"]["code"], "protected_evidence_override")

    def test_inspection_exposes_only_known_passage_and_counts_facts(self):
        tools = GraphTools(self.index, [], [passage_id("shared source")])
        result = tools.inspect_passage(passage_id("shared source"))
        self.assertTrue(result["ok"])
        self.assertEqual(result["content"], "shared source")
        self.assertEqual(tools.usage["expanded_edges"], 2)
        self.assertIn(entity_id("C"), tools.exposed_entity_ids)
        self.assertTrue(tools.validate_path(self.path)["ok"])
        self.assertEqual(tools.inspect_passage(passage_id("shared source"))["error"]["code"], "repeated_inspection")

    def test_pagination_is_stable_and_repeat_attempts_consume_calls(self):
        tools = GraphTools(self.index, [entity_id("B")], max_neighbors=1)
        first = tools.expand_entity(entity_id("B"), limit=1)
        second = tools.expand_entity(entity_id("B"), limit=1, cursor=first["next_cursor"])
        self.assertEqual(first["next_cursor"], 1)
        self.assertIsNone(second["next_cursor"])
        self.assertNotEqual(first["facts"][0]["fact_id"], second["facts"][0]["fact_id"])
        repeated = tools.expand_entity(entity_id("B"), limit=1)
        self.assertEqual(repeated["error"]["code"], "repeated_expansion")
        self.assertEqual(repeated["usage"]["tool_calls"], 3)
        self.assertEqual(repeated["usage"]["expanded_edges"], 2)

    def test_omitted_expand_limit_respects_configured_neighbor_budget(self):
        tools = GraphTools(self.index, [entity_id("B")], max_neighbors=1)
        result = tools.execute("expand_entity", {"entity_id": entity_id("B")})
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["facts"]), 1)

    def test_call_and_edge_budgets_cannot_be_bypassed(self):
        tools = GraphTools(self.index, [entity_id("A")], max_calls=2, max_expansions=1)
        self.assertTrue(tools.expand_entity(entity_id("A"))["ok"])
        failed = tools.expand_entity(entity_id("B"))
        self.assertEqual(failed["error"]["code"], "expansion_budget_exhausted")
        self.assertEqual(tools.validate_path([self.path[0]])["error"]["code"], "call_budget_exhausted")
        self.assertEqual(tools.usage["expanded_edges"], 1)

    def test_hybrid_entity_allowlist_applies_to_expansion_and_inspection(self):
        tools = GraphTools(self.index, [entity_id("A")], allowed_entity_ids=[entity_id("A"), entity_id("B")])
        tools.expand_entity(entity_id("A"))
        expanded = tools.expand_entity(entity_id("B"))
        inspected = tools.inspect_passage(passage_id("shared source"))
        self.assertEqual(len(expanded["facts"]), 1)
        self.assertEqual(len(inspected["facts"]), 1)
        self.assertNotIn(entity_id("C"), tools.exposed_entity_ids)

    def test_argument_schema_is_strict_for_public_and_dispatch_methods(self):
        for args in ({"entity_id": entity_id("A"), "limit": True},
                     {"entity_id": entity_id("A"), "cursor": -1},
                     {"entity_id": entity_id("A"), "limit": 10000},
                     {"entity_id": entity_id("A"), "invented_option": True}):
            self.assertFalse(self.tools.execute("expand_entity", args)["ok"])
        self.assertEqual(self.tools.execute("run_python", {})["error"]["code"], "unknown_action")
        self.expose_path()
        self.assertFalse(self.tools.validate_path([dict(self.path[0], source="invented")])["ok"])

    def test_cover_compute_budget_fails_instead_of_claiming_nonminimal_result(self):
        self.expose_path()
        self.tools.MAX_COVER_SEARCH_STATES = 0
        result = self.tools.commit_paths([self.path])
        self.assertEqual(result["error"]["code"], "cover_search_budget_exhausted")


if __name__ == "__main__":
    unittest.main()
