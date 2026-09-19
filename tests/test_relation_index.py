"""Offline provenance/index compatibility checks using only the standard library."""

from __future__ import annotations

from hashlib import md5
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "SRgraphrag"))

from graph.relation_index import RelationIndex
from graph.schema import entity_id, fact_id, normalize_text, passage_id


class RelationIndexTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"idx": "old-external-id", "passage": "Source one", "extracted_triples": [
                ["Alice", "works-for", "ACME"], ["Alice", "founded", "ACME"],
                ["Alice", "works-for", "ACME"], ["", "invalid", "ACME"], ["incomplete"],
            ]},
            {"passage": "Source two", "extracted_triples": [["ALICE", "works for", "Acme"]]},
            {"passage": "No extracted facts", "extracted_triples": []},
        ]

    def test_legacy_hashes_and_normalization_are_preserved(self):
        self.assertEqual(normalize_text("  A--B / C  "), "a  b   c")
        self.assertEqual(entity_id("Alice"), "entity-" + md5(b"alice").hexdigest())
        self.assertEqual(fact_id(["Alice", "works-for", "ACME"]),
                         "fact-" + md5(str(("alice", "works for", "acme")).encode()).hexdigest())
        index = RelationIndex.from_openie(self.records)
        self.assertIn(passage_id("Source one"), index.passages)
        self.assertNotIn("old-external-id", index.passages)

    def test_multiple_predicates_sources_and_original_labels(self):
        index = RelationIndex.from_openie({"docs": self.records})
        self.assertEqual(len(index.entities), 2)
        self.assertEqual(len(index.facts), 2)
        self.assertEqual(len(index.passages), 3)
        fact = index.facts[fact_id(["Alice", "works for", "ACME"])]
        self.assertEqual(set(fact.passage_ids), {passage_id("Source one"), passage_id("Source two")})
        self.assertEqual(len(fact.raw_triple_variants), 2)
        self.assertEqual(index.entities[entity_id("Alice")].display_labels, ("ALICE", "Alice"))
        self.assertEqual(len(index.outgoing_fact_ids[entity_id("Alice")]), 2)
        self.assertEqual(index.incoming_fact_ids[entity_id("Alice")], ())
        self.assertEqual(len(index.incoming_fact_ids[entity_id("ACME")]), 2)

    def test_subset_excludes_external_sources_and_entities(self):
        allowed = [passage_id("Source two")]
        index = RelationIndex.from_openie(self.records, passage_ids=allowed)
        self.assertEqual(list(index.passages), allowed)
        self.assertEqual(len(index.facts), 1)
        self.assertEqual(next(iter(index.facts.values())).passage_ids, tuple(allowed))
        self.assertEqual(RelationIndex.from_openie(self.records, passage_ids=[]).manifest.passage_count, 0)

    def test_order_and_duplicate_records_do_not_change_fingerprints(self):
        original = RelationIndex.from_openie(self.records)
        reordered = [dict(row, extracted_triples=list(reversed(row["extracted_triples"])))
                     for row in reversed(self.records)]
        reordered.append(self.records[0])
        self.assertEqual(original.to_dict(), RelationIndex.from_openie(reordered).to_dict())

    def test_source_changes_invalidate_without_changing_corpus_fingerprint(self):
        original = RelationIndex.from_openie(self.records)
        changed = [dict(row) for row in self.records]
        changed[1]["extracted_triples"] = [["Alice", "left", "ACME"]]
        newer = RelationIndex.from_openie(changed)
        self.assertEqual(original.manifest.corpus_fingerprint, newer.manifest.corpus_fingerprint)
        self.assertNotEqual(original.manifest.source_fingerprint, newer.manifest.source_fingerprint)

    def test_roundtrip_and_corrupt_provenance_rejection(self):
        index = RelationIndex.from_openie(self.records)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "relations.json"
            index.save(target)
            self.assertEqual(RelationIndex.load(target).to_dict(), index.to_dict())
            with self.assertRaisesRegex(ValueError, "Stale"):
                RelationIndex.load(target, corpus_fingerprint="another-corpus")
            broken = json.loads(target.read_text(encoding="utf-8"))
            next(iter(broken["facts"].values()))["passage_ids"] = ["chunk-invented"]
            target.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                RelationIndex.load(target)

    def test_unknown_schema_and_invalid_inputs_fail_explicitly(self):
        for records in (None, "not records", {"not_docs": []}, [{"passage": None}],
                        [{"passage": "P", "extracted_triples": "bad"}]):
            with self.assertRaises(ValueError):
                RelationIndex.from_openie(records)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "relations.json"
            payload = RelationIndex.from_openie([]).to_dict()
            payload["manifest"]["schema_version"] = 999
            target.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                RelationIndex.load(target)

    def test_cli_runs_without_site_packages_and_refuses_source_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "openie.json"
            source.write_text(json.dumps({"docs": self.records}), encoding="utf-8")
            command = [sys.executable, "-S", str(PROJECT_ROOT / "scripts" / "build_relation_index.py"), "--openie", str(source)]
            result = subprocess.run(command + ["--check-only"], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["manifest"]["fact_count"], 2)
            refused = subprocess.run(command + ["--output", str(source), "--force"], capture_output=True, text=True, timeout=10)
            self.assertNotEqual(refused.returncode, 0)
            self.assertEqual(json.loads(source.read_text())["docs"], self.records)


if __name__ == "__main__":
    unittest.main()
