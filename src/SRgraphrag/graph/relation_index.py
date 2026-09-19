"""Directed fact sidecar built from existing OpenIE data; no embedding or LLM."""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable

from .schema import (
    NORMALIZATION_VERSION, SCHEMA_VERSION, EntityRecord, FactRecord,
    IndexManifest, PassageRecord, entity_id, fact_id, fingerprint,
    normalize_text, passage_id,
)


class RelationIndex:
    def __init__(self, entities, facts, passages, manifest):
        self.entities = dict(sorted(entities.items()))
        self.facts = dict(sorted(facts.items()))
        self.passages = dict(sorted(passages.items()))
        self.manifest = manifest
        outgoing, incoming = defaultdict(list), defaultdict(list)
        for key, fact in self.facts.items():
            outgoing[fact.subject_id].append(key)
            incoming[fact.object_id].append(key)
        self.outgoing_fact_ids = {key: tuple(outgoing[key]) for key in self.entities}
        self.incoming_fact_ids = {key: tuple(incoming[key]) for key in self.entities}
        self.fact_ids_by_passage = {key: value.fact_ids for key, value in self.passages.items()}

    @classmethod
    def from_openie(cls, records: Iterable[dict], passage_ids=None) -> "RelationIndex":
        """Build only the requested corpus, recomputing chunk IDs like the legacy loader.

        Accept either the OpenIE ``docs`` array or its enclosing JSON object.
        Repeated records/triples merge deterministically. Invalid/empty triples
        are ignored; invalid records fail rather than silently inventing text.
        ``passage_ids`` is a whitelist, not an instruction to create missing docs.
        """
        if isinstance(records, dict):
            if "docs" not in records:
                raise ValueError("OpenIE object must contain a docs array")
            records = records["docs"]
        if isinstance(records, (str, bytes)) or records is None:
            raise ValueError("OpenIE records must be an iterable of objects")
        if isinstance(passage_ids, (str, bytes)):
            raise ValueError("passage_ids must be a collection of chunk IDs")
        allowed = None if passage_ids is None else set(passage_ids)
        if allowed is not None and not all(isinstance(key, str) for key in allowed):
            raise ValueError("passage_ids must contain strings")

        contents, raw_by_passage = {}, defaultdict(set)
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("passage"), str):
                raise ValueError("Each OpenIE record must contain string passage content")
            content = record["passage"]
            key = passage_id(content)
            if allowed is not None and key not in allowed:
                continue
            if key in contents and contents[key] != content:
                raise ValueError("Conflicting passage content for the same hash")
            contents[key] = content
            triples = record.get("extracted_triples") or []
            if not isinstance(triples, (list, tuple)):
                raise ValueError("extracted_triples must be an array")
            for triple in triples:
                if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                    continue
                raw = tuple(str(part) for part in triple)
                if all(normalize_text(part) for part in raw):
                    raw_by_passage[key].add(raw)

        labels, canonicals = defaultdict(set), {}
        facts_data, passage_facts = {}, defaultdict(set)
        for key in sorted(contents):
            for raw in sorted(raw_by_passage[key]):
                normalized = tuple(normalize_text(part) for part in raw)
                subject, obj = entity_id(raw[0]), entity_id(raw[2])
                fid = fact_id(raw)
                for eid, canonical, label in ((subject, normalized[0], raw[0]), (obj, normalized[2], raw[2])):
                    if eid in canonicals and canonicals[eid] != canonical:
                        raise ValueError("Conflicting entity content for the same hash")
                    canonicals[eid] = canonical
                    labels[eid].add(label)
                if fid not in facts_data:
                    facts_data[fid] = [subject, normalized[1], obj, set(), set()]
                data = facts_data[fid]
                if data[:3] != [subject, normalized[1], obj]:
                    raise ValueError("Conflicting fact content for the same hash")
                data[3].add(raw)
                data[4].add(key)
                passage_facts[key].add(fid)

        entities = {key: EntityRecord(key, canonicals[key], tuple(sorted(values)))
                    for key, values in labels.items()}
        facts = {key: FactRecord(key, data[0], data[1], data[2], tuple(sorted(data[3])), tuple(sorted(data[4])))
                 for key, data in facts_data.items()}
        passages = {key: PassageRecord(key, contents[key], tuple(sorted(passage_facts[key])), tuple(sorted(raw_by_passage[key])))
                    for key in sorted(contents)}
        corpus = [[key, contents[key]] for key in sorted(contents)]
        source = [[key, contents[key], sorted(raw_by_passage[key])] for key in sorted(contents)]
        manifest = IndexManifest(
            SCHEMA_VERSION, NORMALIZATION_VERSION, fingerprint(corpus), fingerprint(source),
            len(passages), len(entities), len(facts), sum(len(values) for values in raw_by_passage.values()),
        )
        return cls(entities, facts, passages, manifest)

    def to_dict(self) -> dict:
        return {
            "manifest": self.manifest.to_dict(),
            "entities": {key: value.to_dict() for key, value in self.entities.items()},
            "facts": {key: value.to_dict() for key, value in self.facts.items()},
            "passages": {key: value.to_dict() for key, value in self.passages.items()},
        }

    def save(self, path) -> None:
        """Atomically write this sidecar; never touches the original igraph file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, suffix=".tmp", delete=False) as handle:
                temporary = handle.name
                json.dump(self.to_dict(), handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
                handle.write("\n")
            os.replace(temporary, target)
            temporary = None
        finally:
            if temporary is not None:
                os.unlink(temporary)

    @classmethod
    def load(cls, path, *, corpus_fingerprint=None, source_fingerprint=None) -> "RelationIndex":
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or not isinstance(payload.get("manifest"), dict):
            raise ValueError("Invalid relation index manifest")
        manifest = payload["manifest"]
        if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("normalization_version") != NORMALIZATION_VERSION:
            raise ValueError("Unsupported relation index schema/normalization version; rebuild the sidecar")
        if not isinstance(payload.get("passages"), dict):
            raise ValueError("Invalid relation index passages")
        try:
            records = [{"passage": value["content"], "extracted_triples": value["raw_triples"]}
                       for value in payload["passages"].values()]
            result = cls.from_openie(records)
        except (KeyError, TypeError) as exc:
            raise ValueError("Invalid relation index source records") from exc
        # Rebuild from source witnesses to validate all IDs, direction, provenance,
        # cached records and fingerprints together, not just trust stored counters.
        if result.to_dict() != payload:
            raise ValueError("Relation index content/fingerprint mismatch; rebuild the sidecar")
        for expected, actual, name in (
            (corpus_fingerprint, result.manifest.corpus_fingerprint, "corpus"),
            (source_fingerprint, result.manifest.source_fingerprint, "source"),
        ):
            if expected is not None and expected != actual:
                raise ValueError(f"Stale relation index: {name} fingerprint differs")
        return result
