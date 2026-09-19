"""Stable records using the original GraphRAG normalization and hash scheme."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import md5, sha256
import json
import re


SCHEMA_VERSION = 1
NORMALIZATION_VERSION = "legacy-ascii-v1"


def normalize_text(value: object) -> str:
    """Match utils.misc_utils.text_processing without importing model packages."""
    return re.sub(r"[^A-Za-z0-9 ]", " ", str(value).lower()).strip()


def stable_id(content: str, prefix: str) -> str:
    return prefix + md5(content.encode()).hexdigest()


def entity_id(text: str) -> str:
    return stable_id(normalize_text(text), "entity-")


def fact_id(triple) -> str:
    # The original fact store hashes str(tuple(normalized_triple)), not JSON.
    return stable_id(str(tuple(normalize_text(part) for part in triple)), "fact-")


def passage_id(content: str) -> str:
    return stable_id(content, "chunk-")


def fingerprint(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


class JsonRecord:
    def to_dict(self) -> dict:
        # JSON roundtrip also converts immutable tuples into JSON arrays.
        return json.loads(json.dumps(asdict(self), ensure_ascii=False))


@dataclass(frozen=True)
class EntityRecord(JsonRecord):
    entity_id: str
    canonical_text: str
    display_labels: tuple[str, ...]


@dataclass(frozen=True)
class FactRecord(JsonRecord):
    fact_id: str
    subject_id: str
    predicate: str
    object_id: str
    raw_triple_variants: tuple[tuple[str, str, str], ...]
    passage_ids: tuple[str, ...]


@dataclass(frozen=True)
class PassageRecord(JsonRecord):
    passage_id: str
    content: str
    fact_ids: tuple[str, ...]
    raw_triples: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class IndexManifest(JsonRecord):
    schema_version: int
    normalization_version: str
    corpus_fingerprint: str
    source_fingerprint: str
    passage_count: int
    entity_count: int
    fact_count: int
    source_triple_count: int
