"""Stable-ID contracts shared by PPR and graph exploration strategies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GraphSearchRequest:
    original_query: str
    retrieval_query: str
    round_index: int
    seed_fact_ids: List[str]
    seed_entity_ids: List[str]
    seed_scores: Dict[str, float]
    protected_passage_ids: List[str]
    retrieval_limit: int = 30
    evidence_limit: int = 5
    budget: Dict[str, Any] = field(default_factory=dict)
    candidate_entity_ids: Optional[List[str]] = None


@dataclass
class GraphSearchResult:
    ranked_passage_ids: List[str] = field(default_factory=list)
    scores: List[Optional[float]] = field(default_factory=list)
    score_sources: List[str] = field(default_factory=list)
    selected_paths: List[Any] = field(default_factory=list)
    stop_reason: str = ""
    fallback_reason: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    trace: List[Any] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)
