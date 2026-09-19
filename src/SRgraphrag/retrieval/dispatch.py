"""Strategy selection; evaluation labels never enter this module."""

from __future__ import annotations

from dataclasses import replace
import os

from .ppr import PPRGraphSearch
from .types import GraphSearchResult


def dense_result(owner, request, reason=None):
    ids, scores = owner.dense_passage_retrieval(request.retrieval_query)
    ids = list(ids[:request.retrieval_limit])
    return GraphSearchResult(
        ranked_passage_ids=[owner.passage_node_keys[int(i)] for i in ids],
        scores=[float(s) for s in scores[:len(ids)]],
        score_sources=["dpr"] * len(ids), stop_reason="dpr_fallback",
        fallback_reason=reason,
    )


def get_relation_index(owner):
    source_path = getattr(owner, "openie_results_path", None)
    source_stat = os.stat(source_path) if source_path and os.path.isfile(source_path) else None
    signature = (tuple(owner.passage_node_keys),
                 (source_stat.st_mtime_ns, source_stat.st_size) if source_stat else None)
    cached_signature = getattr(owner, "_relation_index_signature", signature)
    if getattr(owner, "_relation_index", None) is None or signature != cached_signature:
        from ..graph.relation_index import RelationIndex
        records, _ = owner.load_existing_openie([])
        owner._relation_index = RelationIndex.from_openie(records, passage_ids=owner.passage_node_keys)
        owner._relation_index_signature = signature
    return owner._relation_index


def search_graph(owner, request, mode="ppr", fallback="ppr"):
    if mode not in ("ppr", "agent", "hybrid") or fallback not in ("ppr", "dpr", "none"):
        raise ValueError("Unknown graph search or fallback mode")
    ppr = PPRGraphSearch(owner)
    if mode == "ppr":
        result = ppr.search(request)
        return result if result.ranked_passage_ids else dense_result(owner, request, result.fallback_reason)
    prior = None
    try:
        from .agent import AgentGraphSearch
        from .agent_llm import AgentLLMAdapter
        index = get_relation_index(owner)
        if mode == "hybrid":
            prior = ppr.search(request)
            node_scores = getattr(owner, "_last_ppr_node_scores", {}) if prior.ranked_passage_ids else {}
            limit = max(1, int(request.budget.get("max_expansions", 64))) * 4
            candidates = [key for key in sorted(node_scores, key=lambda k: (-node_scores[k], k)) if key in index.entities][:limit]
            allowed = set(candidates) | set(request.seed_entity_ids)
            # One bounded boundary layer prevents the region from being only a
            # projection of retrieved passages. This is explicitly a heuristic.
            boundary = set()
            for fact in index.facts.values():
                if fact.subject_id in allowed:
                    boundary.add(fact.object_id)
                if fact.object_id in allowed:
                    boundary.add(fact.subject_id)
            candidates = sorted(allowed | set(sorted(boundary - allowed)[:limit]))
            request = replace(request, candidate_entity_ids=candidates)
        llm = getattr(owner, "_agent_llm", None)
        if llm is None:
            llm = AgentLLMAdapter(owner.llm_model, os.path.join(owner.working_dir, "agent_cache.sqlite"))
        result = AgentGraphSearch(index, llm, budget=request.budget).search(request)
        if prior is not None:
            result.usage["hybrid_ppr"] = prior.usage
            result.usage["candidate_entity_count"] = len(request.candidate_entity_ids or [])
    except Exception as exc:
        # Do not serialize credentials or provider payloads in exception text.
        result = GraphSearchResult(stop_reason="agent_error", fallback_reason=type(exc).__name__,
                                   trace=[{"event": "agent_error", "error_type": type(exc).__name__}])
    if result.ranked_passage_ids and result.selected_paths:
        return result
    reason = result.fallback_reason or result.stop_reason or "no_committed_path"
    if fallback == "none":
        result.fallback_reason = reason
        return result
    recovered = prior if fallback == "ppr" and prior and prior.ranked_passage_ids else None
    if recovered is None and fallback == "ppr":
        recovered = ppr.search(request)
    if recovered is None or not recovered.ranked_passage_ids:
        recovered = dense_result(owner, request, reason)
    return replace(recovered, stop_reason="fallback_" + recovered.stop_reason,
                   fallback_reason=reason, selected_paths=[],
                   usage={"agent": result.usage, "fallback": recovered.usage,
                          "fallback_reused_hybrid_ppr": prior is not None and recovered is prior},
                   trace=result.trace + [{"event": "fallback", "requested": fallback, "used": recovered.stop_reason, "reason": reason}])
