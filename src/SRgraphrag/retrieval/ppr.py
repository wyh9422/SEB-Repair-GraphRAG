"""Adapter for the existing weighted, undirected PPR calculation."""

from __future__ import annotations

import time

from .types import GraphSearchResult


class PPRGraphSearch:
    def __init__(self, owner, passage_node_weight=None):
        self.owner = owner
        self.passage_node_weight = (
            owner.global_config.passage_node_weight
            if passage_node_weight is None else passage_node_weight
        )

    def search(self, request):
        import numpy as np

        owner = self.owner
        invalid = getattr(owner, "_graph_validation_error", None)
        if invalid:
            return GraphSearchResult(stop_reason="invalid_graph", fallback_reason=invalid)
        mapping = owner.node_name_to_vertex_idx
        if not owner.passage_node_keys:
            return GraphSearchResult(stop_reason="empty_corpus", fallback_reason="empty_corpus")
        if owner.graph.vcount() != len(mapping) or owner.graph.ecount() == 0:
            return GraphSearchResult(stop_reason="invalid_graph", fallback_reason="missing_graph_edges_or_mapping")
        if any(key not in mapping for key in owner.passage_node_keys):
            return GraphSearchResult(stop_reason="invalid_graph", fallback_reason="missing_passage_nodes")
        if "weight" not in owner.graph.es.attributes():
            return GraphSearchResult(stop_reason="invalid_graph", fallback_reason="missing_graph_weights")
        edge_weights = np.asarray(owner.graph.es["weight"], dtype=float)
        if not np.all(np.isfinite(edge_weights)) or np.any(edge_weights < 0):
            return GraphSearchResult(stop_reason="invalid_graph", fallback_reason="invalid_graph_weights")
        weights = np.zeros(owner.graph.vcount(), dtype=float)
        for entity_id, score in request.seed_scores.items():
            if entity_id in mapping and np.isfinite(score) and score > 0:
                weights[mapping[entity_id]] += score

        ids, scores = owner.dense_passage_retrieval(request.retrieval_query)
        scores = np.asarray(scores, dtype=float)
        if scores.size:
            if not np.all(np.isfinite(scores)):
                return GraphSearchResult(stop_reason="invalid_scores", fallback_reason="nonfinite_dense_scores")
            span = float(scores.max() - scores.min())
            normalized = np.ones_like(scores) if span == 0 else (scores - scores.min()) / span
            for position, passage_index in enumerate(ids):
                key = owner.passage_node_keys[int(passage_index)]
                weights[mapping[key]] += normalized[position] * self.passage_node_weight
        if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
            return GraphSearchResult(stop_reason="no_seeds", fallback_reason="no_positive_seed_weights")
        started = time.monotonic()
        ranked_ids, ranked_scores = owner.run_ppr(weights, damping=owner.global_config.damping)
        elapsed = time.monotonic() - started
        owner.ppr_time += elapsed
        limit = request.retrieval_limit
        return GraphSearchResult(
            ranked_passage_ids=[owner.passage_node_keys[int(i)] for i in ranked_ids[:limit]],
            scores=[float(x) for x in ranked_scores[:limit]],
            score_sources=["ppr"] * len(ranked_ids[:limit]),
            stop_reason="ppr_complete", usage={"ppr_seconds": elapsed},
        )
