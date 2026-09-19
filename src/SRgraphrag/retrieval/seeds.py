"""Fact-derived seeds with the baseline occurrence and document-frequency weights."""

from __future__ import annotations

from hashlib import md5
import math


def build_fact_seeds(facts, fact_indices, fact_scores, known_entity_ids,
                     entity_to_passages, link_top_k=0):
    if link_top_k < 0:
        raise ValueError("link_top_k must be nonnegative")
    known = set(known_entity_ids)
    sums, counts, fact_ids, missing = {}, {}, [], set()
    for rank, fact in enumerate(facts):
        if not isinstance(fact, (list, tuple)) or len(fact) != 3:
            continue
        if rank >= len(fact_indices):
            raise ValueError("fact indices must align with selected facts")
        score = float(fact_scores[fact_indices[rank]])
        if not math.isfinite(score) or score < 0:
            continue
        triple = tuple(str(value).lower() for value in fact)
        fact_ids.append("fact-" + md5(str(triple).encode()).hexdigest())
        for phrase in (triple[0], triple[2]):
            entity_id = "entity-" + md5(phrase.encode()).hexdigest()
            if entity_id not in known:
                missing.add(entity_id)
                continue
            weight = score / max(1, len(entity_to_passages.get(entity_id, ())))
            sums[entity_id] = sums.get(entity_id, 0.0) + weight
            counts[entity_id] = counts.get(entity_id, 0) + 1
    weights = {key: sums[key] / counts[key] for key in sums}
    # Stable tie handling makes repeated runs reproducible. Zero-score facts do
    # not become seeds; an all-zero request is handled by the caller's fallback.
    ranked = sorted(weights, key=lambda key: (-weights[key], key))
    if link_top_k:
        ranked = ranked[:link_top_k]
    weights = {key: weights[key] for key in ranked if weights[key] > 0}
    return list(dict.fromkeys(fact_ids)), weights, sorted(missing)
