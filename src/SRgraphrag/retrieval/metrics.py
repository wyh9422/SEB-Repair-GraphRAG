"""Legacy retrieval metrics. Callers supply rankings already sliced to Top-k.

Empty gold sets are skipped by recall metrics; gold multiplicity follows the
original evaluator so this extraction does not change reported baseline values."""

from __future__ import annotations


def avg_recall_at_k(golds: list[list[str]], retrieved_topk_list: list[list[str]]) -> float:
    n = len(golds)
    s = 0.0
    denom = 0
    for i in range(n):
        g = golds[i] or []
        if len(g) == 0:
            continue
        denom += 1
        rset = set(retrieved_topk_list[i])
        hit = sum(1 for x in g if x in rset)
        s += hit / len(g)
    return s / denom if denom > 0 else 0.0


def all_recall_at_k(golds: list[list[str]], retrieved_topk_list: list[list[str]]) -> float:
    n = len(golds)
    hit = 0
    denom = 0
    for i in range(n):
        g = golds[i] or []
        if len(g) == 0:
            continue
        denom += 1
        rset = set(retrieved_topk_list[i])
        if all(x in rset for x in g):
            hit += 1
    return hit / denom if denom > 0 else 0.0


def missing_list(gold: list[str], topk_docs: list[str]) -> list[str]:
    rset = set(topk_docs)
    return [g for g in (gold or []) if g not in rset]


def hit_at_1(gold: list[str], ranked: list[str]) -> int:
    if not gold or not ranked:
        return 0
    return 1 if ranked[0] in set(gold) else 0


def mrr_first_hit(gold: list[str], ranked: list[str]) -> float:
    if not gold:
        return 0.0
    gset = set(gold)
    for idx, doc in enumerate(ranked):
        if doc in gset:
            return 1.0 / (idx + 1)
    return 0.0
