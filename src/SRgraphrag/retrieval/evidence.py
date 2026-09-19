"""Top-5 evidence protection policies, extracted without changing baseline semantics."""

from __future__ import annotations

import math


def dedup_preserve(xs: list[str]) -> list[str]:
    out, seen = [], set()
    for x in xs:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _find_tail_replace_idx(cand5: list[str], protected: set[str]) -> int | None:
    # from tail -> head, pick first slot whose doc is non-empty AND not protected
    for j in range(4, -1, -1):
        if cand5[j] and cand5[j] not in protected:
            return j
    # if all are protected or empty, allow replacing an empty slot that is not protected (rare)
    for j in range(4, -1, -1):
        if (not cand5[j]) and cand5[j] not in protected:
            return j
    return None


def apply_evidence_injection_top5(
    top5: list[str],
    evidence_docs: list[str] | None,
    base_ranked_docs: list[str] | None,
) -> tuple[list[str], dict]:
    """
    Inject missing evidence into top5 by tail replacement, never replacing existing evidence.
    If evidence already covered, do nothing (no extra tail replacement).
    """
    log = {
        "evidence_enabled": bool(evidence_docs),
        "evidence_used": False,
        "evidence_missing_cnt": 0,
        "evidence_missing": [],
        "replaced_cnt": 0,
    }

    cand = list(top5 or [])[:5]
    while len(cand) < 5:
        cand.append("")

    ev = [x for x in (evidence_docs or []) if x]
    ev = dedup_preserve(ev)[:5]  # caller should pass <=5, still guard here
    if not ev:
        # normalize (dedup+refill) but no injection
        out = dedup_preserve(cand)
        if len(out) < 5 and base_ranked_docs:
            out += [x for x in base_ranked_docs if x and x not in set(out)]
        out = out[:5]
        while len(out) < 5:
            out.append("")
        return out, log

    ev_set = set(ev)
    cand_set = set([x for x in cand if x])
    missing = [x for x in ev if x not in cand_set]

    log["evidence_missing"] = missing[:]
    log["evidence_missing_cnt"] = len(missing)
    log["evidence_used"] = (len(missing) > 0)

    replaced = 0
    if missing:
        # replace from tail: pick first slot that is NOT evidence
        for e in missing:
            idx = _find_tail_replace_idx(cand, protected=ev_set)
            if idx is None:
                break
            cand[idx] = e
            replaced += 1

    log["replaced_cnt"] = replaced

    # dedup + refill
    out = dedup_preserve(cand)
    if len(out) < 5 and base_ranked_docs:
        out += [x for x in base_ranked_docs if x and x not in set(out)]
    out = out[:5]
    while len(out) < 5:
        out.append("")

    # invariant best-effort: all ev that can fit should be in out
    # (if ev>5 we truncated; if protected slots fill all 5, cannot inject more)
    return out, log


def apply_guard_top5_protect_evidence(
    top5: list[str],
    dpr1: str | None,
    protected_docs: set[str],
) -> tuple[list[str], dict]:
    """
    Tail replace with THIS round's dpr1 if not already present.
    Never replace protected evidence docs.
    """
    log = {"guard_enabled": True, "guard_triggered": False, "guard_replaced_idx": None}

    if not dpr1:
        return top5, log
    if dpr1 in set([x for x in top5 if x]):
        return top5, log

    cand = list(top5 or [])[:5]
    while len(cand) < 5:
        cand.append("")

    idx = _find_tail_replace_idx(cand, protected=protected_docs)
    if idx is None:
        return cand[:5], log

    cand[idx] = dpr1
    log["guard_triggered"] = True
    log["guard_replaced_idx"] = int(idx)

    # dedup but keep length 5 with refill by internal caller
    out = dedup_preserve(cand)[:5]
    while len(out) < 5:
        out.append("")
    return out, log


def align_scores_to_top5(top5_docs: list[str], base_docs: list[str], base_scores):
    if base_scores is None:
        base_scores_list = []
    else:
        base_scores_list = [float(x) if x is not None and math.isfinite(float(x)) else None for x in list(base_scores)]
    base_docs_list = list(base_docs or [])

    doc2score = {}
    for d, s in zip(base_docs_list, base_scores_list):
        if d and d not in doc2score:
                doc2score[d] = s

    out_scores = []
    for d in top5_docs:
        out_scores.append(doc2score.get(d) if d else None)
    return out_scores
