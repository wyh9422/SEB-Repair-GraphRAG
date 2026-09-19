"""Offline-only evidence repair evaluation. Gold labels never enter retrieval."""

from __future__ import annotations

from collections import Counter
import json
import math
import random


COUNTERS = (
    "llm_calls", "cache_hits", "tool_calls", "expanded_edges", "prompt_tokens",
    "completion_tokens", "total_tokens", "estimated_tokens", "budget_tokens",
    "retries", "invalid_actions", "tool_failures",
)


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON value is forbidden: {value}")


def load_trace(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Empty trace line {line_number}; positional alignment would be ambiguous")
            try:
                rows.append(json.loads(line, parse_constant=_reject_constant))
            except ValueError as exc:
                raise ValueError(f"Invalid trace JSON at line {line_number}: {exc}") from exc
    return rows


def load_gold(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle, parse_constant=_reject_constant)


def _finite_tree(value, location="input"):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite value at {location}")
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_tree(item, f"{location}[{index}]")


def _object(value, name, optional=False):
    if value is None and optional:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _docs(value, name, optional=False):
    if value is None and optional:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be an array of passage strings")
    return [item for item in value if item.strip()]


def _top5(value, name):
    _docs(value, name)  # Validate all entries before applying the rank cutoff.
    return _docs(value[:5], name)


def _numeric(value, name):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative finite number or null")
    return value


def _sum_known(values):
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def _rate(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def _percentile(values, quantile):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _distribution(values, total_queries):
    present = [value for value in values if value is not None]
    return {
        "sum": sum(present) if present else None,
        "mean": sum(present) / len(present) if present else None,
        "p50": _percentile(present, 0.5), "p95": _percentile(present, 0.95),
        "observed_queries": len(present), "missing_queries": total_queries - len(present),
    }


def aggregate_usage(usage, *, fallback_events=()):
    """Aggregate the known dispatch usage tree without counting aliases twice.

    agent/fallback are components; direct counters on such a wrapper are
    authoritative aggregate values when supplied. Agent elapsed time excludes
    hybrid_ppr, so the latter is added separately. Reusing that prior for fallback
    counts its time only once. Estimates/budget tokens are never added to the
    reported token total. Trace snapshots are not counted as extra calls.
    """
    usage = _object(usage, "usage", optional=True)
    _finite_tree(usage, "usage")
    warnings = []
    children = {}
    for key in ("agent", "fallback", "hybrid_ppr"):
        if usage.get(key) is not None:
            children[key] = _object(usage[key], f"usage.{key}")
    reused = usage.get("fallback_reused_hybrid_ppr")
    if reused is not None and not isinstance(reused, bool):
        raise ValueError("fallback_reused_hybrid_ppr must be boolean or null")
    if reused is None and "agent" in children and "fallback" in children:
        prior = children["agent"].get("hybrid_ppr")
        # Compatibility for traces produced before dispatch recorded aliasing.
        if (isinstance(prior, dict) and prior and prior == children["fallback"]
                and any(event.get("used") == "ppr_complete" for event in fallback_events)):
            reused = True
            warnings.append("hybrid_prior_reuse_inferred_from_legacy_trace")
    if reused:
        if "agent" not in children or not isinstance(children["agent"].get("hybrid_ppr"), dict):
            raise ValueError("Reused hybrid prior marker has no agent.hybrid_ppr component")
        children.pop("fallback", None)

    combined = [aggregate_usage(child, fallback_events=fallback_events) for child in children.values()]
    for child in combined:
        warnings.extend(child["warnings"])
    wrapper = "agent" in children or "fallback" in children
    counters = {}
    for name in COUNTERS:
        local = _numeric(usage.get(name), f"usage.{name}")
        descendant = _sum_known(child["counters"][name] for child in combined)
        counters[name] = local if wrapper and local is not None else _sum_known((local, descendant))
    if counters["total_tokens"] is None:
        counters["total_tokens"] = _sum_known((counters["prompt_tokens"], counters["completion_tokens"]))
    own_elapsed = _numeric(usage.get("elapsed_seconds"), "usage.elapsed_seconds")
    own_ppr = _numeric(usage.get("ppr_seconds"), "usage.ppr_seconds")
    # When both appear, ppr_seconds is a subcomponent of own elapsed time.
    own_seconds = own_elapsed if own_elapsed is not None else own_ppr
    child_seconds = _sum_known(child["measured_graph_seconds"] for child in combined)
    measured = own_elapsed if wrapper and own_elapsed is not None else _sum_known((own_seconds, child_seconds))
    return {"counters": counters, "measured_graph_seconds": measured, "warnings": warnings}


def _round_observation(round_data):
    if not round_data:
        return None
    graph = _object(round_data.get("graph_search"), "graph_search", optional=True)
    events = graph.get("trace") or []
    if not isinstance(events, list) or not all(isinstance(event, dict) for event in events):
        raise ValueError("graph_search.trace must be an array of event objects or null")
    fallbacks = [event for event in events if event.get("event") == "fallback"]
    usage = aggregate_usage(graph.get("usage"), fallback_events=fallbacks)
    selected = graph.get("selected_paths") or []
    if not isinstance(selected, list):
        raise ValueError("graph_search.selected_paths must be an array or null")
    reason = graph.get("fallback_reason")
    stop = graph.get("stop_reason") or ""
    if not isinstance(stop, str) or (reason is not None and not isinstance(reason, str)):
        raise ValueError("Graph stop/fallback reasons must be strings or null")
    actual_fallback = bool(fallbacks) or stop.startswith("fallback_") or stop == "dpr_fallback"
    if (round_data.get("final_source") == "dpr_fallback"
            and round_data.get("repair_applied") is not False
            and round_data.get("graph_search_mode", "ppr") == "ppr"):
        actual_fallback = True
    counters = usage["counters"]
    executed = ((counters["llm_calls"] or 0) > 0 or (counters["tool_calls"] or 0) > 0
                or any(event.get("event") in ("llm_response", "llm_error", "action") for event in events))
    uncached_tokens = []
    for event in events:
        if event.get("event") == "llm_response" and event.get("tokens_reported") is True:
            tokens = _object(event.get("tokens"), "llm_response.tokens")
            total = _numeric(tokens.get("total_tokens"), "llm_response.total_tokens")
            if total is None:
                total = _sum_known((_numeric(tokens.get("prompt_tokens"), "prompt_tokens"),
                                    _numeric(tokens.get("completion_tokens"), "completion_tokens")))
            if event.get("cache_hit") is True:
                uncached_tokens.append(0)
            elif event.get("cache_hit") is False and total is not None:
                uncached_tokens.append(total)
    return {"usage": usage, "agent_executed": executed, "fallback": actual_fallback,
            "fallback_reason": (reason or next((event.get("reason") for event in fallbacks if event.get("reason")), None)
                                or stop or "unspecified") if actual_fallback else None,
            "selected_paths": selected, "stop_reason": stop,
            "uncached_reported_tokens": _sum_known(uncached_tokens)}


def _validate_rows(rows):
    if not isinstance(rows, list):
        raise ValueError("Trace must be an array of rows")
    _finite_tree(rows, "trace")
    for index, row in enumerate(rows):
        _object(row, f"trace[{index}]")
        if not isinstance(row.get("query"), str):
            raise ValueError(f"trace[{index}].query must be a string")
        first = _object(row.get("round1"), f"trace[{index}].round1")
        if first.get("query") is not None and first["query"] != row["query"]:
            raise ValueError(f"Round-1 query mismatch at row {index}")
        _docs(first.get("final_top5"), "round1.final_top5")
        _docs(row.get("final_top5"), "final_top5")
        _object(row.get("round2"), "round2", optional=True)
        _object(row.get("judge1"), "judge1", optional=True)


def _align_gold(rows, gold):
    if not isinstance(gold, list) or len(gold) != len(rows):
        raise ValueError("Gold and trace lengths must match exactly")
    _finite_tree(gold, "gold")
    named = bool(gold) and isinstance(gold[0], dict)
    aligned = []
    for index, (row, item) in enumerate(zip(rows, gold)):
        if named:
            if not isinstance(item, dict) or item.get("query") != row["query"]:
                raise ValueError(f"Gold query/order mismatch at row {index}")
            if "gold_docs" not in item:
                raise ValueError(f"Missing gold_docs at row {index}")
            item = item["gold_docs"]
        elif isinstance(item, dict):
            raise ValueError("Do not mix keyed and positional gold formats")
        aligned.append(set(_docs(item, f"gold[{index}]", optional=True)))
    return aligned, "query_and_order_checked" if named else "positional_length_checked"


def _quality(gold, retrieved):
    if not gold:
        return {"recall": None, "full_recall": None, "missing": None}
    missing = len(gold - set(retrieved[:5]))
    return {"recall": (len(gold) - missing) / len(gold), "full_recall": int(missing == 0), "missing": missing}


def _paired(per_query, baseline_rows, gold, mode, samples, seed):
    deltas = {"avg_recall@5": [], "full_recall@5": []}
    for record, row, references in zip(per_query, baseline_rows, gold):
        docs = row["round1"]["final_top5"] if mode == "round1" else row["final_top5"]
        base = _quality(references, _top5(docs, "baseline.final_top5"))
        if references:
            deltas["avg_recall@5"].append(record["final"]["recall"] - base["recall"])
            deltas["full_recall@5"].append(record["final"]["full_recall"] - base["full_recall"])
        record["baseline"] = base
    n = len(deltas["avg_recall@5"])
    rng = random.Random(seed)
    draws = {name: [] for name in deltas}
    if n:
        for _ in range(samples):
            indices = [rng.randrange(n) for _ in range(n)]
            for name, values in deltas.items():
                draws[name].append(sum(values[index] for index in indices) / n)
    result = {"baseline_mode": mode, "paired_queries": n,
              "bootstrap_samples": samples, "seed": seed, "confidence_level": 0.95, "metrics": {}}
    for name, values in deltas.items():
        result["metrics"][name] = {
            "mean_delta": sum(values) / n if n else None,
            "bootstrap_ci95": [_percentile(draws[name], 0.025), _percentile(draws[name], 0.975)] if draws[name] else None,
            "wins": sum(value > 0 for value in values), "ties": sum(value == 0 for value in values),
            "losses": sum(value < 0 for value in values),
        }
    return result


def evaluate_repair(trace_rows, gold, baseline_rows=None, *, baseline_mode="final", bootstrap_samples=1000, seed=17):
    """Return JSON-safe metrics; empty-gold queries remain in usage/coverage stats."""
    _validate_rows(trace_rows)
    references, alignment = _align_gold(trace_rows, gold)
    if baseline_mode not in ("final", "round1"):
        raise ValueError("baseline_mode must be final or round1")
    if type(bootstrap_samples) is not int or bootstrap_samples < 0 or type(seed) is not int:
        raise ValueError("bootstrap_samples must be a non-negative integer and seed an integer")
    if baseline_rows is None and baseline_mode == "round1":
        baseline_rows = trace_rows
    if baseline_rows is not None:
        _validate_rows(baseline_rows)
        if len(baseline_rows) != len(trace_rows) or any(left["query"] != right["query"] for left, right in zip(trace_rows, baseline_rows)):
            raise ValueError("Baseline query order and length must exactly match the candidate trace")

    per_query, reasons, warnings, stops = [], Counter(), Counter(), Counter()
    for index, (row, required) in enumerate(zip(trace_rows, references)):
        first = _quality(required, _top5(row["round1"]["final_top5"], "round1.final_top5"))
        final = _quality(required, _top5(row["final_top5"], "final_top5"))
        observations = [_round_observation(row["round1"]), _round_observation(row.get("round2"))]
        active = [observation for observation in observations if observation is not None]
        second = observations[1]
        values = {name: _sum_known(obs["usage"]["counters"][name] for obs in active) for name in COUNTERS}
        values["measured_graph_seconds"] = _sum_known(obs["usage"]["measured_graph_seconds"] for obs in active)
        values["uncached_reported_tokens"] = _sum_known(obs["uncached_reported_tokens"] for obs in active)
        if values["llm_calls"] is not None and values["cache_hits"] is not None:
            if values["cache_hits"] > values["llm_calls"]:
                raise ValueError("cache_hits cannot exceed llm_calls")
            values["noncache_llm_attempts"] = values["llm_calls"] - values["cache_hits"]
        else:
            values["noncache_llm_attempts"] = None
        for observation in active:
            warnings.update(observation["usage"]["warnings"])
            if observation["fallback"]:
                reasons[observation["fallback_reason"]] += 1
        mode = row.get("graph_search_mode") or (row.get("round2") or {}).get("graph_search_mode")
        requested = row.get("round2") is not None and mode in ("agent", "hybrid")
        committed = bool(second and second["selected_paths"])
        protected = _docs((row.get("round2") or {}).get("path_protected_docs"), "path_protected_docs", optional=True)
        retained = committed and bool(protected) and set(protected) <= set(row["final_top5"][:5])
        executed = bool(second and second["agent_executed"])
        if requested and second:
            stops[second["stop_reason"] or "unspecified"] += 1
        per_query.append({
            "index": index, "query": row["query"], "round1": first, "final": final,
            "has_gold": bool(required), "second_round": row.get("round2") is not None,
            "agent_requested": requested, "agent_executed": executed,
            "fallback": any(obs["fallback"] for obs in active),
            "round2_fallback": bool(second and second["fallback"]),
            "committed_path_set": committed, "path_set_retained": retained if committed else None,
            "path_sources_missing": committed and not protected, "usage": values,
        })

    total = len(per_query)
    eligible = [row for row in per_query if row["has_gold"]]
    miss1 = [row for row in eligible if row["round1"]["missing"] == 1]
    miss0 = [row for row in eligible if row["round1"]["missing"] == 0]
    committed = [row for row in per_query if row["committed_path_set"]]
    second_count = sum(row["second_round"] for row in per_query)
    requested_count = sum(row["agent_requested"] for row in per_query)
    quality = {}
    for stage in ("round1", "final"):
        quality[stage] = {"avg_recall@5": sum(row[stage]["recall"] for row in eligible) / len(eligible) if eligible else None,
                          "full_recall@5": sum(row[stage]["full_recall"] for row in eligible) / len(eligible) if eligible else None,
                          "denominator": len(eligible)}
    usage_fields = COUNTERS + ("measured_graph_seconds", "uncached_reported_tokens", "noncache_llm_attempts")
    report = {
        "schema_version": 1, "total_queries": total, "empty_gold_queries": total - len(eligible),
        "gold_alignment": alignment, "quality": quality,
        "repair": {"miss1_to_miss0": _rate(sum(row["final"]["missing"] == 0 for row in miss1), len(miss1)),
                   "miss0_damage": _rate(sum(row["final"]["missing"] > 0 for row in miss0), len(miss0))},
        "coverage": {
            "second_round": _rate(second_count, total),
            "agent_requested": _rate(requested_count, total),
            "agent_executed": _rate(sum(row["agent_executed"] for row in per_query), total),
            "agent_execution_given_requested": _rate(sum(row["agent_executed"] and row["agent_requested"] for row in per_query), requested_count),
            "fallback_queries": _rate(sum(row["fallback"] for row in per_query), total),
            "round2_fallback": _rate(sum(row["round2_fallback"] for row in per_query), second_count),
            "committed_path_set_retention": _rate(sum(row["path_set_retained"] is True for row in committed), len(committed)),
            "commits_missing_source_docs": sum(row["path_sources_missing"] for row in per_query),
        },
        "fallback_reasons_by_round": dict(sorted(reasons.items())), "agent_stop_reasons": dict(sorted(stops.items())),
        "usage": {name: _distribution([row["usage"][name] for row in per_query], total) for name in usage_fields},
        "warnings": dict(sorted(warnings.items())),
        "metric_notes": [
            "Passages match by exact text; duplicate gold passages are counted once. Empty-gold queries are excluded only from quality/repair denominators.",
            "Path-set retention measures preservation of recorded source passages, not path factual correctness; no gold reasoning paths are assumed.",
            "Usage sums only recorded graph-search components, including failed/fallback samples. Missing observations are null, not zero. This is not end-to-end latency or complete filter/judge/QA cost.",
            "Reported tokens/LLM calls may include cache hits. uncached_reported_tokens uses per-response cache flags; failures may have only estimates. Estimates and budget tokens are reported separately, never priced in dollars.",
            "Positional gold has length validation only; its query order must be supplied correctly by the caller. Named gold and baseline traces are checked without reordering.",
        ],
        "per_query": per_query,
    }
    if baseline_rows is not None:
        report["paired_comparison"] = _paired(per_query, baseline_rows, references, baseline_mode, bootstrap_samples, seed)
    return report
