#!/usr/bin/env python3
"""Resumable, serial, frozen-input thesis experiments. No API calls without --live.

Examples (from repository root):
  python scripts/run_thesis_experiments.py --run-dir result_outputs/thesis-dev --start 17 --stop 18 --datasets 2wikimultihopqa --extra-n 1 --live
  python scripts/run_thesis_experiments.py --run-dir result_outputs/thesis-main --live
  python scripts/run_thesis_experiments.py --run-dir result_outputs/thesis-main --live --resume
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
SCHEMA = "thesis-experiment-v1"
CORE = ("round1", "ppr", "rerank", "static", "agent", "agent_no_fallback", "agent_unprotected_merge")
EXTRAS = ("agent_mask_predicate", "agent_steps4", "agent_steps12")
BUDGET = dict(max_steps=8, max_tool_calls=16, max_expansions=64, max_neighbors=8,
              max_path_length=4, max_seconds=60, max_retries=2, max_tokens=None)


def serialize(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(type(value).__name__)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, default=serialize)


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(encoded(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def lock(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another experiment already holds this run/GPU lock") from exc
        yield


def snapshot(dataset):
    directory = ROOT / "outputs" / dataset / "deepseek-chat_nvidia_NV-Embed-v2"
    paths = sorted(directory.glob("*_embeddings/*.parquet"))
    if len(paths) != 3 or not (directory / "graph.pickle").exists():
        raise RuntimeError("Existing graph and all three corpus embedding stores are required")
    return {str(path.relative_to(ROOT)): [path.stat().st_size, path.stat().st_mtime_ns] for path in paths}


def resources():
    result = {"time": time.time()}
    for key in ("memory.current", "memory.max"):
        path = Path("/sys/fs/cgroup") / key
        if path.exists():
            value = path.read_text().strip()
            result[key] = int(value) if value.isdigit() else value
    try:
        value = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                                         "--format=csv,noheader,nounits"], text=True, timeout=5)
        result["gpu"] = [float(item) for item in value.strip().split(",")]
    except (OSError, ValueError, subprocess.SubprocessError):
        result["gpu"] = None
    return result


def compact_trace(row):
    row = deepcopy(row)
    for obj in (row, row.get("round1"), row.get("round2")):
        if obj:
            for key in ("final_docs", "final_scores"):
                if isinstance(obj.get(key), list):
                    obj[key] = obj[key][:5]
    return row


def frozen_request(request):
    value = asdict(request)
    value.pop("budget")
    return value


def unprotected_merge(row):
    """Offline re-fusion diagnostic ONLY; same legal committed paths, no new search."""
    from SRgraphrag.retrieval.evidence import apply_evidence_injection_top5, apply_guard_top5_protect_evidence, dedup_preserve
    row = deepcopy(row)
    r2 = row.get("round2")
    if not r2 or not r2["graph_search"]["selected_paths"]:
        return row
    ranked = list(r2["path_protected_docs"])
    ranked += [doc for doc in r2["dpr_top5"] if doc not in ranked]
    protected = row["judge1"].get("evidence_docs", [])
    top5, _ = apply_evidence_injection_top5(ranked[:5], protected, ranked)
    top5, _ = apply_guard_top5_protect_evidence(top5, r2.get("dpr1"), set(protected))
    top5 = dedup_preserve(top5)
    top5 += [doc for doc in ranked if doc not in top5]
    top5 = (top5 + [""] * 5)[:5]
    r2["final_top5"] = row["final_top5"] = top5
    r2["final_docs"] = row["final_docs"] = top5
    r2["final_scores"] = row["final_scores"] = [None] * 5
    row["diagnostic"] = "Offline re-fusion without new-path protection; NOT an independent Agent search."
    return row


class Meter:
    def __init__(self):
        self.events = []
        self.stage = "initialization"

    def wrap(self, function, kind):
        def call(*args, **kwargs):
            started = time.monotonic()
            item = {"stage": self.stage, "kind": kind, "time": time.time()}
            try:
                result = function(*args, **kwargs)
                if isinstance(result, tuple) and len(result) == 3:
                    item.update(metadata=result[1], cache_hit=bool(result[2]))
                return result
            except Exception as exc:
                item["error_type"] = type(exc).__name__
                raise
            finally:
                item["seconds"] = time.monotonic() - started
                self.events.append(item)
        return call


def finish_report(directory, dataset, samples, stop, *, final=False):
    """Evaluation is downstream only; no labels are passed to search or QA."""
    from main import get_gold_answers, get_gold_docs
    from SRgraphrag.evaluation.repair_eval import evaluate_repair
    from SRgraphrag.evaluation.qa_eval import QAExactMatch, QAF1Score
    grouped = {}
    for path in sorted((directory / "questions").glob("*/results/*.json")):
        item = read(path)
        grouped.setdefault(item["variant"], []).append(item)
    summary = {"schema": SCHEMA, "dataset": dataset, "final": final, "time": time.time(), "variants": {}}
    baseline = {item["source_index"]: item for item in grouped.get("ppr", [])}
    for variant, items in grouped.items():
        items.sort(key=lambda item: item["source_index"])
        rows = [item["trace"] for item in items]
        selected = [samples[item["source_index"]] for item in items]
        gold_docs = get_gold_docs(selected, dataset)
        gold_answers = [list(value) for value in get_gold_answers(selected)]
        gold = [{"query": row["query"], "gold_docs": docs} for row, docs in zip(rows, gold_docs)]
        matched = all(item["source_index"] in baseline for item in items)
        compared = [baseline[item["source_index"]]["trace"] for item in items] if matched else None
        report = evaluate_repair(rows, gold, compared, bootstrap_samples=0)
        predictions = [item["qa"]["answer"] for item in items]
        em, em_rows = QAExactMatch().calculate_metric_scores(gold_answers, predictions)
        f1, f1_rows = QAF1Score().calculate_metric_scores(gold_answers, predictions)
        report["qa"] = {**em, **f1, "denominator": len(items),
                        "failed_or_skipped": sum(bool(item["qa"].get("error")) for item in items)}
        per_query = []
        for item, details, e, f in zip(items, report["per_query"], em_rows, f1_rows):
            per_query.append({"source_index": item["source_index"], "sample_id": item["sample_id"],
                              "query": item["trace"]["query"], "triggered": item["triggered"],
                              "round1_missing": details["round1"]["missing"],
                              "recall": details["final"]["recall"], "full": details["final"]["full_recall"],
                              "em": e["ExactMatch"], "f1": f["F1"],
                              "type": samples[item["source_index"]].get("type"),
                              "qa_input_chars": sum(map(len, item["trace"]["final_top5"]))})
        report["paired_qa_and_quality_rows"] = per_query
        costs = []
        for item in items:
            qdir = directory / "questions" / f"{item['source_index']:04d}"
            first = read(qdir / "round1.json")
            judge_record = read(qdir / "judge.json")
            events = list(first.get("events", []))
            if item["triggered"] and variant != "round1":
                for frozen in sorted((qdir / "filters").glob("*.json")):
                    events.extend(read(frozen).get("events", []))
            event_item = read(qdir / "results/agent.json") if item["diagnostic_reuse"] else item
            events += [event for event in event_item["events"] if event["kind"] == "graph_llm"]
            events += item["qa"].get("events", [])
            tokens = 0
            missing = 0
            for event in events:
                metadata = event.get("metadata", {})
                usage = metadata.get("usage", metadata)
                if "prompt_tokens" in usage and "completion_tokens" in usage:
                    tokens += usage["prompt_tokens"] + usage["completion_tokens"]
                else:
                    missing += 1
            judge_tokens = judge_record["result"].get("_usage") if variant != "round1" else None
            if variant != "round1":
                if isinstance(judge_tokens, dict) and "total_tokens" in judge_tokens:
                    tokens += judge_tokens["total_tokens"]
                else:
                    missing += 1
            costs.append({"source_index": item["source_index"], "triggered": item["triggered"],
                          "logical_reported_tokens": tokens, "missing_usage_events": missing,
                          "logical_model_calls": len(events) + int(variant != "round1"),
                          "cached_model_calls_in_original_observations": sum(bool(event.get("cache_hit")) for event in events),
                          "round1_seconds": first["seconds"], "judge_seconds": judge_record["seconds"] if variant != "round1" else 0,
                          "round2_seconds_measured_with_frozen_inputs": item["retrieval_seconds_measured"],
                          "reader_seconds_first_execution_of_this_input": item["qa"]["seconds"],
                          "qa_reused": item["qa_reused"], "diagnostic_reuse": item["diagnostic_reuse"]})
        report["stage_cost_rows"] = costs
        report["stage_cost_notes"] = [
            "Logical tokens include shared stages once per compared system, NOT a bill for this experiment suite.",
            "Raw events and cache flags identify actual new API calls. SDK internal retries may not have usage metadata.",
            "Round-two timings are measured with frozen upstream inputs; they are not standalone cold end-to-end latency.",
            "Diagnostic variants reuse the original Agent attempt and must not be counted as additional searches."]
        report["path_audit"] = {"committed_questions": sum(bool(item.get("path_audit", {}).get("path_lengths")) for item in items),
                                "missing_source_questions": sum(bool(item.get("path_audit", {}).get("missing_source_facts")) for item in items),
                                "all_path_lengths": [length for item in items for length in item.get("path_audit", {}).get("path_lengths", [])]}
        trigger = [item for item in per_query if item["triggered"]]
        report["fixed_trigger_subset"] = {"count": len(trigger), **{
            key: sum(item[key] for item in trigger) / len(trigger) if trigger else None
            for key in ("recall", "full", "em", "f1")}}
        save(directory / "reports" / f"{variant}.json", report)
        summary["variants"][variant] = {"count": len(items), "quality": report["quality"]["final"],
                                        "qa": report["qa"], "triggered": report["fixed_trigger_subset"],
                                        "coverage": report["coverage"]}
    if final and "ppr" in grouped:
        import numpy as np
        base = {row["source_index"]: row for row in read(directory / "reports/ppr.json")["paired_qa_and_quality_rows"]}
        for variant in grouped:
            report_path = directory / "reports" / f"{variant}.json"
            report = read(report_path)
            pairs = report["paired_qa_and_quality_rows"]
            if not pairs or not all(row["source_index"] in base for row in pairs):
                continue
            values = np.array([[row[key] - base[row["source_index"]][key] for key in ("recall", "full", "em", "f1")]
                               for row in pairs], dtype=float)
            rng = np.random.default_rng(20260919)
            draws = np.concatenate([values[rng.integers(0, len(values), (250, len(values)))].mean(axis=1) for _ in range(40)])
            report["paired_bootstrap"] = {"baseline": "ppr", "samples": 10000, "seed": 20260919, "n": len(pairs),
                "metrics": {key: {"difference": float(values[:, i].mean()),
                                   "ci95": np.quantile(draws[:, i], [0.025, 0.975]).tolist()}
                            for i, key in enumerate(("recall", "full", "em", "f1"))}}
            save(report_path, report)
    save(directory / "summary.json", summary)
    return summary


def worker(args):
    from SRgraphrag.SRgraphrag import SRgraphrag
    from SRgraphrag.utils.config_utils import BaseConfig
    from SRgraphrag.utils.misc_utils import QuerySolution
    from SRgraphrag.graph.schema import passage_id
    from SRgraphrag.retrieval.agent_llm import AgentLLMAdapter
    from SRgraphrag.retrieval.types import GraphSearchResult
    from SRgraphrag.retrieval.dispatch import get_relation_index
    from SRgraphrag.experiments.controls import PredicateMaskedLLM, rerank_passages, static_graph_search
    dispatch = importlib.import_module("SRgraphrag.retrieval.dispatch")
    original_search = dispatch.search_graph
    dataset = args.worker
    directory = args.run_dir / dataset
    directory.mkdir(parents=True, exist_ok=True)
    samples = read(ROOT / "reproduce/dataset" / f"{dataset}.json")
    corpus = read(ROOT / "reproduce/dataset" / f"{dataset}_corpus.json")
    stop = min(args.stop, len(samples))
    before = snapshot(dataset)
    status = {"status": "loading", "dataset": dataset, "pid": os.getpid(), "start": args.start, "stop": stop,
              "embedding_before": before, "started_at": time.time()}
    save(directory / "status.json", status)
    config = BaseConfig(save_dir=str(ROOT / "outputs" / dataset), llm_base_url="https://api.deepseek.com/v1",
                        llm_name="deepseek-chat", dataset=dataset, embedding_model_name="nvidia/NV-Embed-v2",
                        force_index_from_scratch=False, force_openie_from_scratch=False,
                        rerank_dspy_file_path="src/SRgraphrag/prompts/dspy_prompts/filter_deepseek3.2-Instruct.json",
                        retrieval_top_k=200, linking_top_k=30, max_qa_steps=3, qa_top_k=5,
                        graph_type="facts_and_sim_passage_node_unidirectional", embedding_batch_size=4,
                        max_new_tokens=None, corpus_len=len(corpus), openie_mode="online")
    rag = SRgraphrag(global_config=config)
    rag.global_config.output_dir = str(directory)
    rag.prepare_retrieval_objects()  # Deliberately never call index() or rebuild embeddings.
    expected = {passage_id(f"{doc['title']}\n{doc['text']}") for doc in corpus}
    if expected != set(rag.passage_node_keys):
        raise RuntimeError("Corpus and cached passage IDs differ; refusing implicit re-indexing")
    meter = Meter()
    original_infer = rag.llm_model.infer
    rag.llm_model.infer = meter.wrap(original_infer, "qa")
    rag.rerank_filter.llm_infer_fn = meter.wrap(original_infer, "fact_filter")
    adapter = meter.wrap(AgentLLMAdapter(rag.llm_model, args.run_dir / "agent_cache.sqlite"), "graph_llm")
    rag._agent_llm = adapter
    original_filter = rag.rerank_facts
    status.update(status="running", loaded_at=time.time())
    save(directory / "status.json", status)

    for source_index in range(args.start, stop):
        sample = samples[source_index]
        query = sample["question"]
        sample_id = str(sample.get("_id", sample.get("id", source_index)))
        qdir = directory / "questions" / f"{source_index:04d}"
        qdir.mkdir(parents=True, exist_ok=True)
        identity = {"query": query, "source_index": source_index, "sample_id": sample_id}
        if (qdir / "input.json").exists() and read(qdir / "input.json") != identity:
            raise RuntimeError("Checkpoint identity does not match the current dataset")
        save(qdir / "input.json", identity)
        if args.dev_replay_root and not (qdir / "round1.json").exists():
            from main import load_round1_replay
            historical = load_round1_replay(str(args.dev_replay_root / dataset / "ITER_cap4_agent/retrieval_trace.jsonl"))[source_index]
            if historical["query"] != query:
                raise RuntimeError("Development replay question order differs")
            save(qdir / "round1.json", {"result": historical["round1"], "seconds": None, "events": [], "historical_replay": True})
            save(qdir / "judge.json", {"result": historical["judge1"], "seconds": None, "historical_replay": True})
        variants = list(CORE) + (list(EXTRAS) if source_index < args.start + args.extra_n else [])
        if all((qdir / "results" / f"{variant}.json").exists() for variant in variants):
            continue
        status.update(source_index=source_index, sample_id=sample_id, phase="round1", updated_at=time.time())
        save(directory / "status.json", status)
        meter.events = []
        meter.stage = "round1"
        query_started = time.monotonic()
        rag.get_query_embeddings([query])
        if not (qdir / "round1.json").exists():
            started = time.monotonic()
            r1 = rag.retrieve_full_once(query, num_to_retrieve=200, subject_cap=4)
            if any(event.get("error_type") for event in meter.events):
                raise RuntimeError("Round-one provider error; checkpoint not frozen")
            save(qdir / "round1.json", {"result": r1, "seconds": time.monotonic() - started, "events": meter.events})
        r1 = read(qdir / "round1.json")["result"]
        if not (qdir / "judge.json").exists() or read(qdir / "judge.json")["result"].get("_error"):
            started = time.monotonic()
            status.update(phase="judge", updated_at=time.time())
            save(directory / "status.json", status)
            judge = rag.judge_answerability_and_bridge(queries=[query], top5_docs_list=[r1["final_top5"]],
                                                       judge_model_name="deepseek-reasoner", judge_concurrency=1, judge_batch_size=1)[0]
            judge.pop("_raw", None)
            save(qdir / "judge.json", {"result": judge, "seconds": time.monotonic() - started})
            if judge.get("_error"):
                raise RuntimeError("Judge failed; refusing to silently freeze a no-trigger decision")
        judge = read(qdir / "judge.json")["result"]
        replay = [{"query": query, "round1": r1, "judge1": judge}]
        triggered = bool(not judge.get("can_answer") and judge.get("bridge_possible") and str(judge.get("bridge_question", "")).strip())
        state = {"variant": "ppr", "prior": None}

        def freeze_filter(query, scores, instruction=None):
            key = digest({"query": query, "instruction": instruction})
            path = qdir / "filters" / f"{key}.json"
            if path.exists():
                saved = read(path)
                if saved["fact_scores_sha256"] != digest(scores):
                    raise RuntimeError("Second-round fact scores changed across strategies")
                return deepcopy(saved["result"])
            previous = len(meter.events)
            result = original_filter(query, scores, instruction)
            if any(item.get("error_type") for item in meter.events[previous:]):
                raise RuntimeError("Fact filter provider failure; refusing to freeze an empty filter")
            save(path, {"result": result, "fact_scores_sha256": digest(scores), "events": meter.events[previous:]})
            return result

        def experiment_search(owner, request, mode="ppr", fallback="ppr"):
            if request.round_index != 2:
                return original_search(owner, request, mode, fallback)
            shared = frozen_request(request)
            request_path = qdir / "second_round_request.json"
            if request_path.exists() and read(request_path) != shared:
                raise RuntimeError("Second-round seeds/protected evidence differ across strategies")
            save(request_path, shared)
            variant = state["variant"]
            if variant == "ppr":
                result = original_search(owner, request, "ppr", "ppr")
                save(qdir / "ppr_graph.json", result.to_dict())
                return result
            if variant in ("rerank", "static"):
                prior = GraphSearchResult(**read(qdir / "ppr_graph.json")) if (qdir / "ppr_graph.json").exists() else original_search(owner, request, "ppr", "ppr")
                result = (rerank_passages(owner, request, prior, adapter) if variant == "rerank"
                          else static_graph_search(get_relation_index(owner), request, adapter))
                result.usage["shared_ppr_candidate_cost"] = prior.usage if variant == "rerank" else None
                if result.ranked_passage_ids:
                    return result
                return replace(prior, stop_reason="fallback_" + prior.stop_reason,
                               fallback_reason=result.fallback_reason or result.stop_reason,
                               usage={"control": result.usage, "fallback": prior.usage},
                               trace=result.trace + [{"event": "fallback", "requested": "ppr", "used": prior.stop_reason}])
            return original_search(owner, request, "agent", "ppr")

        for variant in variants:
            result_path = qdir / "results" / f"{variant}.json"
            if result_path.exists():
                continue
            state["variant"] = variant
            status.update(phase=variant, updated_at=time.time())
            save(directory / "status.json", status)
            meter.stage, meter.events = variant, []
            started = time.monotonic()
            diagnostic_reuse = False
            if variant == "round1" or not triggered:
                trace = {"query": query, "round1": r1, "judge1": judge, "round2": None, "round_used": 1,
                         "final_top5": r1["final_top5"], "final_docs": r1["final_top5"],
                         "final_scores": r1["final_scores"][:5] if r1["final_scores"] else None,
                         "score_round": 1, "graph_search_mode": "ppr" if variant in ("round1", "ppr") else "agent"}
            elif variant in ("agent_no_fallback", "agent_unprotected_merge"):
                trace = deepcopy(read(qdir / "results/agent.json")["trace"])
                diagnostic_reuse = True
                r2 = trace.get("round2") or {}
                if variant == "agent_no_fallback" and not r2.get("graph_search", {}).get("selected_paths"):
                    trace.update(final_top5=r1["final_top5"], final_docs=r1["final_top5"], final_scores=r1["final_scores"][:5] if r1["final_scores"] else None, score_round=1)
                    trace["diagnostic"] = "Same Agent attempt; failure retains round1 instead of executing fallback."
                    search = r2["graph_search"]
                    search["usage"] = search.get("usage", {}).get("agent", search.get("usage", {}))
                    search["trace"] = [event for event in search.get("trace", []) if event.get("event") != "fallback"]
                    search["stop_reason"] = search.get("fallback_reason") or "no_committed_path"
                    search.update(ranked_passage_ids=[], scores=[], score_sources=[])
                    r2.update(repair_applied=False, final_top5=[""] * 5, final_docs=[""] * 5,
                              final_scores=[None] * 5, final_source="no_repair")
                elif variant == "agent_unprotected_merge":
                    trace = unprotected_merge(trace)
            else:
                budget = dict(BUDGET)
                if variant == "agent_steps4":
                    budget["max_steps"] = 4
                if variant == "agent_steps12":
                    budget["max_steps"] = 12
                rag._agent_llm = PredicateMaskedLLM(adapter) if variant == "agent_mask_predicate" else adapter
                with patch.object(rag, "rerank_facts", freeze_filter), patch.object(dispatch, "search_graph", experiment_search):
                    rag.retrieve([query], gold_docs=None, num_to_retrieve=200, dataset_name=dataset,
                                 subject_cap=4, graph_search_mode="ppr" if variant == "ppr" else "agent",
                                 agent_budget=budget, agent_fallback="ppr", round1_replay=replay)
                trace = rag.last_retrieval_trace[0]
                rag._agent_llm = adapter
            retrieval_seconds = time.monotonic() - started
            trace = compact_trace(trace)
            trace["experiment_variant"] = variant
            qa_key = digest({"query": query, "docs": trace["final_top5"], "dataset": dataset, "model": config.llm_name})
            qa_path = qdir / "qa" / f"{qa_key}.json"
            meter.stage = variant + "/qa"
            qa_started = time.monotonic()
            qa_reused = qa_path.exists()
            if not qa_reused:
                qa_event_start = len(meter.events)
                solutions, responses, metadata = rag.qa([QuerySolution(question=query, docs=trace["final_top5"], doc_scores=[None] * 5)])
                qa = {"answer": solutions[0].answer, "response": responses[0], "metadata": metadata[0],
                      "seconds": time.monotonic() - qa_started, "error": bool(metadata[0].get("skipped")),
                      "events": meter.events[qa_event_start:]}
                save(qa_path, qa)
            qa = read(qa_path)
            paths = (trace.get("round2") or {}).get("graph_search", {}).get("selected_paths", [])
            path_audit = {"path_lengths": [len(path) for path in paths], "missing_source_facts": [], "structurally_verified": True}
            if paths:
                index = get_relation_index(rag)
                returned_ids = {passage_id(doc) for doc in trace["final_top5"] if doc}
                for path in paths:
                    previous = None
                    for step in path:
                        fact = index.facts[step["fact_id"]]
                        endpoints = (fact.subject_id, fact.object_id) if step["traversal_direction"] in ("out", "forward") else (fact.object_id, fact.subject_id)
                        if endpoints != (step["from_entity_id"], step["to_entity_id"]) or (previous is not None and previous != endpoints[0]):
                            raise RuntimeError("Committed path failed offline structural audit")
                        previous = endpoints[1]
                        if not returned_ids.intersection(fact.passage_ids):
                            path_audit["missing_source_facts"].append(step["fact_id"])
                if path_audit["missing_source_facts"] and variant != "agent_unprotected_merge":
                    raise RuntimeError("A committed path source was lost from final Top-5")
            save(result_path, {**identity, "variant": variant, "triggered": triggered, "trace": trace, "qa": qa,
                               "qa_reused": qa_reused, "diagnostic_reuse": diagnostic_reuse,
                               "retrieval_seconds_measured": retrieval_seconds, "events": meter.events, "path_audit": path_audit})
        status.update(completed_questions=source_index - args.start + 1, updated_at=time.time(),
                      last_question_seconds=time.monotonic() - query_started)
        save(directory / "status.json", status)
        print(encoded({"dataset": dataset, "source_index": source_index, "completed": True,
                       "triggered": triggered, "seconds": status["last_question_seconds"]}), flush=True)
        if (source_index - args.start + 1) % 25 == 0:
            finish_report(directory, dataset, samples, stop)
    finish_report(directory, dataset, samples, stop, final=True)
    after = snapshot(dataset)
    if before != after:
        raise RuntimeError("Corpus embedding snapshot changed during retrieval-only experiment")
    status.update(status="completed", finished_at=time.time(), embedding_after=after, corpus_embedding_unchanged=True)
    save(directory / "status.json", status)


def manifest(args):
    source_files = sorted((ROOT / "src/SRgraphrag").rglob("*.py")) + [Path(__file__), ROOT / "main.py"]
    source_files += sorted((ROOT / "src/SRgraphrag/prompts").rglob("*.json"))
    return {"schema": SCHEMA, "start": args.start, "stop": args.stop, "extra_n": args.extra_n,
            "dev_replay_root": str(args.dev_replay_root) if args.dev_replay_root else None,
            "datasets": args.datasets, "core": list(CORE), "extras": list(EXTRAS), "budget": BUDGET,
            "readme": "Indices [0,20) are development; [20,1000) is the main fixed set. No gold enters online retrieval.",
            "sources": {str(path.relative_to(ROOT)): fingerprint(path) for path in source_files},
            "data": {str(path.relative_to(ROOT)): fingerprint(path) for name in args.datasets
                     for path in (ROOT / "reproduce/dataset" / f"{name}.json", ROOT / "reproduce/dataset" / f"{name}_corpus.json")},
            "models": {"retrieval_and_reader": "deepseek-chat", "judge": "deepseek-reasoner", "embedding": "nvidia/NV-Embed-v2"},
            "note": "Alias model names may evolve at the provider; raw response metadata and run time are retained."}


def supervise(args):
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with lock(args.run_dir / "run.lock"):
        config = manifest(args)
        path = args.run_dir / "manifest.json"
        if path.exists():
            if not args.resume:
                raise RuntimeError("Existing run: inspect it, then use --resume with identical source/configuration")
            if read(path) != config:
                raise RuntimeError("Manifest changed; use a NEW directory, never mix code versions in one experiment")
        else:
            save(path, config)
        state = {"status": "running", "pid": os.getpid(), "started_at": time.time(), "datasets": {}}
        save(args.run_dir / "status.json", state)
        env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4",
                   CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        for dataset in args.datasets:
            item = {"status": "running", "started_at": time.time()}
            state["datasets"][dataset] = item
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--run-dir", str(args.run_dir),
                       "--worker", dataset, "--live", "--start", str(args.start), "--stop", str(args.stop),
                       "--extra-n", str(args.extra_n)]
            if args.dev_replay_root:
                command += ["--dev-replay-root", str(args.dev_replay_root)]
            with (args.run_dir / f"{dataset}.log").open("a") as log, (args.run_dir / "resources.jsonl").open("a") as resource_log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                item["pid"] = child.pid
                save(args.run_dir / "status.json", state)
                while True:
                    resource_log.write(encoded({**resources(), "dataset": dataset, "pid": child.pid}) + "\n")
                    resource_log.flush()
                    try:
                        code = child.wait(timeout=5)
                        break
                    except subprocess.TimeoutExpired:
                        pass
            item.update(status="completed" if code == 0 else "failed", return_code=code, finished_at=time.time())
            save(args.run_dir / "status.json", state)
            if code:
                state.update(status="failed", finished_at=time.time())
                save(args.run_dir / "status.json", state)
                raise SystemExit(code)
        state.update(status="completed", finished_at=time.time())
        save(args.run_dir / "status.json", state)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-dir", type=Path, required=True)
    result.add_argument("--datasets", nargs="+", choices=("musique", "2wikimultihopqa", "hotpotqa"), default=["musique", "2wikimultihopqa", "hotpotqa"])
    result.add_argument("--start", type=int, default=20)
    result.add_argument("--stop", type=int, default=1000)
    result.add_argument("--extra-n", type=int, default=200, help="Predeclared prefix size for predicate and step-budget variants")
    result.add_argument("--live", action="store_true")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--worker", choices=("musique", "2wikimultihopqa", "hotpotqa"))
    result.add_argument("--dev-replay-root", type=Path, help="Historical frozen smoke traces, development indices only")
    return result


def main():
    args = parser().parse_args()
    if not 0 <= args.start < args.stop <= 1000 or args.extra_n < 0:
        raise SystemExit("Require 0 <= start < stop <= 1000 and extra-n >= 0")
    if args.dev_replay_root and args.stop > 20:
        raise SystemExit("Historical smoke replay is restricted to development indices below 20")
    if args.dev_replay_root:
        args.dev_replay_root = args.dev_replay_root.resolve()
    args.run_dir = args.run_dir.resolve()
    os.chdir(ROOT)
    if not args.live:
        print(encoded(manifest(args)))
        return
    if not os.getenv("DEEPSEEK_API_KEY") or not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Required API credentials are not configured in the environment")
    if args.worker:
        with lock(ROOT / "outputs/.thesis_experiment_gpu.lock"):
            worker(args)
    else:
        supervise(args)


if __name__ == "__main__":
    main()
