"""Budgeted, closed-action graph retrieval with deterministic tool replay.

The LLM callback accepts chat messages and returns text or
``(text, metadata, cache_hit)``. Its adapter must honor the last message's
``llm_limits`` (completion tokens and timeout). The controller checks budgets
before and after every call, but cannot forcibly interrupt a synchronous callback.
No answer generation, arbitrary code execution, or gold-label access occurs here.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
import math
import time

from ..graph.tools import GraphTools
from ..prompts.templates.agent_graph_search import PROMPT_VERSION, SYSTEM_PROMPT
from .agent_actions import ACTION_SCHEMA_VERSION, ActionError, AgentBudget, parse_action
from .types import GraphSearchResult


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _record(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return value
    return str(value)


def _manifest(index):
    return _record(getattr(index, "manifest", {}))


def _nonnegative_count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _token_usage(metadata):
    """Read both CacheOpenAI's flat metadata and OpenAI-shaped usage dictionaries."""
    if not isinstance(metadata, dict):
        return {}, False
    source = metadata.get("usage") if isinstance(metadata.get("usage"), dict) else metadata
    prompt = source.get("prompt_tokens", source.get("input_tokens"))
    completion = source.get("completion_tokens", source.get("output_tokens"))
    total = source.get("total_tokens")
    valid = all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in (prompt, completion))
    # Every Agent call includes a non-empty system prompt. Zero input tokens
    # therefore indicates missing/placeholder metadata, not a free model call.
    valid = valid and prompt > 0
    if not valid:
        return {}, False
    result = {"prompt_tokens": prompt, "completion_tokens": completion,
              "total_tokens": max(prompt + completion, _nonnegative_count(total))}
    return result, True


class AgentGraphSearch:
    """Select only tool-committed, capacity-checked evidence paths."""

    def __init__(self, index, llm, budget=None):
        self.index = index
        self.llm = llm
        self.budget = budget

    def _tools(self, request, budget):
        return GraphTools(
            self.index, seed_entity_ids=request.seed_entity_ids,
            protected_passage_ids=request.protected_passage_ids,
            evidence_limit=request.evidence_limit, max_calls=budget.max_tool_calls,
            max_expansions=budget.max_expansions, max_neighbors=budget.max_neighbors,
            max_path_length=budget.max_path_length,
            allowed_entity_ids=request.candidate_entity_ids,
        )

    def _initial_context(self, request, budget):
        entities = getattr(self.index, "entities", {})
        seeds = []
        for entity_id in dict.fromkeys(request.seed_entity_ids):
            record = _record(entities[entity_id]) if entity_id in entities else {"entity_id": entity_id}
            score = request.seed_scores.get(entity_id)
            if isinstance(record, dict):
                record = dict(record)
                if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score):
                    record["prior_score"] = float(score)
            seeds.append(record)
        candidates = (None if request.candidate_entity_ids is None
                      else sorted(set(request.candidate_entity_ids)))
        # A hybrid region can contain hundreds of opaque IDs. They are not
        # directly accessible capabilities, so show only observed nodes and bind
        # the full region through a fingerprint instead of exhausting the prompt.
        candidate_region = None if candidates is None else {
            "entity_count": len(candidates),
            "fingerprint": sha256(_json(candidates).encode("utf-8")).hexdigest(),
        }
        return {
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
            "index_manifest": _manifest(self.index),
            "original_query": request.original_query,
            "retrieval_query": request.retrieval_query,
            "round_index": request.round_index,
            "seed_entities": seeds,
            "seed_fact_ids": request.seed_fact_ids,
            "protected_passage_ids": request.protected_passage_ids,
            "evidence_limit": request.evidence_limit,
            "candidate_region": candidate_region,
            "budget": budget.to_dict(),
        }

    @staticmethod
    def _finish(reason, trace, usage, started, tools=None, committed=None, budget_check=None):
        usage = dict(usage)
        if tools is not None:
            usage.update(tools.usage)
        usage["elapsed_seconds"] = max(0.0, time.monotonic() - started)
        paths = committed.get("selected_paths", []) if committed else []
        passages = committed.get("passage_ids", []) if committed else []
        trace.append({"event": "stop", "stop_reason": reason,
                      "selected_paths": paths, "passage_ids": passages})
        if budget_check is not None:
            trace[-1]["budget_check"] = budget_check
        return GraphSearchResult(
            ranked_passage_ids=passages,
            scores=[None] * len(passages),
            score_sources=["agent_path"] * len(passages),
            selected_paths=paths,
            stop_reason=reason,
            fallback_reason=None if committed else reason,
            usage=usage,
            trace=trace,
        )

    def search(self, request):
        started = time.monotonic()
        trace = []
        usage = {"llm_calls": 0, "cache_hits": 0, "prompt_tokens": 0,
                 "completion_tokens": 0, "total_tokens": 0, "estimated_tokens": 0,
                 "budget_tokens": 0, "retries": 0, "invalid_actions": 0,
                 "tool_failures": 0}
        try:
            budget = AgentBudget.from_values(self.budget, request.budget)
            tools = self._tools(request, budget)
            context = self._initial_context(request, budget)
        except (ActionError, ValueError, TypeError, KeyError) as exc:
            trace.append({"event": "configuration_error", "error_type": type(exc).__name__,
                          "message": str(exc)[:512]})
            return self._finish("invalid_configuration", trace, usage, started)
        trace.append({"event": "start", "context": context,
                      "candidate_entity_ids": request.candidate_entity_ids})
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _json(context)}]
        seen = set()
        visited_entity_ids = set()
        last_failure = None
        for step_index in range(budget.max_steps):
            remaining_seconds = budget.max_seconds - (time.monotonic() - started)
            if remaining_seconds <= 0:
                return self._finish("time_budget", trace, usage, started, tools)
            if usage["retries"] > budget.max_retries:
                return self._finish(last_failure or "retry_budget", trace, usage, started, tools)
            remaining_tokens = budget.max_tokens - usage["budget_tokens"]
            if tools.usage.get("remaining_calls", 1) == 0:
                return self._finish("tool_budget", trace, usage, started, tools)
            state = {
                "step": step_index + 1,
                "frontier_entity_ids": sorted(tools.exposed_entity_ids - visited_entity_ids),
                "visited_entity_ids": sorted(visited_entity_ids),
                "observed_fact_ids": sorted(tools.exposed_fact_ids),
                "remaining_budget": {
                    "steps": budget.max_steps - step_index,
                    "tokens": remaining_tokens,
                    "tool_calls": tools.usage.get("remaining_calls", budget.max_tool_calls),
                    "expansions": tools.usage.get("remaining_expansions", budget.max_expansions),
                    "retries": max(0, budget.max_retries - usage["retries"]),
                },
                "llm_limits": {
                    "max_completion_tokens": 1024,
                    # The adapter excludes this transport-only deadline from the
                    # semantic prompt/cache identity, while honoring the timeout.
                    "timeout_seconds": max(0.001, min(budget.max_seconds, remaining_seconds)),
                },
            }
            call_messages = messages + [{"role": "user", "content": _json(state)}]
            # UTF-8 bytes plus chat framing conservatively bound input tokens for
            # the supported byte-tokenized models without a tokenizer dependency.
            # Count the full state (including frontier IDs), not just history.
            prompt_estimate = sum(len(message["content"].encode("utf-8")) + 16 for message in call_messages) + 256
            if remaining_tokens <= prompt_estimate:
                return self._finish("token_budget", trace, usage, started, tools, budget_check={
                    "stage": "before_llm", "limit_tokens": budget.max_tokens,
                    "used_budget_tokens": usage["budget_tokens"],
                    "remaining_tokens": remaining_tokens,
                    "next_prompt_estimate": prompt_estimate,
                    "estimator": "utf8_bytes_with_framing_upper_bound",
                })
            state["llm_limits"]["max_completion_tokens"] = min(1024, remaining_tokens - prompt_estimate)
            call_messages[-1] = {"role": "user", "content": _json(state)}
            usage["llm_calls"] += 1
            try:
                reply = self.llm(call_messages)
                if isinstance(reply, tuple) and len(reply) == 3:
                    response, metadata, cache_hit = reply
                else:
                    response, metadata, cache_hit = reply, {}, False
            except Exception as exc:
                # Do not copy provider exception strings: they can contain keys,
                # URLs with credentials, or full private request bodies.
                trace.append({"event": "llm_error", "step": step_index + 1,
                              "error_type": type(exc).__name__})
                usage["retries"] += 1
                # A timeout may occur after the provider generated an unknown
                # amount of output. Reserve the full permitted completion too.
                failed_estimate = prompt_estimate + state["llm_limits"]["max_completion_tokens"]
                usage["estimated_tokens"] += failed_estimate
                usage["budget_tokens"] += failed_estimate
                last_failure = "llm_failure"
                messages.append({"role": "user", "content": _json({"error": "llm_failure", "instruction": "Return a valid action if budget remains."})})
                continue
            usage["cache_hits"] += int(bool(cache_hit))
            tokens, reported = _token_usage(metadata)
            response_size = len(response.encode("utf-8")) if isinstance(response, str) else 0
            if reported:
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    usage[key] += tokens[key]
                usage["budget_tokens"] += tokens["total_tokens"]
            else:
                estimated = prompt_estimate + response_size
                usage["estimated_tokens"] += estimated
                usage["budget_tokens"] += estimated
            trace.append({"event": "llm_response", "step": step_index + 1,
                          "response": response if isinstance(response, str) else None,
                          "tokens": tokens, "tokens_reported": reported,
                          "cache_hit": bool(cache_hit), "budget_tokens": usage["budget_tokens"]})
            if time.monotonic() - started >= budget.max_seconds:
                return self._finish("time_budget", trace, usage, started, tools)
            if usage["budget_tokens"] > budget.max_tokens:
                return self._finish("token_budget", trace, usage, started, tools)
            try:
                command = parse_action(response, budget)
            except ActionError as exc:
                usage["invalid_actions"] += 1
                usage["retries"] += 1
                last_failure = "invalid_action"
                observation = {"ok": False, "error": {"code": "invalid_action", "message": str(exc)}}
                trace.append({"event": "action_error", "step": step_index + 1, "observation": observation})
                # Bounded malformed text may be useful for one repair attempt.
                if isinstance(response, str):
                    messages.append({"role": "assistant", "content": response[:4096]})
                messages.append({"role": "user", "content": _json(observation)})
                continue
            signature = _json(command)
            if signature in seen:
                usage["retries"] += 1
                last_failure = "repeated_action"
                observation = {"ok": False, "error": {"code": "repeated_action", "message": "Choose a different action or stop."}}
            else:
                seen.add(signature)
                action, arguments = command["action"], command["arguments"]
                if action == "plan":
                    observation = {"ok": True, "action": "plan", "goal": arguments["goal"]}
                elif action == "stop":
                    observation = {"ok": True, "action": "stop", "reason": arguments.get("reason", "")}
                else:
                    try:
                        observation = tools.execute(action, arguments)
                    except Exception as exc:
                        observation = {"ok": False, "action": action,
                                       "error": {"code": "tool_exception", "message": type(exc).__name__}}
                    if not observation.get("ok"):
                        usage["tool_failures"] += 1
                        usage["retries"] += 1
                        last_failure = observation.get("error", {}).get("code", "tool_failure")
                    elif action == "expand_entity":
                        visited_entity_ids.add(arguments["entity_id"])
            trace.append({"event": "action", "step": step_index + 1,
                          "command": command, "observation": observation,
                          "tool_usage": dict(tools.usage)})
            messages.extend([{"role": "assistant", "content": signature},
                             {"role": "user", "content": _json({"tool_observation": observation})}])
            if time.monotonic() - started >= budget.max_seconds:
                return self._finish("time_budget", trace, usage, started, tools)
            if observation.get("ok") and command["action"] == "commit_paths":
                if not observation.get("selected_paths") or not observation.get("passage_ids"):
                    return self._finish("invalid_commit", trace, usage, started, tools)
                if len(set(observation["passage_ids"])) > request.evidence_limit or not set(request.protected_passage_ids) <= set(observation["passage_ids"]):
                    return self._finish("evidence_capacity", trace, usage, started, tools)
                return self._finish("committed", trace, usage, started, tools, observation)
            if observation.get("ok") and command["action"] == "stop":
                return self._finish("agent_stop", trace, usage, started, tools)
        reason = last_failure if usage["retries"] > budget.max_retries else "step_budget"
        return self._finish(reason, trace, usage, started, tools)

    def replay(self, request, trace):
        """Re-execute recorded actions without calling a model; detect changed graph
        or observations. This validates structural reproducibility, not truth or
        whether the recorded LLM actually generated those actions.
        """
        started = time.monotonic()
        replay_trace = []
        usage = {"llm_calls": 0, "replayed": True}
        try:
            budget = AgentBudget.from_values(self.budget, request.budget)
            context = self._initial_context(request, budget)
            expected_start = {"event": "start", "context": context,
                              "candidate_entity_ids": request.candidate_entity_ids}
            if not isinstance(trace, list) or not trace or trace[0] != expected_start:
                raise ActionError("Replay context or index manifest differs")
            if len(trace) > budget.max_steps * 3 + 2 or trace[-1].get("event") != "stop":
                raise ActionError("Replay is incomplete or exceeds its bounded event count")
            tools = self._tools(request, budget)
            seen = set()
            pending = None
            pending_invalid = False
            replay_step = 0
            terminal = None
            committed = None
            for position, item in enumerate(trace[1:], start=1):
                event = item.get("event")
                if terminal is not None and event != "stop":
                    raise ActionError("Replay has events after a terminal action")
                if event in ("llm_response", "llm_error"):
                    if pending is not None or pending_invalid:
                        raise ActionError("Replay skipped an action result")
                    replay_step += 1
                    if replay_step > budget.max_steps or item.get("step") != replay_step:
                        raise ActionError("Replay step order or count differs")
                    if event == "llm_response":
                        try:
                            pending = parse_action(item.get("response"), budget)
                        except ActionError:
                            pending_invalid = True
                    continue
                if event == "action_error":
                    if not pending_invalid or item.get("step") != replay_step:
                        raise ActionError("Replay format-error event does not match its response")
                    pending_invalid = False
                    continue
                if event == "stop":
                    if position != len(trace) - 1:
                        raise ActionError("Replay stopped before the end of its trace")
                    expected_paths = committed.get("selected_paths", []) if committed else []
                    expected_passages = committed.get("passage_ids", []) if committed else []
                    if item.get("selected_paths") != expected_paths or item.get("passage_ids") != expected_passages:
                        raise ActionError("Replay final evidence differs from the committed evidence")
                    if terminal is not None and item.get("stop_reason") != terminal:
                        raise ActionError("Replay final stop reason differs")
                    if item.get("stop_reason") == "committed" and committed is None:
                        raise ActionError("Replay claims evidence without a successful commit")
                    return self._finish(item.get("stop_reason", "replay_incomplete"), replay_trace,
                                        usage, started, tools, committed)
                if event != "action":
                    raise ActionError("Unknown replay event")
                command = parse_action(_json(item["command"]), budget)
                if command != pending or item.get("step") != replay_step:
                    raise ActionError("Replay command differs from the recorded model response")
                pending = None
                if pending_invalid:
                    raise ActionError("Replay executed an invalid model response")
                signature = _json(command)
                action, args = command["action"], command["arguments"]
                if signature in seen:
                    observation = {"ok": False, "error": {"code": "repeated_action", "message": "Choose a different action or stop."}}
                elif action == "plan":
                    observation = {"ok": True, "action": "plan", "goal": args["goal"]}
                elif action == "stop":
                    observation = {"ok": True, "action": "stop", "reason": args.get("reason", "")}
                else:
                    observation = tools.execute(action, args)
                seen.add(signature)
                if _json(observation) != _json(item["observation"]) or dict(tools.usage) != item.get("tool_usage"):
                    raise ActionError("Replay tool observation differs")
                replay_trace.append(dict(item))
                if observation.get("ok") and action == "commit_paths":
                    committed = observation
                    terminal = "committed"
                if observation.get("ok") and action == "stop":
                    terminal = "agent_stop"
            raise ActionError("Replay is missing its final stop event")
        except (ActionError, ValueError, TypeError, KeyError, AttributeError) as exc:
            replay_trace.append({"event": "replay_error", "message": str(exc)[:512]})
            return self._finish("replay_mismatch", replay_trace, usage, started)
