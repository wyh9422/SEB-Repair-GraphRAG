"""Closed, versioned JSON action schema for graph exploration (no code execution)."""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
import math


ACTION_SCHEMA_VERSION = "graph-actions-v1"

ACTION_ENVELOPE_HELP = (
    'Return one JSON object with exactly "action" and "arguments". '
    'Put every action parameter inside the "arguments" object, not at the top level. '
    'Example: {"action":"expand_entity","arguments":{"entity_id":"OBSERVED_ENTITY_ID"}}. '
    'Replace the placeholder with an entity ID already observed in this task.'
)


class ActionError(ValueError):
    """A model action or budget does not match the public protocol."""


@dataclass(frozen=True)
class AgentBudget:
    max_steps: int = 8
    max_tool_calls: int = 16
    max_expansions: int = 64
    max_path_length: int = 4
    max_neighbors: int = 8
    # None disables only the cumulative token cutoff. Steps/tools/time and the
    # per-response output cap remain bounded; token usage is still recorded.
    max_tokens: int | None = None
    max_seconds: float = 60.0
    max_retries: int = 2

    @classmethod
    def from_values(cls, *overrides):
        values = {field.name: field.default for field in fields(cls)}
        for override in overrides:
            if override is None:
                continue
            if not isinstance(override, dict) or set(override) - set(values):
                raise ActionError("Unknown budget fields or non-object budget")
            values.update(override)
        for name, value in values.items():
            if name == "max_tokens" and value is None:
                continue
            if name == "max_seconds":
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ActionError("max_seconds must be finite and non-negative")
            elif isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ActionError(name + " must be a non-negative integer")
        if values["max_path_length"] < 1 or values["max_neighbors"] < 1:
            raise ActionError("max_path_length and max_neighbors must be positive")
        return cls(**values)

    def to_dict(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ActionError("Duplicate JSON object key: " + key)
        result[key] = value
    return result


def _keys(value, required, optional=()):
    # Name only schema-owned fields, never echo arbitrary model-supplied keys or
    # values into the correction instruction. Validation remains fail-closed.
    expected = "Required fields: " + ", ".join(required or ("none",))
    expected += "; optional fields: " + ", ".join(optional or ("none",)) + "."
    if not isinstance(value, dict):
        raise ActionError("Action arguments (or path step) must be a JSON object. " + expected)
    missing = set(required) - set(value)
    if missing:
        raise ActionError("Missing required fields: " + ", ".join(sorted(missing)) + ". " + expected)
    if set(value) - set(required) - set(optional):
        raise ActionError("Unexpected fields are not allowed. " + expected)


def _text(value, name, maximum=256, allow_empty=False):
    if not isinstance(value, str) or len(value) > maximum or (not allow_empty and not value.strip()):
        raise ActionError(name + " must be a bounded non-empty string")


def _integer(value, name, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or (maximum is not None and value > maximum):
        raise ActionError(name + " is outside the integer budget")


def _path(steps, budget):
    if not isinstance(steps, list) or not 1 <= len(steps) <= budget.max_path_length:
        raise ActionError("Path length is outside the path budget")
    for step in steps:
        _keys(step, ("fact_id", "from_entity_id", "to_entity_id", "traversal_direction"))
        for name in ("fact_id", "from_entity_id", "to_entity_id"):
            _text(step[name], name)
        if step["traversal_direction"] not in ("out", "in", "forward", "reverse"):
            raise ActionError("Invalid traversal_direction")


def parse_action(response, budget):
    """Parse exactly one JSON object; reject prose, fences, duplicate keys and extras."""
    if not isinstance(response, str) or len(response) > 65536:
        raise ActionError("Response must be bounded JSON text")
    try:
        command = json.loads(response, object_pairs_hook=_unique_object,
                             parse_constant=lambda value: (_ for _ in ()).throw(ActionError("Non-finite JSON number")))
    except (ValueError, TypeError, RecursionError) as exc:
        raise ActionError("Response is not a valid JSON action. " + ACTION_ENVELOPE_HELP) from exc
    if not isinstance(command, dict) or set(command) != {"action", "arguments"}:
        raise ActionError("Invalid action envelope. " + ACTION_ENVELOPE_HELP)
    action, args = command["action"], command["arguments"]
    _text(action, "action", 64)
    if action == "plan":
        _keys(args, ("goal",), ("rationale",))
        _text(args["goal"], "goal", 512)
        if "rationale" in args:
            _text(args["rationale"], "rationale", 512)
    elif action == "expand_entity":
        _keys(args, ("entity_id",), ("direction", "limit", "cursor"))
        _text(args["entity_id"], "entity_id")
        if args.get("direction", "both") not in ("out", "in", "both"):
            raise ActionError("Invalid expansion direction")
        _integer(args.get("limit", budget.max_neighbors), "limit", 1, budget.max_neighbors)
        _integer(args.get("cursor", 0), "cursor")
        # Canonical defaults make equivalent repeated actions detectable.
        args = dict(args, direction=args.get("direction", "both"),
                    limit=args.get("limit", budget.max_neighbors), cursor=args.get("cursor", 0))
    elif action == "inspect_passage":
        _keys(args, ("passage_id",))
        _text(args["passage_id"], "passage_id")
    elif action == "validate_path":
        _keys(args, ("steps",))
        _path(args["steps"], budget)
    elif action == "commit_paths":
        _keys(args, ("paths",))
        if not isinstance(args["paths"], list) or not 1 <= len(args["paths"]) <= max(1, budget.max_expansions):
            raise ActionError("At least one bounded path is required")
        for steps in args["paths"]:
            _path(steps, budget)
    elif action == "stop":
        _keys(args, (), ("reason",))
        if "reason" in args:
            _text(args["reason"], "reason", 256)
    else:
        raise ActionError("Unknown action")
    return {"action": action, "arguments": args}
