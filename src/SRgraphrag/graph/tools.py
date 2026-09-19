"""Deterministic, bounded graph tools. The controller never executes model code."""

from __future__ import annotations

import json

from .relation_index import RelationIndex


class ToolError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ToolError("invalid_arguments", f"{name} must be an integer >= {minimum}")
    return value


def _strings(values, name):
    if isinstance(values, (str, bytes)) or values is None:
        raise ValueError(f"{name} must be a collection of IDs")
    result = tuple(dict.fromkeys(values))
    if not all(isinstance(value, str) for value in result):
        raise ValueError(f"{name} must contain string IDs")
    return result


class GraphTools:
    """Per-query capability set with fixed evidence and exploration budgets.

    Only initial seeds/protected passages and subsequently returned IDs are
    accessible. Every execute attempt consumes a call (up to the hard limit),
    including validation failures; returning facts also consumes edge budget.
    """

    ACTION_FIELDS = {
        "expand_entity": ({"entity_id"}, {"entity_id", "direction", "limit", "cursor"}),
        "inspect_passage": ({"passage_id"}, {"passage_id"}),
        "validate_path": ({"steps"}, {"steps"}),
        "commit_paths": ({"paths"}, {"paths", "protected_passage_ids"}),
    }
    MAX_PASSAGE_CHARS = 16000
    MAX_COVER_SEARCH_STATES = 50000

    def __init__(
        self, index: RelationIndex, seed_entity_ids, protected_passage_ids=(),
        evidence_limit=5, max_calls=40, max_expansions=160, max_neighbors=8,
        max_path_length=4, allowed_entity_ids=None,
    ):
        self.index = index
        self.seed_entity_ids = _strings(seed_entity_ids, "seed_entity_ids")
        self.protected_passage_ids = _strings(protected_passage_ids, "protected_passage_ids")
        self.evidence_limit = _integer(evidence_limit, "evidence_limit", 1)
        self.max_calls = _integer(max_calls, "max_calls")
        self.max_expansions = _integer(max_expansions, "max_expansions")
        self.max_neighbors = _integer(max_neighbors, "max_neighbors", 1)
        self.max_path_length = _integer(max_path_length, "max_path_length", 1)
        if set(self.seed_entity_ids) - index.entities.keys():
            raise ValueError("Unknown seed entity ID")
        if set(self.protected_passage_ids) - index.passages.keys():
            raise ValueError("Unknown protected passage ID")
        if len(self.protected_passage_ids) > self.evidence_limit:
            raise ValueError("Protected passages exceed the evidence limit")
        if allowed_entity_ids is None:
            self.allowed_entity_ids = None
        else:
            allowed = set(_strings(allowed_entity_ids, "allowed_entity_ids"))
            if allowed - index.entities.keys():
                raise ValueError("Unknown allowed entity ID")
            # Anchors must remain usable even when the PPR candidate cutoff drops them.
            self.allowed_entity_ids = allowed | set(self.seed_entity_ids)
        self.exposed_entity_ids = set(self.seed_entity_ids)
        self.exposed_fact_ids = set()
        self.exposed_passage_ids = set(self.protected_passage_ids)
        self.tool_calls = 0
        self.expanded_edges = 0
        self._expanded_pages = set()
        self._inspected_passages = set()
        self.committed_paths = []
        self.committed_passage_ids = []

    @property
    def usage(self):
        return {
            "tool_calls": self.tool_calls,
            "expanded_edges": self.expanded_edges,
            "remaining_calls": max(0, self.max_calls - self.tool_calls),
            "remaining_expansions": max(0, self.max_expansions - self.expanded_edges),
        }

    def execute(self, action, arguments) -> dict:
        label = action if isinstance(action, str) else "invalid_action"
        if self.tool_calls >= self.max_calls:
            return self._error(label, ToolError("call_budget_exhausted", "Tool call budget exhausted"))
        self.tool_calls += 1
        try:
            if not isinstance(action, str) or action not in self.ACTION_FIELDS:
                raise ToolError("unknown_action", "Action is not an available graph tool")
            if not isinstance(arguments, dict):
                raise ToolError("invalid_arguments", "Tool arguments must be an object")
            required, allowed = self.ACTION_FIELDS[action]
            if not required <= arguments.keys() or arguments.keys() - allowed:
                raise ToolError("invalid_arguments", "Missing or unsupported tool arguments")
            if action == "expand_entity" and "limit" not in arguments:
                arguments = {**arguments, "limit": min(8, self.max_neighbors)}
            result = getattr(self, "_" + action)(**arguments)
            # Detach return values from the state stored for later commits/replays.
            return json.loads(json.dumps({"ok": True, "action": action, **result, "usage": self.usage}, allow_nan=False))
        except ToolError as exc:
            return self._error(label, exc)

    def _error(self, action, exc):
        return {"ok": False, "action": action,
                "error": {"code": exc.code, "message": str(exc)}, "usage": self.usage}

    def expand_entity(self, entity_id, direction="both", limit=8, cursor=0):
        return self.execute("expand_entity", {"entity_id": entity_id, "direction": direction, "limit": limit, "cursor": cursor})

    def inspect_passage(self, passage_id):
        return self.execute("inspect_passage", {"passage_id": passage_id})

    def validate_path(self, steps):
        return self.execute("validate_path", {"steps": steps})

    def commit_paths(self, paths, protected_passage_ids=None):
        arguments = {"paths": paths}
        if protected_passage_ids is not None:
            arguments["protected_passage_ids"] = protected_passage_ids
        return self.execute("commit_paths", arguments)

    def _visible_id(self, value, exposed, kind):
        if not isinstance(value, str):
            raise ToolError("invalid_arguments", f"{kind} ID must be a string")
        if value not in exposed:
            raise ToolError("unexposed_id", f"{kind} ID was not exposed by the graph tools")

    def _fact_allowed(self, fact):
        return self.allowed_entity_ids is None or (
            fact.subject_id in self.allowed_entity_ids and fact.object_id in self.allowed_entity_ids
        )

    def _expose_facts(self, fact_ids):
        entities = set()
        facts = []
        for fid in fact_ids:
            fact = self.index.facts[fid]
            self.exposed_fact_ids.add(fid)
            self.exposed_passage_ids.update(fact.passage_ids)
            entities.update((fact.subject_id, fact.object_id))
            facts.append(fact.to_dict())
        self.expanded_edges += len(facts)
        self.exposed_entity_ids.update(entities)
        return facts, [self.index.entities[key].to_dict() for key in sorted(entities)]

    def _expand_entity(self, entity_id, direction="both", limit=8, cursor=0):
        self._visible_id(entity_id, self.exposed_entity_ids, "Entity")
        if not isinstance(direction, str) or direction not in {"out", "in", "both"}:
            raise ToolError("invalid_arguments", "direction must be out, in or both")
        _integer(limit, "limit", 1)
        _integer(cursor, "cursor")
        if limit > self.max_neighbors:
            raise ToolError("invalid_arguments", "limit exceeds the per-call neighbor budget")
        page = (entity_id, direction, cursor)
        if page in self._expanded_pages:
            raise ToolError("repeated_expansion", "This entity/direction/cursor was already expanded")
        candidates = set()
        if direction in ("out", "both"):
            candidates.update(self.index.outgoing_fact_ids[entity_id])
        if direction in ("in", "both"):
            candidates.update(self.index.incoming_fact_ids[entity_id])
        candidates = sorted(fid for fid in candidates if self._fact_allowed(self.index.facts[fid]))
        if cursor > len(candidates):
            raise ToolError("invalid_arguments", "cursor is outside the adjacency list")
        remaining = self.max_expansions - self.expanded_edges
        if cursor < len(candidates) and remaining <= 0:
            raise ToolError("expansion_budget_exhausted", "Graph expansion budget exhausted")
        selected = candidates[cursor:cursor + min(limit, remaining)]
        facts, discovered = self._expose_facts(selected)
        self._expanded_pages.add(page)
        following = cursor + len(selected)
        return {
            "entity_id": entity_id, "direction": direction, "cursor": cursor,
            "facts": facts, "discovered_entities": discovered,
            "next_cursor": following if following < len(candidates) else None,
            "total_available": len(candidates),
        }

    def _inspect_passage(self, passage_id):
        self._visible_id(passage_id, self.exposed_passage_ids, "Passage")
        if passage_id in self._inspected_passages:
            raise ToolError("repeated_inspection", "This passage was already inspected")
        passage = self.index.passages[passage_id]
        candidates = [fid for fid in passage.fact_ids if self._fact_allowed(self.index.facts[fid])]
        count = min(self.max_neighbors, self.max_expansions - self.expanded_edges)
        facts, discovered = self._expose_facts(candidates[:count])
        self._inspected_passages.add(passage_id)
        return {
            "passage_id": passage_id, "content": passage.content[:self.MAX_PASSAGE_CHARS],
            "content_truncated": len(passage.content) > self.MAX_PASSAGE_CHARS,
            "facts": facts, "discovered_entities": discovered,
            "omitted_fact_count": len(candidates) - len(facts),
        }

    def _validate_path(self, steps):
        if not isinstance(steps, list) or not steps:
            raise ToolError("invalid_path", "Path steps must be a non-empty array")
        if len(steps) > self.max_path_length:
            raise ToolError("path_length_exceeded", "Path exceeds the maximum length")
        expected_fields = {"fact_id", "from_entity_id", "to_entity_id", "traversal_direction"}
        normalized, visited, previous = [], set(), None
        for step in steps:
            if not isinstance(step, dict) or step.keys() != expected_fields:
                raise ToolError("invalid_path", "Each path step must have exactly the four required fields")
            fid, source, target = step["fact_id"], step["from_entity_id"], step["to_entity_id"]
            self._visible_id(fid, self.exposed_fact_ids, "Fact")
            self._visible_id(source, self.exposed_entity_ids, "Entity")
            self._visible_id(target, self.exposed_entity_ids, "Entity")
            direction = step["traversal_direction"]
            if not isinstance(direction, str) or direction not in {"out", "in", "forward", "reverse"}:
                raise ToolError("invalid_path", "Invalid traversal direction")
            direction = {"forward": "out", "reverse": "in"}.get(direction, direction)
            fact = self.index.facts[fid]
            expected = (fact.subject_id, fact.object_id) if direction == "out" else (fact.object_id, fact.subject_id)
            if (source, target) != expected or not self._fact_allowed(fact):
                raise ToolError("invalid_path", "Path endpoints/direction do not match the indexed fact")
            if previous is not None and previous != source:
                raise ToolError("disconnected_path", "Consecutive path steps do not share an endpoint")
            if not normalized:
                visited.add(source)
            if target in visited:
                raise ToolError("cyclic_path", "A committed path cannot revisit a node")
            if not fact.passage_ids or any(key not in self.index.passages for key in fact.passage_ids):
                raise ToolError("missing_provenance", "Fact has no valid source passage")
            visited.add(target)
            previous = target
            normalized.append({"fact_id": fid, "from_entity_id": source, "to_entity_id": target,
                               "traversal_direction": direction})
        return {"path": normalized, "fact_ids": [step["fact_id"] for step in normalized], "valid": True}

    def _minimum_cover(self, fact_ids):
        """Exact cardinality-minimum cover within the fixed evidence capacity.

        Branch on an uncovered fact, with deterministic candidate order. A
        separate computation limit fails explicitly rather than claiming a
        heuristic solution is minimal. One selected source per fact suffices.
        """
        protected = set(self.protected_passage_ids)
        missing = [fid for fid in sorted(fact_ids) if not protected.intersection(self.index.facts[fid].passage_ids)]
        if not missing:
            return list(self.protected_passage_ids)
        slots = self.evidence_limit - len(protected)
        if slots <= 0:
            raise ToolError("evidence_capacity_exceeded", "Protected passages leave no room for the path")
        masks = {}
        for position, fid in enumerate(missing):
            for pid in self.index.facts[fid].passage_ids:
                if pid in self.exposed_passage_ids and pid not in protected:
                    masks[pid] = masks.get(pid, 0) | (1 << position)
        # Equal coverage: the lexicographically first passage is sufficient.
        unique = {}
        for pid in sorted(masks):
            unique.setdefault(masks[pid], pid)
        candidates = sorted((pid, mask) for mask, pid in unique.items())
        full = (1 << len(missing)) - 1
        states = 0

        def count(mask):
            return bin(mask).count("1")

        def search(remaining, capacity, failed):
            nonlocal states
            states += 1
            if states > self.MAX_COVER_SEARCH_STATES:
                raise ToolError("cover_search_budget_exhausted", "Source cover search budget exhausted")
            if not remaining:
                return []
            if capacity == 0 or (remaining, capacity) in failed:
                return None
            relevant = [(pid, mask & remaining) for pid, mask in candidates if mask & remaining]
            if not relevant:
                return None
            maximum = max(count(mask) for _, mask in relevant)
            if (count(remaining) + maximum - 1) // maximum > capacity:
                return None
            options = None
            bits = remaining
            while bits:
                bit = bits & -bits
                covers = [(pid, mask) for pid, mask in relevant if mask & bit]
                if not covers:
                    return None
                if options is None or len(covers) < len(options):
                    options = covers
                bits ^= bit
            for pid, mask in sorted(options, key=lambda pair: (-count(pair[1]), pair[0])):
                result = search(remaining & ~mask, capacity - 1, failed)
                if result is not None:
                    return [pid] + result
            failed.add((remaining, capacity))
            return None

        for capacity in range(1, slots + 1):
            result = search(full, capacity, set())
            if result is not None:
                return list(self.protected_passage_ids) + sorted(result)
        raise ToolError("evidence_capacity_exceeded", "No source cover fits the protected Top-k evidence budget")

    def _commit_paths(self, paths, protected_passage_ids=None):
        if protected_passage_ids is not None:
            if not isinstance(protected_passage_ids, list) or not all(isinstance(pid, str) for pid in protected_passage_ids):
                raise ToolError("invalid_arguments", "protected_passage_ids must be an array of strings")
            if set(protected_passage_ids) != set(self.protected_passage_ids):
                raise ToolError("protected_evidence_override", "Protected evidence is fixed for this query")
        if not isinstance(paths, list) or not paths:
            raise ToolError("invalid_path", "paths must be a non-empty array")
        if len(paths) > max(1, self.max_expansions):
            raise ToolError("invalid_path", "Too many paths for the exploration budget")
        validated, seen, fact_ids = [], set(), set()
        for path in paths:
            if isinstance(path, dict):
                if set(path) != {"steps"}:
                    raise ToolError("invalid_path", "Path object must only contain steps")
                path = path["steps"]
            result = self._validate_path(path)
            signature = json.dumps(result["path"], sort_keys=True)
            if signature not in seen:
                validated.append(result["path"])
                seen.add(signature)
                fact_ids.update(result["fact_ids"])
        passages = self._minimum_cover(fact_ids)
        witnesses = {fid: next(pid for pid in passages if pid in self.index.facts[fid].passage_ids)
                     for fid in sorted(fact_ids)}
        self.committed_paths = validated
        self.committed_passage_ids = passages
        return {"selected_paths": validated, "passage_ids": passages,
                "fact_passage_ids": witnesses, "protected_passage_ids": list(self.protected_passage_ids)}
