"""Single-attempt text-action adapter with an isolated, configuration-aware cache."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
import sqlite3
import time


class AgentLLMAdapter:
    def __init__(self, llm, cache_path):
        self.llm = llm
        self.cache_path = Path(cache_path)

    def __call__(self, messages):
        if not hasattr(self.llm, "openai_client"):
            raise ValueError("Agent MVP requires an OpenAI-compatible text backend")
        limits = json.loads(messages[-1]["content"])["llm_limits"]
        max_tokens = int(limits["max_completion_tokens"])
        timeout = float(limits["timeout_seconds"])
        if max_tokens <= 0 or timeout <= 0:
            raise ValueError("Agent request budget exhausted")
        deadline = time.monotonic() + timeout
        params = dict(self.llm.llm_config.generate_params)
        messages = deepcopy(messages)
        state = json.loads(messages[-1]["content"])
        state["llm_limits"].pop("timeout_seconds", None)
        messages[-1]["content"] = json.dumps(state, sort_keys=True, ensure_ascii=False)
        # Single response, deterministic actions; no hidden SDK/decorator retries.
        params.update(messages=messages, n=1, temperature=0.0)
        params.pop("max_completion_tokens", None)
        params.pop("max_tokens", None)
        token_key = "max_completion_tokens" if "gpt" in params.get("model", "") else "max_tokens"
        params[token_key] = max_tokens
        key_data = {"version": "agent-text-v1", "endpoint": str(self.llm.openai_client.base_url), "params": params}
        key = hashlib.sha256(json.dumps(key_data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.cache_path), timeout=min(1.0, timeout)) as db:
            db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, content TEXT NOT NULL, metadata TEXT NOT NULL)")
            row = db.execute("SELECT content, metadata FROM responses WHERE key = ?", (key,)).fetchone()
        if row:
            return row[0], json.loads(row[1]), True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Agent deadline exhausted during cache lookup")
        client = self.llm.openai_client.with_options(max_retries=0, timeout=remaining)
        response = client.chat.completions.create(**params)
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise ValueError("Agent text response must contain string content")
        usage = response.usage
        metadata = {
            "finish_reason": response.choices[0].finish_reason,
            "model": params.get("model"),
        }
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        if type(prompt_tokens) is int and prompt_tokens > 0 and type(completion_tokens) is int and completion_tokens >= 0:
            metadata.update(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        remaining = deadline - time.monotonic()
        if remaining > 0:
            try:
                with sqlite3.connect(str(self.cache_path), timeout=min(1.0, remaining)) as db:
                    db.execute("INSERT OR REPLACE INTO responses VALUES (?, ?, ?)", (key, content, json.dumps(metadata)))
            except sqlite3.OperationalError:
                pass  # A busy optional cache must not force a second model call.
        return content, metadata, False
