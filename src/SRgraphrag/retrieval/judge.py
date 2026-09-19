"""Batched answerability and bridge service, preserving the legacy HTTP protocol."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


DEFAULT_PROMPT_JSON = "src/SRgraphrag/prompts/dspy_prompts/judge_prompt.json"


def resolve_judge_metadata(
    judge_model_name: str = "deepseek-reasoner",
    judge_concurrency: int = 100,
    judge_batch_size: int = 100,
    prompt_json: str = DEFAULT_PROMPT_JSON,
) -> dict:
    """Return the effective non-secret configuration used for judge requests."""
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    api_root = base_url if base_url.endswith("/v1") else base_url + "/v1"
    return {
        "judge_model": judge_model_name,
        "judge_base_url": base_url,
        "judge_endpoint": api_root + "/chat/completions",
        "judge_concurrency": max(1, judge_concurrency),
        "judge_batch_size": max(1, judge_batch_size),
        "judge_prompt_sha256": hashlib.sha256(Path(prompt_json).read_bytes()).hexdigest(),
        "judge_temperature": 0.0,
    }


def _failure(error: str, raw="", analysis: str = "") -> dict:
    return {
        "can_answer": False,
        "evidence_ids": [],
        "evidence_docs": [],
        "analysis_zh": analysis or f"Call failed: {error}",
        "bridge_possible": False,
        "bridge_question": "",
        "bridge_evidence_ids": [],
        "bridge_analysis_zh": "",
        "_error": error,
        "_raw": raw,
    }


def judge_answerability_and_bridge(
    queries: list[str],
    top5_docs_list: list[list[str]],
    *,
    judge_model_name: str = "deepseek-reasoner",
    judge_concurrency: int = 100,
    judge_batch_size: int = 100,
    prompt_json: str = DEFAULT_PROMPT_JSON,
) -> list[dict]:
    """
    Batched concurrent judging with DeepSeek.

    Input:
    - queries: N queries
    - top5_docs_list: N lists, each is top-5 docs (strings)

    Output:
    - list[dict] length N, each:
        {
        "can_answer": bool,
        "evidence_ids": ["D1"...],
        "evidence_docs": [doc_str...],
        "analysis_zh": str,
        "bridge_possible": bool,
        "bridge_question": str,
        "bridge_evidence_ids": ["D1"...],
        "bridge_analysis_zh": str,
        optional "_error","_raw"
        }
    """
    if len(queries) != len(top5_docs_list):
        raise ValueError("len(queries) must equal len(top5_docs_list)")
    if not queries:
        return []

    import json, asyncio, random

    metadata = resolve_judge_metadata(
        judge_model_name, judge_concurrency, judge_batch_size, prompt_json,
    )
    MODEL = metadata["judge_model"]
    API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is empty")

    import aiohttp

    # -------------------------
    # Load prompts from JSON
    # Expected keys:
    # - system_prompt: str
    # - user_prefix: str
    # - doc_max_chars: int (optional)
    # - query_max_chars: int (optional)
    # -------------------------
    with open(prompt_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("prompt_json must be a JSON object")

    if "system_prompt" not in cfg or "user_prefix" not in cfg:
        raise KeyError("prompt_json must contain keys: system_prompt, user_prefix")

    SYSTEM_PROMPT = str(cfg["system_prompt"])
    USER_PREFIX = str(cfg["user_prefix"])

    DOC_MAX_CHARS = int(cfg.get("doc_max_chars", 200000))
    QUERY_MAX_CHARS = int(cfg.get("query_max_chars", 2000))

    def _compact_text(s, max_chars: int) -> str:
        if s is None:
            return ""
        s = str(s).replace("\x00", " ")
        if len(s) > max_chars:
            return s[:max_chars] + " ...[TRUNCATED]"
        return s

    def _prep_docs(top5):
        docs = list(top5 or [])[:5]
        while len(docs) < 5:
            docs.append("")
        return [_compact_text(d, DOC_MAX_CHARS) for d in docs]

    def _normalize_id(x) -> str:
        x = str(x).strip().upper()
        if x.startswith("D") and len(x) >= 2 and x[1].isdigit():
            return "D" + x[1]
        return x

    def _ids_to_docs(ids, top5_docs):
        out = []
        top5_docs = list(top5_docs or [])
        for i in ids or []:
            i = _normalize_id(i)
            if i in ("D1", "D2", "D3", "D4", "D5"):
                idx = int(i[1]) - 1
                if 0 <= idx < len(top5_docs):
                    out.append(top5_docs[idx])
        return out

    async def _deepseek_chat(session: aiohttp.ClientSession, payload: dict, max_retries: int = 6) -> dict:
        url = metadata["judge_endpoint"]
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        last_err = None
        for attempt in range(max_retries):
            try:
                async with session.post(url, headers=headers, json=payload) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        if resp.status in (408, 429, 500, 502, 503, 504):
                            last_err = f"HTTP {resp.status}"
                            await asyncio.sleep(min(30, 2 ** attempt) + random.random())
                            continue
                        return {"_error": f"HTTP {resp.status}", "_raw": text}
                    try:
                        return json.loads(text)
                    except Exception:
                        return {"_error": "JSON_DECODE_FAIL", "_raw": text}
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                last_err = f"{type(e).__name__}: {e}"
                await asyncio.sleep(min(30, 2 ** attempt) + random.random())
        return {"_error": "retry_exhausted", "_raw": str(last_err)}

    def _extract_json(raw: dict) -> dict:
        if not isinstance(raw, dict):
            return {
                "can_answer": False,
                "evidence_ids": [],
                "analysis_zh": "Call failed: raw is not a dictionary",
                "bridge_possible": False,
                "bridge_question": "",
                "bridge_evidence_ids": [],
                "bridge_analysis_zh": "",
                "_error": "raw_not_dict",
                "_raw": str(raw),
            }

        if "_error" in raw:
            return {
                "can_answer": False,
                "evidence_ids": [],
                "analysis_zh": f"Call failed: {raw.get('_error','unknown')}",
                "bridge_possible": False,
                "bridge_question": "",
                "bridge_evidence_ids": [],
                "bridge_analysis_zh": "",
                "_error": raw.get("_error", ""),
                "_raw": raw.get("_raw", ""),
            }

        try:
            content = raw["choices"][0]["message"]["content"].strip()
        except Exception:
            return {
                "can_answer": False,
                "evidence_ids": [],
                "analysis_zh": "Returns a structural exception",
                "bridge_possible": False,
                "bridge_question": "",
                "bridge_evidence_ids": [],
                "bridge_analysis_zh": "",
                "_error": "bad_response_shape",
                "_raw": raw,
            }

        try:
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                return _failure("parsed_not_dict", content)
            return parsed
        except Exception:
            l = content.find("{")
            r = content.rfind("}")
            if l != -1 and r != -1 and r > l:
                try:
                    parsed = json.loads(content[l:r + 1])
                    if not isinstance(parsed, dict):
                        return _failure("parsed_not_dict", content)
                    return parsed
                except Exception:
                    pass
            return {
                "can_answer": False,
                "evidence_ids": [],
                "analysis_zh": "The model output cannot be parsed as JSON.",
                "bridge_possible": False,
                "bridge_question": "",
                "bridge_evidence_ids": [],
                "bridge_analysis_zh": "",
                "_error": "non_json_output",
                "_raw": content,
            }

    def _build_user_prompt(query: str, top5_docs: list[str]) -> str:
        q = _compact_text(query, QUERY_MAX_CHARS)
        d1, d2, d3, d4, d5 = _prep_docs(top5_docs)

        body = (
            "query: " + q + "\n\n"
            "Retrieved top-5 docs (ID -> FULL content):\n"
            "D1: " + d1 + "\n\n"
            "D2: " + d2 + "\n\n"
            "D3: " + d3 + "\n\n"
            "D4: " + d4 + "\n\n"
            "D5: " + d5
        )
        return USER_PREFIX + body

    async def _judge_one(
        sem: asyncio.Semaphore,
        session: aiohttp.ClientSession,
        query: str,
        top5_docs: list[str],
    ) -> dict:
        user_prompt = _build_user_prompt(query, top5_docs)

        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": metadata["judge_temperature"],
        }

        async with sem:
            raw = await _deepseek_chat(session, payload)

        parsed = _extract_json(raw)

        can_answer = bool(parsed.get("can_answer", False))

        ev_ids = parsed.get("evidence_ids", [])
        if not isinstance(ev_ids, list):
            ev_ids = []
        ev_ids = [_normalize_id(x) for x in ev_ids]
        ev_ids = [x for x in ev_ids if x in ("D1", "D2", "D3", "D4", "D5")][:5]
        ev_docs = _ids_to_docs(ev_ids, top5_docs)

        bridge_possible = bool(parsed.get("bridge_possible", False))
        bridge_question = str(parsed.get("bridge_question", "") or "").strip()
        bridge_ids = parsed.get("bridge_evidence_ids", [])
        if not isinstance(bridge_ids, list):
            bridge_ids = []
        bridge_ids = [_normalize_id(x) for x in bridge_ids]
        bridge_ids = [x for x in bridge_ids if x in ("D1", "D2", "D3", "D4", "D5")][:5]

        if can_answer:
            bridge_possible = False
            bridge_question = ""
            bridge_ids = []
            bridge_analysis_zh = ""
        else:
            if not bridge_possible:
                bridge_question = ""
                bridge_ids = []
                bridge_analysis_zh = ""
            else:
                bridge_analysis_zh = str(parsed.get("bridge_analysis_zh", "") or "")

        out = {
            "can_answer": can_answer,
            "evidence_ids": ev_ids,
            "evidence_docs": ev_docs,
            "analysis_zh": str(parsed.get("analysis_zh", "") or ""),
            "bridge_possible": bridge_possible,
            "bridge_question": bridge_question,
            "bridge_evidence_ids": bridge_ids,
            "bridge_analysis_zh": bridge_analysis_zh,
            "_judge_metadata": dict(metadata),
        }
        if isinstance(parsed, dict) and "_error" in parsed:
            out["_error"] = parsed.get("_error")
        if isinstance(parsed, dict) and "_raw" in parsed:
            out["_raw"] = parsed.get("_raw")
        return out

    async def _judge_batch(all_queries: list[str], all_top5: list[list[str]]) -> list[dict]:
        from tqdm import tqdm

        timeout = aiohttp.ClientTimeout(total=360, connect=30, sock_connect=60, sock_read=300)
        connector = aiohttp.TCPConnector(
            limit=max(1, judge_concurrency) * 2,
            limit_per_host=max(1, judge_concurrency),
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
        )
        sem = asyncio.Semaphore(max(1, judge_concurrency))

        results: list[dict] = [None] * len(all_queries)  # type: ignore
        total_n = len(all_queries)
        pbar = tqdm(total=total_n, desc=f"Judge[{MODEL}] (conc={judge_concurrency}, batch={judge_batch_size})")

        async def _judge_one_with_idx(i: int):
            try:
                r = await _judge_one(sem, session, all_queries[i], all_top5[i])
            except Exception as e:
                r = _failure("judge_exception", str(e), f"judge异常: {type(e).__name__}")
                r["_judge_metadata"] = dict(metadata)
            return i, r

        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            for start in range(0, total_n, max(1, judge_batch_size)):
                end = min(total_n, start + max(1, judge_batch_size))
                tasks = [asyncio.create_task(_judge_one_with_idx(i)) for i in range(start, end)]

                for fut in asyncio.as_completed(tasks):
                    i, r = await fut
                    results[i] = r
                    pbar.update(1)

        pbar.close()

        # Ensure no None leaks out (prevents AttributeError upstream)
        for i in range(len(results)):
            if results[i] is None:
                results[i] = {
                    "can_answer": False,
                    "evidence_ids": [],
                    "evidence_docs": [],
                    "analysis_zh": "judge异常: result is None",
                    "bridge_possible": False,
                    "bridge_question": "",
                    "bridge_evidence_ids": [],
                    "bridge_analysis_zh": "",
                    "_error": "none_result",
                    "_raw": "",
                }

        return results  # type: ignore

    return asyncio.run(_judge_batch(queries, top5_docs_list))
