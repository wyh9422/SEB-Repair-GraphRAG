"""Bounded CPU/server smoke checks; --live explicitly enables small API calls.

This validates plumbing, provenance and cached-graph numerics, not benchmark
accuracy. It never loads an embedding model or modifies existing graph caches.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.SRgraphrag.graph.relation_index import RelationIndex
from src.SRgraphrag.graph.schema import entity_id, passage_id
from src.SRgraphrag.retrieval.agent import AgentGraphSearch
from src.SRgraphrag.retrieval.agent_llm import AgentLLMAdapter
from src.SRgraphrag.retrieval.types import GraphSearchRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--openie")
    parser.add_argument("--graph")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", default="deepseek-chat")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = {"scope": "smoke_only_not_benchmark", "live_requested": args.live}
    if args.openie:
        with open(args.openie, encoding="utf-8") as handle:
            records = json.load(handle)
        index = RelationIndex.from_openie(records)
        summary["real_openie"] = asdict(index.manifest)
        rebuilt = RelationIndex.from_openie(records)
        assert rebuilt.manifest == index.manifest
        summary["real_openie"]["repeat_build_identical"] = True
    if args.graph:
        import igraph as ig
        import numpy as np
        graph = ig.Graph.Read_Pickle(args.graph)
        # A numerical fixture using existing nodes, not an embedding of a query.
        reset = np.zeros(graph.vcount())
        reset[0] = 1
        scores = graph.personalized_pagerank(damping=0.5, directed=False, weights="weight", reset=reset, implementation="prpack")
        assert np.isfinite(scores).all() and abs(sum(scores) - 1) < 1e-6
        summary["real_graph"] = {"vertices": graph.vcount(), "edges": graph.ecount(), "finite_ppr": True, "score_sum": float(sum(scores))}
    if args.live:
        from openai import OpenAI
        from src.SRgraphrag.retrieval.judge import judge_answerability_and_bridge
        key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Live checks need a configured API key")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        client = OpenAI(api_key=key, base_url=base_url, max_retries=0, timeout=45)
        llm = SimpleNamespace(openai_client=client, llm_config=SimpleNamespace(generate_params={"model": args.model, "temperature": 0, "n": 1}))
        adapter = AgentLLMAdapter(llm, output / "agent_smoke_cache.sqlite")
        docs = [
            {"passage": "Ada works for Northstar Lab.", "extracted_triples": [["Ada", "works for", "Northstar Lab"]]},
            {"passage": "Northstar Lab is based in Lumen City.", "extracted_triples": [["Northstar Lab", "based in", "Lumen City"]]},
            {"passage": "The River Guild is based in Harbor Town.", "extracted_triples": [["River Guild", "based in", "Harbor Town"]]},
        ]
        index = RelationIndex.from_openie(docs)
        question = "Which city is Ada's employer based in?"
        request = GraphSearchRequest(
            question, question, 2, [], [entity_id("Ada")], {entity_id("Ada"): 1.0},
            [passage_id(docs[0]["passage"])], budget={"max_steps": 8, "max_tokens": 24000, "max_seconds": 90},
        )
        agent = AgentGraphSearch(index, adapter)
        result = agent.search(request)
        with (output / "live_agent_result.json").open("w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, indent=2, ensure_ascii=False, allow_nan=False)
        summary["live_agent"] = {"model": args.model, "stop_reason": result.stop_reason, "usage": result.usage, "selected_path_count": len(result.selected_paths)}
        assert result.selected_paths, "Live Agent did not commit a path; inspect saved result"
        required = {passage_id(row["passage"]) for row in docs[:2]}
        assert required.issubset(result.ranked_passage_ids), "Live fixture lost necessary evidence"
        replayed = agent.replay(request, result.trace)
        assert replayed.ranked_passage_ids == result.ranked_passage_ids
        summary["live_agent"]["replay_identical"] = True
        selected_docs = [index.passages[key].content for key in result.ranked_passage_ids]
        judge = judge_answerability_and_bridge([question], [selected_docs], judge_model_name=args.model, judge_concurrency=1, judge_batch_size=1)[0]
        summary["live_judge"] = {"can_answer": judge.get("can_answer"), "error": judge.get("_error"), "metadata": judge.get("_judge_metadata")}
        assert not judge.get("_error") and judge.get("can_answer"), "Live judge failed the complete-evidence fixture"
        response = client.chat.completions.create(model=args.model, temperature=0, max_tokens=128,
            messages=[{"role":"system", "content":"Answer the question using only the supplied evidence; give a short answer."},
                      {"role":"user", "content":question + "\nEvidence:\n" + "\n".join(selected_docs)}])
        answer = response.choices[0].message.content or ""
        summary["live_reader"] = {"answer": answer, "fixture_answer_correct": "lumen city" in answer.lower()}
        assert summary["live_reader"]["fixture_answer_correct"]
    with (output / "runtime_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
