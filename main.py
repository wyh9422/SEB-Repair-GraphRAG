import os
import json
import argparse
import logging
import math
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union, Optional

if TYPE_CHECKING:
    from src.SRgraphrag.SRgraphrag import SRgraphrag

# os.environ["LOG_LEVEL"] = "DEBUG"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"
# os.environ["HF_HUB_DISABLE_XET"] = "1"
# os.environ["DEEPSEEK_API_KEY"] = "YOUR_KEY"

# -------------------------
# JSONL loader
# -------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def load_round1_replay(path: str) -> List[Dict[str, Any]]:
    """Read frozen retrieval/judge inputs, excluding evaluation and gold fields."""
    round_fields = {
        "query", "final_top5", "final_docs", "final_scores", "dpr_top5", "dpr1",
        "graph_top5", "graph_runnable", "final_source", "repair_applied",
        "graph_search_mode", "path_protected_docs", "subject_cap", "filter_instruction",
        "raw_top30_triples", "filtered_triples", "unique_subject_count_before",
        "unique_subject_count_after", "evidence_docs", "evidence_injected",
        "evidence_missing", "evidence_missing_cnt", "evidence_replaced_cnt",
        "guard_triggered@5", "guard_replaced_idx@5",
    }
    judge_fields = {
        "can_answer", "evidence_ids", "evidence_docs", "analysis_zh",
        "bridge_possible", "bridge_question", "bridge_evidence_ids",
        "bridge_analysis_zh", "_error",
    }
    metadata_fields = {
        "judge_model", "judge_base_url", "judge_endpoint", "judge_concurrency",
        "judge_batch_size", "judge_prompt_sha256", "judge_temperature",
    }
    rows = []
    with open(path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid replay JSON at line {line_number}") from error
            if not isinstance(row, dict) or not isinstance(row.get("query"), str):
                raise ValueError(f"Replay line {line_number} needs a query string")
            if not isinstance(row.get("round1"), dict) or not isinstance(row.get("judge1"), dict):
                raise ValueError(f"Replay line {line_number} needs round1 and judge1 objects")
            r1 = {key: value for key, value in row["round1"].items() if key in round_fields}
            for key in ("final_top5", "final_docs", "final_scores"):
                if key not in r1 or (r1[key] is not None and not isinstance(r1[key], list)):
                    raise ValueError(f"Replay line {line_number} needs a valid round1.{key}")
            if not isinstance(r1["final_top5"], list) or not isinstance(r1["final_docs"], list):
                raise ValueError(f"Replay line {line_number} needs document lists")
            judge = {key: value for key, value in row["judge1"].items() if key in judge_fields}
            metadata = row["judge1"].get("_judge_metadata")
            if isinstance(metadata, dict):
                judge["_judge_metadata"] = {key: value for key, value in metadata.items() if key in metadata_fields}
            rows.append({"query": row["query"], "round1": r1, "judge1": judge})
    return rows


def get_gold_docs(samples: List, dataset_name: str = None) -> List:
    gold_docs = []
    for sample in samples:
        if 'supporting_facts' in sample:  # hotpotqa, 2wikimultihopqa
            gold_title = set([item[0] for item in sample['supporting_facts']])
            gold_title_and_content_list = [item for item in sample['context'] if item[0] in gold_title]
            if dataset_name.startswith('hotpotqa'):
                gold_doc = [item[0] + '\n' + ''.join(item[1]) for item in gold_title_and_content_list]
            else:
                gold_doc = [item[0] + '\n' + ' '.join(item[1]) for item in gold_title_and_content_list]
        elif 'contexts' in sample:
            gold_doc = [item['title'] + '\n' + item['text'] for item in sample['contexts'] if item['is_supporting']]
        else:
            assert 'paragraphs' in sample, "`paragraphs` should be in sample, or consider the setting not to evaluate retrieval"
            gold_paragraphs = []
            for item in sample['paragraphs']:
                if 'is_supporting' in item and item['is_supporting'] is False:
                    continue
                gold_paragraphs.append(item)
            gold_doc = [item['title'] + '\n' + (item['text'] if 'text' in item else item['paragraph_text']) for item in gold_paragraphs]

        gold_doc = list(set(gold_doc))
        gold_docs.append(gold_doc)
    return gold_docs


def get_gold_answers(samples):
    gold_answers = []
    for sample_idx in range(len(samples)):
        gold_ans = None
        sample = samples[sample_idx]

        if 'answer' in sample or 'gold_ans' in sample:
            gold_ans = sample['answer'] if 'answer' in sample else sample['gold_ans']
        elif 'reference' in sample:
            gold_ans = sample['reference']
        elif 'obj' in sample:
            gold_ans = set(
                [sample['obj']] + [sample['possible_answers']] + [sample['o_wiki_title']] + [sample['o_aliases']])
            gold_ans = list(gold_ans)
        assert gold_ans is not None
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        assert isinstance(gold_ans, list)
        gold_ans = set(gold_ans)
        if 'answer_aliases' in sample:
            gold_ans.update(sample['answer_aliases'])

        gold_answers.append(gold_ans)

    return gold_answers


def save_rag_qa_results_to_json(
    rag_qa_return: Union[
        Tuple[List[Any], List[str], List[Dict]],
        Tuple[List[Any], List[str], List[Dict], Dict, Dict]
    ],
    save_path: str,
    extra_info: Dict[str, Any] = None,
    ensure_ascii: bool = False
) -> str:
    def _safe_serialize(obj: Any) -> Any:
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, (list, tuple)):
            return [_safe_serialize(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): _safe_serialize(v) for k, v in obj.items()}
        try:
            import numpy as np
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
        except Exception:
            pass
        return str(obj)

    if not isinstance(rag_qa_return, tuple):
        raise TypeError(f"rag_qa_return must be a tuple, got {type(rag_qa_return)}")

    if len(rag_qa_return) == 3:
        queries_solutions, all_response_message, all_metadata = rag_qa_return
        overall_retrieval_result = None
        overall_qa_results = None
    elif len(rag_qa_return) == 5:
        queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results = rag_qa_return
    else:
        raise ValueError(f"Unexpected rag_qa return length: {len(rag_qa_return)} (expected 3 or 5)")

    per_query: List[Dict[str, Any]] = []
    for i, q in enumerate(queries_solutions):
        record = {
            "idx": i,
            "question": _safe_serialize(getattr(q, "question", None)),
            "docs": _safe_serialize(getattr(q, "docs", None)),
            "answer": _safe_serialize(getattr(q, "answer", None)),
            "gold_docs": _safe_serialize(getattr(q, "gold_docs", None)),
            "gold_answers": _safe_serialize(getattr(q, "gold_answers", None)),
        }
        for opt_field in ["doc_scores", "metadata", "rerank_log", "facts", "thoughts", "sources"]:
            if hasattr(q, opt_field):
                record[opt_field] = _safe_serialize(getattr(q, opt_field))

        if isinstance(all_response_message, list) and i < len(all_response_message):
            record["response_message"] = _safe_serialize(all_response_message[i])
        if isinstance(all_metadata, list) and i < len(all_metadata):
            record["qa_metadata"] = _safe_serialize(all_metadata[i])

        per_query.append(record)

    payload: Dict[str, Any] = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "num_queries": len(per_query),
        "overall_retrieval_result": _safe_serialize(overall_retrieval_result),
        "overall_qa_results": _safe_serialize(overall_qa_results),
        "per_query": per_query,
    }

    if extra_info:
        payload["extra_info"] = _safe_serialize(extra_info)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=ensure_ascii, indent=2)

    return save_path


def run_pipeline(
    srgraphrag: "SRgraphrag",
    *,
    mode: str,  # "retrieve" | "qa"
    dataset_name: str,
    queries: List[str],
    gold_docs: Optional[List[List[str]]],
    gold_answers: Optional[List[Any]],
    subject_cap: int,
    result_save_root: Optional[str],
    save_dir: str,
    llm_name: str,
    num_to_retrieve: Optional[int] = 200,
    judge_model_name: str = "deepseek-reasoner",
    eval_k_list: Tuple[int, ...] = (1, 2, 5, 30),
    judge_concurrency: int = 100,
    judge_batch_size: int = 100,
    graph_search_mode: str = "ppr",
    agent_apply_to: str = "round2",
    agent_budget: Optional[Dict[str, Any]] = None,
    agent_fallback: str = "ppr",
    round1_replay_path: Optional[str] = None,
) -> None:
    """
    mode="retrieve": retrieve only.
    mode="qa": QA after retrieve.
    """
    assert mode in ("retrieve", "qa"), f"Unknown mode: {mode}"

    round1_replay = None
    if round1_replay_path is not None:
        replay_rows = load_round1_replay(round1_replay_path)
        if len(replay_rows) < len(queries):
            raise ValueError("Round-1 replay contains fewer rows than the query batch")
        round1_replay = replay_rows[:len(queries)]
    retrieval_options = {
        "queries": queries,
        "gold_docs": gold_docs,
        "num_to_retrieve": num_to_retrieve,
        "eval_k_list": eval_k_list,
        "result_save_root": result_save_root,
        "dataset_name": dataset_name,
        "subject_cap": subject_cap,
        "judge_model_name": judge_model_name,
        "judge_concurrency": judge_concurrency,
        "judge_batch_size": judge_batch_size,
        "keep_miss_distribution": True,
        "graph_search_mode": graph_search_mode,
        "agent_apply_to": agent_apply_to,
        "agent_budget": agent_budget,
        "agent_fallback": agent_fallback,
        "round1_replay": round1_replay,
    }
    if mode == "retrieve":
        srgraphrag.retrieve(**retrieval_options)

    if mode == "qa":
        rag_ret = srgraphrag.rag_qa(
            gold_answers=gold_answers,
            **retrieval_options,
        )

        save_rag_qa_results_to_json(
            rag_qa_return=rag_ret,
            save_path=os.path.join(save_dir, f"rag_qa_{dataset_name}.json"),
            extra_info={
                "dataset": dataset_name,
                "llm": llm_name,
                "mode": mode,
                "subject_cap": subject_cap,
                "judge_model_name": judge_model_name,
                "judge_concurrency": judge_concurrency,
                "judge_batch_size": judge_batch_size,
                "num_to_retrieve": num_to_retrieve,
                "eval_k_list": eval_k_list,
                "graph_search_mode": graph_search_mode,
                "agent_apply_to": agent_apply_to,
                "agent_budget": agent_budget,
                "agent_fallback": agent_fallback,
                "round1_replay_path": round1_replay_path,
            },
            ensure_ascii=False,
        )


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SEB_Repair_GraphRAG retrieval and QA")
    parser.add_argument('--dataset', type=str, default='musique', help='Dataset name')
    parser.add_argument('--llm_base_url', type=str, default='https://api.openai.com/v1', help='LLM base URL')
    parser.add_argument('--llm_name', type=str, default='deepseek-chat', help='LLM name')
    parser.add_argument('--judge_llm_name', type=str, default='deepseek-reasoner', help='judge_LLM name')
    parser.add_argument('--judge_concurrency', type=_positive_int, default=100)
    parser.add_argument('--judge_batch_size', type=_positive_int, default=100)
    parser.add_argument('--embedding_name', type=str, default='nvidia/NV-Embed-v2', help='embedding model name')

    parser.add_argument('--force_index_from_scratch', type=str, default='false')
    parser.add_argument('--force_openie_from_scratch', type=str, default='false')
    parser.add_argument('--openie_mode', choices=['online', 'offline'], default='online')

    parser.add_argument('--save_dir', type=str, default='outputs')
    parser.add_argument("--result_save_root", type=str, default=None)
    parser.add_argument("--subject_level_Entity_Cap", type=str, default="4")

    parser.add_argument("--mode", choices=["retrieve", "qa"], default="qa",
                        help="retrieve=only retrieval; qa=retrieval + QA")

    parser.add_argument("--test_n", type=_nonnegative_int, default=1000)
    parser.add_argument("--num_to_retrieve", type=_positive_int, default=None,
                        help="Candidate count; defaults to the configured retrieval_top_k")
    parser.add_argument("--graph_search_mode", choices=("ppr", "agent", "hybrid"), default="ppr")
    parser.add_argument("--agent_apply_to", choices=("round2",), default="round2")
    parser.add_argument("--agent_fallback", choices=("ppr", "dpr", "none"), default="ppr")
    parser.add_argument("--round1_replay_path", default=None,
                        help="Frozen retrieval_trace.jsonl; reuse first-round evidence and judge outputs")
    parser.add_argument("--agent_max_steps", type=_positive_int, default=8)
    parser.add_argument("--agent_max_tool_calls", type=_positive_int, default=16)
    parser.add_argument("--agent_max_expansions", type=_positive_int, default=64)
    parser.add_argument("--agent_max_path_length", type=_positive_int, default=4)
    parser.add_argument("--agent_max_neighbors", type=_positive_int, default=8)
    parser.add_argument("--agent_max_tokens", type=_positive_int, default=12000)
    parser.add_argument("--agent_max_seconds", type=_positive_float, default=60.0)
    return parser


def agent_budget_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        name: getattr(args, "agent_" + name)
        for name in (
            "max_steps", "max_tool_calls", "max_expansions", "max_path_length",
            "max_neighbors", "max_tokens", "max_seconds",
        )
    }


def main():
    args = build_parser().parse_args()
    if args.test_n == 0:
        logging.info("No queries requested; skipping indexing and inference.")
        return

    # Argument parsing and --help do not need model backends or API clients.
    from src.SRgraphrag.SRgraphrag import SRgraphrag
    from src.SRgraphrag.utils.misc_utils import string_to_bool
    from src.SRgraphrag.utils.config_utils import BaseConfig

    dataset_name = args.dataset
    llm_base_url = args.llm_base_url
    llm_name = args.llm_name

    save_dir = args.save_dir
    if save_dir == 'outputs':
        save_dir = os.path.join(save_dir, dataset_name)
    else:
        save_dir = f"{save_dir}_{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)

    corpus_path = f"reproduce/dataset/{dataset_name}_corpus.json"
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]

    force_index_from_scratch = string_to_bool(args.force_index_from_scratch)
    force_openie_from_scratch = string_to_bool(args.force_openie_from_scratch)

    samples = json.load(open(f"reproduce/dataset/{dataset_name}.json", "r", encoding="utf-8"))
    all_queries = [s['question'] for s in samples]

    gold_answers = get_gold_answers(samples)
    try:
        gold_docs = get_gold_docs(samples, dataset_name)
        assert len(all_queries) == len(gold_docs) == len(gold_answers)
    except Exception:
        gold_docs = None

    config = BaseConfig(
        save_dir=save_dir,
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        dataset=dataset_name,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=force_index_from_scratch,
        force_openie_from_scratch=force_openie_from_scratch,
        rerank_dspy_file_path="src/SRgraphrag/prompts/dspy_prompts/filter_deepseek3.2-Instruct.json",
        retrieval_top_k=200,
        linking_top_k=30,
        max_qa_steps=3,
        qa_top_k=5,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=4,
        max_new_tokens=None,
        corpus_len=len(corpus),
        openie_mode=args.openie_mode
    )

    logging.basicConfig(level=logging.INFO)
    srgraphrag = SRgraphrag(global_config=config)
    srgraphrag.index(docs)

    test_n = min(args.test_n, len(all_queries))
    test_queries = all_queries[:test_n]
    test_gold_docs = gold_docs[:test_n] if gold_docs is not None else None
    test_gold_answers = gold_answers[:test_n] if gold_answers is not None else None

    cap = int(args.subject_level_Entity_Cap)

    _ = run_pipeline(
        srgraphrag,
        mode=args.mode,
        dataset_name=dataset_name,
        queries=test_queries,
        gold_docs=test_gold_docs,
        gold_answers=test_gold_answers,
        subject_cap=cap,
        result_save_root=args.result_save_root,
        save_dir=save_dir,
        llm_name=llm_name,
        num_to_retrieve=args.num_to_retrieve,
        judge_model_name=args.judge_llm_name,
        eval_k_list=(5,),
        judge_concurrency=args.judge_concurrency,
        judge_batch_size=args.judge_batch_size,
        graph_search_mode=args.graph_search_mode,
        agent_apply_to=args.agent_apply_to,
        agent_budget=agent_budget_from_args(args),
        agent_fallback=args.agent_fallback,
        round1_replay_path=args.round1_replay_path,
    )


if __name__ == "__main__":
    main()

# example：
# Retrieval Only：
# python main.py --dataset musique --mode retrieve --llm_base_url https://api.deepseek.com/v1 --llm_name deepseek-chat --judge_llm_name deepseek-reasoner --save_dir outputs --result_save_root result_outputs
#
# Retrieval + QA：
# python main.py --dataset hotpotqa --mode qa --llm_base_url https://api.deepseek.com/v1 --llm_name deepseek-chat --judge_llm_name deepseek-reasoner --save_dir outputs
