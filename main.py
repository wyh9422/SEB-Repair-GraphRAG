import os
import json
import argparse
import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union, Optional

from src.SRgraphrag.SRgraphrag import SRgraphrag
from src.SRgraphrag.utils.misc_utils import string_to_bool
from src.SRgraphrag.utils.config_utils import BaseConfig

# os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"

os.environ["OPENAI_API_KEY"] = "sk-c033b4013f704ca389a0c832c7632fef"
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
    srgraphrag: SRgraphrag,
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
    num_to_retrieve: int = 200,
    judge_model_name: str = "deepseek-reasoner",
    eval_k_list: Tuple[int, ...] = (1, 2, 5, 30),
) -> None:
    """
    mode="retrieve": retrieve only.
    mode="qa": QA after retrieve.
    """
    assert mode in ("retrieve", "qa"), f"Unknown mode: {mode}"

    if mode == "retrieve":
        _ = srgraphrag.retrieve(
            queries=queries,
            gold_docs=gold_docs,
            num_to_retrieve=num_to_retrieve,
            eval_k_list=eval_k_list,
            result_save_root=result_save_root,
            dataset_name=dataset_name,
            subject_cap=subject_cap,
            judge_model_name=judge_model_name,
            keep_miss_distribution=True,
        )


    if mode == "qa":
        rag_ret = srgraphrag.rag_qa(
            queries=queries,
            gold_docs=gold_docs,
            gold_answers=gold_answers,
            dataset_name=dataset_name,
        )

        save_rag_qa_results_to_json(
            rag_qa_return=rag_ret,
            save_path=os.path.join(save_dir, f"rag_qa_{dataset_name}.json"),
            extra_info={
                "dataset": dataset_name,
                "llm": llm_name,
                "mode": mode,
                "subject_cap": subject_cap,
            },
            ensure_ascii=False,
        )


def main():
    parser = argparse.ArgumentParser(description="SEB_Repair_GraphRAG retrieval and QA")
    parser.add_argument('--dataset', type=str, default='musique', help='Dataset name')
    parser.add_argument('--llm_base_url', type=str, default='https://api.openai.com/v1', help='LLM base URL')
    parser.add_argument('--llm_name', type=str, default='deepseek-chat', help='LLM name')
    parser.add_argument('--judge_llm_name', type=str, default='deepseek-reasoner', help='judge_LLM name')
    parser.add_argument('--embedding_name', type=str, default='nvidia/NV-Embed-v2', help='embedding model name')

    parser.add_argument('--force_index_from_scratch', type=str, default='false')
    parser.add_argument('--force_openie_from_scratch', type=str, default='false')
    parser.add_argument('--openie_mode', choices=['online', 'offline'], default='online')

    parser.add_argument('--save_dir', type=str, default='outputs')
    parser.add_argument("--result_save_root", type=str, default=None)
    parser.add_argument("--subject_level_Entity_Cap", type=str, default="4")

    parser.add_argument("--mode", choices=["retrieve", "qa"], default="qa",
                        help="retrieve=only retrieval; qa=retrieval + QA")

    parser.add_argument("--test_n", type=int, default=1000)

    args = parser.parse_args()

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
        num_to_retrieve=200,
        judge_model_name=args.judge_llm_name,
        eval_k_list=(5,),
    )


if __name__ == "__main__":
    main()

# 示例：
# 仅检索：
# python main.py --dataset musique --mode retrieve --llm_base_url https://api.deepseek.com/v1 --llm_name deepseek-chat --judge_llm_name deepseek-reasoner --save_dir outputs --result_save_root result_outputs
#
# 检索+问答：
# python main.py --dataset hotpotqa --mode qa --llm_base_url https://api.deepseek.com/v1 --llm_name deepseek-chat --judge_llm_name deepseek-reasoner --save_dir outputs