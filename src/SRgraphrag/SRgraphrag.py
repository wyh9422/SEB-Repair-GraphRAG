import json
import os
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Union, Optional, List, Set, Dict, Any, Tuple, Literal
import numpy as np
import importlib
from collections import defaultdict
from transformers import HfArgumentParser
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from igraph import Graph
import igraph as ig
import numpy as np
from collections import defaultdict
import re
import time
from openai import BadRequestError

from .llm import _get_llm_class, BaseLLM
from .embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from .embedding_store import EmbeddingStore
from .information_extraction import OpenIE
from .information_extraction.openie_vllm_offline import VLLMOfflineOpenIE
from .information_extraction.openie_transformers_offline import TransformersOfflineOpenIE
from .evaluation.retrieval_eval import RetrievalRecall
from .evaluation.qa_eval import QAExactMatch, QAF1Score
from .prompts.linking import get_query_instruction
from .prompts.prompt_template_manager import PromptTemplateManager
from .rerank import DSPyFilter
from .utils.misc_utils import *
from .utils.misc_utils import NerRawOutput, TripleRawOutput
from .utils.embed_utils import retrieve_knn
from .utils.typing import Triple
from .utils.config_utils import BaseConfig

logger = logging.getLogger(__name__)

class SRgraphrag:

    def __init__(self,
                 global_config=None,
                 save_dir=None,
                 llm_model_name=None,
                 llm_base_url=None,
                 embedding_model_name=None,
                 embedding_base_url=None,
                 azure_endpoint=None,
                 azure_embedding_endpoint=None):
        """
        Initializes an instance of the class and its related components.

        Attributes:
            global_config (BaseConfig): The global configuration settings for the instance. An instance
                of BaseConfig is used if no value is provided.
            saving_dir (str): The directory where specific SRgraphrag instances will be stored. This defaults
                to `outputs` if no value is provided.
            llm_model (BaseLLM): The language model used for processing based on the global
                configuration settings.
            openie (Union[OpenIE, VLLMOfflineOpenIE]): The Open Information Extraction module
                configured in either online or offline mode based on the global settings.
            graph: The graph instance initialized by the `initialize_graph` method.
            embedding_model (BaseEmbeddingModel): The embedding model associated with the current
                configuration.
            chunk_embedding_store (EmbeddingStore): The embedding store handling chunk embeddings.
            entity_embedding_store (EmbeddingStore): The embedding store handling entity embeddings.
            fact_embedding_store (EmbeddingStore): The embedding store handling fact embeddings.
            prompt_template_manager (PromptTemplateManager): The manager for handling prompt templates
                and roles mappings.
            openie_results_path (str): The file path for storing Open Information Extraction results
                based on the dataset and LLM name in the global configuration.
            rerank_filter (Optional[DSPyFilter]): The filter responsible for reranking information
                when a rerank file path is specified in the global configuration.
            ready_to_retrieve (bool): A flag indicating whether the system is ready for retrieval
                operations.

        Parameters:
            global_config: The global configuration object. Defaults to None, leading to initialization
                of a new BaseConfig object.
            working_dir: The directory for storing working files. Defaults to None, constructing a default
                directory based on the class name and timestamp.
            llm_model_name: LLM model name, can be inserted directly as well as through configuration file.
            embedding_model_name: Embedding model name, can be inserted directly as well as through configuration file.
            llm_base_url: LLM URL for a deployed LLM model, can be inserted directly as well as through configuration file.
        """
        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        #Overwriting Configuration if Specified
        if save_dir is not None:
            self.global_config.save_dir = save_dir

        if llm_model_name is not None:
            self.global_config.llm_name = llm_model_name

        if embedding_model_name is not None:
            self.global_config.embedding_model_name = embedding_model_name

        if llm_base_url is not None:
            self.global_config.llm_base_url = llm_base_url

        if embedding_base_url is not None:
            self.global_config.embedding_base_url = embedding_base_url

        if azure_endpoint is not None:
            self.global_config.azure_endpoint = azure_endpoint

        if azure_embedding_endpoint is not None:
            self.global_config.azure_embedding_endpoint = azure_embedding_endpoint

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self.global_config).items()])
        logger.debug(f"SRgraphrag init with config:\n  {_print_config}\n")

        #LLM and embedding model specific working directories are created under every specified saving directories
        llm_label = self.global_config.llm_name.replace("/", "_")
        embedding_label = self.global_config.embedding_model_name.replace("/", "_")
        self.working_dir = os.path.join(self.global_config.save_dir, f"{llm_label}_{embedding_label}")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        self.llm_model: BaseLLM = _get_llm_class(self.global_config)

        if self.global_config.openie_mode == 'online':
            self.openie = OpenIE(llm_model=self.llm_model)
        elif self.global_config.openie_mode == 'offline':
            self.openie = VLLMOfflineOpenIE(self.global_config)
        elif self.global_config.openie_mode ==  'Transformers-offline':
            self.openie = TransformersOfflineOpenIE(self.global_config)

        self.graph = self.initialize_graph()

        if self.global_config.openie_mode == 'offline':
            self.embedding_model = None
        else:
            self.embedding_model: BaseEmbeddingModel = _get_embedding_model_class(
                embedding_model_name=self.global_config.embedding_model_name)(global_config=self.global_config,
                                                                              embedding_model_name=self.global_config.embedding_model_name)
        self.chunk_embedding_store = EmbeddingStore(self.embedding_model,
                                                    os.path.join(self.working_dir, "chunk_embeddings"),
                                                    self.global_config.embedding_batch_size, 'chunk')
        # 实体语义（phrases / nodes）
        self.entity_embedding_store = EmbeddingStore(self.embedding_model,
                                                     os.path.join(self.working_dir, "entity_embeddings"),
                                                     self.global_config.embedding_batch_size, 'entity')
        # 事实语义（triples）
        self.fact_embedding_store = EmbeddingStore(self.embedding_model,
                                                   os.path.join(self.working_dir, "fact_embeddings"),
                                                   self.global_config.embedding_batch_size, 'fact')

        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})

        self.openie_results_path = os.path.join(self.global_config.save_dir,f'openie_results_ner_{self.global_config.llm_name.replace("/", "_")}.json')

        self.rerank_filter = DSPyFilter(self)

        self.ready_to_retrieve = False

        self.ppr_time = 0
        self.rerank_time = 0
        self.all_retrieval_time = 0

        self.ent_node_to_chunk_ids = None


    def initialize_graph(self):
        """
        Initializes a graph using a Pickle file if available or creates a new graph.

        The function attempts to load a pre-existing graph stored in a Pickle file. If the file
        is not present or the graph needs to be created from scratch, it initializes a new directed
        or undirected graph based on the global configuration. If the graph is loaded successfully
        from the file, pertinent information about the graph (number of nodes and edges) is logged.

        Returns:
            ig.Graph: A pre-loaded or newly initialized graph.

        Raises:
            None
        """
        self._graph_pickle_filename = os.path.join(
            self.working_dir, f"graph.pickle"
        )

        preloaded_graph = None

        if not self.global_config.force_index_from_scratch:
            if os.path.exists(self._graph_pickle_filename):
                preloaded_graph = ig.Graph.Read_Pickle(self._graph_pickle_filename)

        if preloaded_graph is None:
            return ig.Graph(directed=self.global_config.is_directed_graph)
        else:
            logger.info(
                f"Loaded graph from {self._graph_pickle_filename} with {preloaded_graph.vcount()} nodes, {preloaded_graph.ecount()} edges"
            )
            return preloaded_graph

    def pre_openie(self,  docs: List[str]):
        logger.info(f"Indexing Documents")
        logger.info(f"Performing OpenIE Offline")

        chunks = self.chunk_embedding_store.get_missing_string_hash_ids(docs)

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunks.keys())
        new_openie_rows = {k : chunks[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        assert False, logger.info('Done with OpenIE, run online indexing for future retrieval.')

    def index(self, docs: List[str]):
        """
        Indexes the given documents based on the SRgraphrag 2 framework which generates an OpenIE knowledge graph
        based on the given documents and encodes passages, entities and facts separately for later retrieval.
        把输入文档 → 变成：
        1） passage embeddings
        2） entity embeddings
        3） fact embeddings（OpenIE triples）
        4） entity–fact–chunk 图结构
        5）并写入图数据库，用于后续检索。

        Parameters:
            docs : List[str]
                A list of documents to be indexed.
        """

        logger.info(f"Indexing Documents")

        logger.info(f"Performing OpenIE")

        if self.global_config.openie_mode == 'offline':
            self.pre_openie(docs)

        self.chunk_embedding_store.insert_strings(docs)
        chunk_to_rows = self.chunk_embedding_store.get_all_id_to_rows()

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunk_to_rows.keys())
        new_openie_rows = {k : chunk_to_rows[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

        assert len(chunk_to_rows) == len(ner_results_dict) == len(triple_results_dict), f"len(chunk_to_rows): {len(chunk_to_rows)}, len(ner_results_dict): {len(ner_results_dict)}, len(triple_results_dict): {len(triple_results_dict)}"

        # prepare data_store
        chunk_ids = list(chunk_to_rows.keys())

        chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in chunk_ids]
        entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
        facts = flatten_facts(chunk_triples)

        logger.info(f"Encoding Entities")
        self.entity_embedding_store.insert_strings(entity_nodes)

        logger.info(f"Encoding Facts")
        self.fact_embedding_store.insert_strings([str(fact) for fact in facts])

        logger.info(f"Constructing Graph")

        self.node_to_node_stats = {}
        self.ent_node_to_chunk_ids = {}

        self.add_fact_edges(chunk_ids, chunk_triples) #add_fact_edges：实体 ↔ 实体，来自 triple 的 (subject, object)
        num_new_chunks = self.add_passage_edges(chunk_ids, chunk_triple_entities) #add_passage_edges：passage ↔ 实体，把 chunk 和它包含的实体连起来

        if num_new_chunks > 0:
            logger.info(f"Found {num_new_chunks} new chunks to save into graph.")
            self.add_synonymy_edges() #add_synonymy_edges：实体 ↔ 实体（同义 / 语义相似），来自 embedding 相似度 KNN

            self.augment_graph() #augment_graph = 把内存里的 node/edge 暂存记录 → 真正写入 igraph.Graph 结构
            self.save_igraph()

    def delete(self, docs_to_delete: List[str]):
        """
        Deletes the given documents from all data structures within the SRgraphrag class.
        Note that triples and entities which are indexed from chunks that are not being removed will not be removed.

        Parameters:
            docs : List[str]
                A list of documents to be deleted.
        """

        #Making sure that all the necessary structures have been built.
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        current_docs = set(self.chunk_embedding_store.get_all_texts())
        docs_to_delete = [doc for doc in docs_to_delete if doc in current_docs]

        #Get ids for chunks to delete
        chunk_ids_to_delete = set(
            [self.chunk_embedding_store.text_to_hash_id[chunk] for chunk in docs_to_delete])

        #Find triples in chunks to delete
        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])
        triples_to_delete = []

        all_openie_info_with_deletes = []

        for openie_doc in all_openie_info:
            if openie_doc['idx'] in chunk_ids_to_delete:
                triples_to_delete.append(openie_doc['extracted_triples'])
            else:
                all_openie_info_with_deletes.append(openie_doc)

        triples_to_delete = flatten_facts(triples_to_delete)

        #Filter out triples that appear in unaltered chunks
        true_triples_to_delete = []

        for triple in triples_to_delete:
            proc_triple = tuple(text_processing(list(triple)))

            doc_ids = self.proc_triples_to_docs[str(proc_triple)]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                true_triples_to_delete.append(triple)

        processed_true_triples_to_delete = [[text_processing(list(triple)) for triple in true_triples_to_delete]]
        entities_to_delete, _ = extract_entity_nodes(processed_true_triples_to_delete)
        processed_true_triples_to_delete = flatten_facts(processed_true_triples_to_delete)

        triple_ids_to_delete = set([self.fact_embedding_store.text_to_hash_id[str(triple)] for triple in processed_true_triples_to_delete])

        #Filter out entities that appear in unaltered chunks
        ent_ids_to_delete = [self.entity_embedding_store.text_to_hash_id[ent] for ent in entities_to_delete]

        filtered_ent_ids_to_delete = []

        for ent_node in ent_ids_to_delete:
            doc_ids = self.ent_node_to_chunk_ids[ent_node]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                filtered_ent_ids_to_delete.append(ent_node)

        logger.info(f"Deleting {len(chunk_ids_to_delete)} Chunks")
        logger.info(f"Deleting {len(triple_ids_to_delete)} Triples")
        logger.info(f"Deleting {len(filtered_ent_ids_to_delete)} Entities")

        self.save_openie_results(all_openie_info_with_deletes)

        self.entity_embedding_store.delete(filtered_ent_ids_to_delete)
        self.fact_embedding_store.delete(triple_ids_to_delete)
        self.chunk_embedding_store.delete(chunk_ids_to_delete)

        #Delete Nodes from Graph
        self.graph.delete_vertices(list(filtered_ent_ids_to_delete) + list(chunk_ids_to_delete))
        self.save_igraph()

        self.ready_to_retrieve = False

    def judge_answerability_and_bridge(
        self,
        queries: list[str],
        top5_docs_list: list[list[str]],
        *,
        judge_model_name: str = "deepseek-reasoner",
        judge_concurrency: int = 100,
        judge_batch_size: int = 100,
        prompt_json: str = "src/SRgraphrag/prompts/dspy_prompts/judge_prompt.json",
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
        import os, json, asyncio, random
        import aiohttp

        assert len(queries) == len(top5_docs_list), "len(queries) must equal len(top5_docs_list)"

        BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
        MODEL = judge_model_name
        API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY is empty")

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
            url = f"{BASE_URL}/v1/chat/completions"
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
                return json.loads(content)
            except Exception:
                l = content.find("{")
                r = content.rfind("}")
                if l != -1 and r != -1 and r > l:
                    try:
                        return json.loads(content[l:r + 1])
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
                "temperature": 0.0,
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
                r = await _judge_one(sem, session, all_queries[i], all_top5[i])
                return i, r

            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                for start in range(0, total_n, max(1, judge_batch_size)):
                    end = min(total_n, start + max(1, judge_batch_size))
                    tasks = [asyncio.create_task(_judge_one_with_idx(i)) for i in range(start, end)]

                    for fut in asyncio.as_completed(tasks):
                        try:
                            i, r = await fut
                            results[i] = r
                        except Exception as e:
                            i = start
                            results[i] = {
                                "can_answer": False,
                                "evidence_ids": [],
                                "evidence_docs": [],
                                "analysis_zh": f"judge异常: {type(e).__name__}",
                                "bridge_possible": False,
                                "bridge_question": "",
                                "bridge_evidence_ids": [],
                                "bridge_analysis_zh": "",
                                "_error": "judge_exception",
                                "_raw": str(e),
                            }
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

    def retrieve_full_once(
        self,
        query: str,
        *,
        num_to_retrieve: int,
        subject_cap: int = 4,
        evidence: list[str] | None = None,   # if provided: force-keep in final top5 (second-round behavior)
    ) -> dict:
        """
        Run ONE FULL retrieval for a single query:
        DPR -> (triple rerank+filter) -> PPR (if runnable) -> (evidence injection, optional) -> guard(dpr1 of THIS round) -> final_top5

        Key invariants when evidence is provided:
        - Any missing evidence must be injected into top5 by tail replacement, but NEVER replace existing evidence.
        - Guard (dpr1 tail replacement) uses THIS round's dpr1, and also NEVER replaces evidence.
        - If evidence already all inside top5, injection does nothing (no extra tail replace).
        """
        import re
        import time
        import numpy as np

        # -------------------------
        # build instruction by subject_cap
        # -------------------------
        def _cap_to_word(n: int) -> str:
            m = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
            return m.get(n, "four")

        def _build_subject_cap_instruction(n: int) -> str:
            n = int(n)
            n = max(1, min(5, n))

            base = (
                "You are a critical component of a high-stakes question-answering system used by top researchers "
                "and decision-makers worldwide. Your task is to filter facts based on their relevance to a given query, "
                "ensuring that the most crucial information is presented to these stakeholders. The query requires careful "
                "analysis and possibly multi-hop reasoning to connect different pieces of information. "
                "You must select all facts from the provided candidate list that are strongly relevant to the query, "
                "prioritizing at most four key subjects and including as many relevant facts as possible under those subjects, "
                "while ensuring that the total number of unique subjects among the selected facts does not exceed four, "
                "aiding in reasoning and providing an accurate answer. "
                'The output should be in JSON format, e.g., {"fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}, '
                'and if no facts are relevant, return an empty list, {"fact": []}. '
                "The accuracy of your response is paramount, as it will directly impact the decisions made by these high-level stakeholders. "
                "You must only use facts from the candidate list and not generate new facts."
            )
            word = _cap_to_word(n)
            out = re.sub(r"\bfour\b", word, base)
            out = re.sub(r"\b4\b", str(n), out)
            return out

        subject_cap = int(subject_cap)
        subject_cap = max(1, min(5, subject_cap))
        llm_filter_instruction = _build_subject_cap_instruction(subject_cap)

        if num_to_retrieve < 5:
            num_to_retrieve = 5

        # -------------------------
        # helpers
        # -------------------------
        def _count_unique_subjects(triples):
            if not triples:
                return 0
            return len({t[0] for t in triples if isinstance(t, (list, tuple)) and len(t) >= 3})

        def _dedup_preserve(xs: list[str]) -> list[str]:
            out, seen = [], set()
            for x in xs:
                if x and x not in seen:
                    out.append(x)
                    seen.add(x)
            return out

        def _find_tail_replace_idx(cand5: list[str], protected: set[str]) -> int | None:
            # from tail -> head, pick first slot whose doc is non-empty AND not protected
            for j in range(4, -1, -1):
                if cand5[j] and cand5[j] not in protected:
                    return j
            # if all are protected or empty, allow replacing an empty slot that is not protected (rare)
            for j in range(4, -1, -1):
                if (not cand5[j]) and cand5[j] not in protected:
                    return j
            return None

        def apply_evidence_injection_top5(
            top5: list[str],
            evidence_docs: list[str] | None,
            base_ranked_docs: list[str] | None,
        ) -> tuple[list[str], dict]:
            """
            Inject missing evidence into top5 by tail replacement, never replacing existing evidence.
            If evidence already covered, do nothing (no extra tail replacement).
            """
            log = {
                "evidence_enabled": bool(evidence_docs),
                "evidence_used": False,
                "evidence_missing_cnt": 0,
                "evidence_missing": [],
                "replaced_cnt": 0,
            }

            cand = list(top5 or [])[:5]
            while len(cand) < 5:
                cand.append("")

            ev = [x for x in (evidence_docs or []) if x]
            ev = _dedup_preserve(ev)[:5]  # caller should pass <=5, still guard here
            if not ev:
                # normalize (dedup+refill) but no injection
                out = _dedup_preserve(cand)
                if len(out) < 5 and base_ranked_docs:
                    out += [x for x in base_ranked_docs if x and x not in set(out)]
                out = out[:5]
                while len(out) < 5:
                    out.append("")
                return out, log

            ev_set = set(ev)
            cand_set = set([x for x in cand if x])
            missing = [x for x in ev if x not in cand_set]

            log["evidence_missing"] = missing[:]
            log["evidence_missing_cnt"] = len(missing)
            log["evidence_used"] = (len(missing) > 0)

            replaced = 0
            if missing:
                # replace from tail: pick first slot that is NOT evidence
                for e in missing:
                    idx = _find_tail_replace_idx(cand, protected=ev_set)
                    if idx is None:
                        break
                    cand[idx] = e
                    replaced += 1

            log["replaced_cnt"] = replaced

            # dedup + refill
            out = _dedup_preserve(cand)
            if len(out) < 5 and base_ranked_docs:
                out += [x for x in base_ranked_docs if x and x not in set(out)]
            out = out[:5]
            while len(out) < 5:
                out.append("")

            # invariant best-effort: all ev that can fit should be in out
            # (if ev>5 we truncated; if protected slots fill all 5, cannot inject more)
            return out, log

        def apply_guard_top5_protect_evidence(
            top5: list[str],
            dpr1: str | None,
            protected_docs: set[str],
        ) -> tuple[list[str], dict]:
            """
            Tail replace with THIS round's dpr1 if not already present.
            Never replace protected evidence docs.
            """
            log = {"guard_enabled": True, "guard_triggered": False, "guard_replaced_idx": None}

            if not dpr1:
                return top5, log
            if dpr1 in set([x for x in top5 if x]):
                return top5, log

            cand = list(top5 or [])[:5]
            while len(cand) < 5:
                cand.append("")

            idx = _find_tail_replace_idx(cand, protected=protected_docs)
            if idx is None:
                return cand[:5], log

            cand[idx] = dpr1
            log["guard_triggered"] = True
            log["guard_replaced_idx"] = int(idx)

            # dedup but keep length 5 with refill by internal caller
            out = _dedup_preserve(cand)[:5]
            while len(out) < 5:
                out.append("")
            return out, log
        
        def _align_scores_to_top5(top5_docs: list[str], base_docs: list[str], base_scores):
            if base_scores is None:
                base_scores_list = []
            else:
                base_scores_list = [float(x) for x in list(base_scores)]
            base_docs_list = list(base_docs or [])

            doc2score = {}
            for d, s in zip(base_docs_list, base_scores_list):
                if d and d not in doc2score:
                    doc2score[d] = float(s)

            out_scores = []
            for d in top5_docs:
                out_scores.append(float(doc2score.get(d, float("-inf"))) if d else float("-inf"))
            return out_scores

        # -------------------------
        # A) DPR (THIS round)
        # -------------------------
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        dpr_top_docs = [
            self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"]
            for idx in dpr_sorted_doc_ids[:num_to_retrieve]
        ]
        dpr_top5 = dpr_top_docs[:5]
        dpr1 = dpr_top_docs[0] if dpr_top_docs else None

        # -------------------------
        # B) FULL: triple -> filter -> PPR (or fallback)
        # -------------------------
        ppr_runnable = False
        ppr_top_docs = None
        final_source = "dpr_fallback"
        rerank_log = {}
        raw_top30_triples = None
        filtered_triples = None
        unique_subj_before = 0
        unique_subj_after = 0

        rerank_start = time.time()
        query_fact_scores = self.get_fact_scores(query)

        top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(
            query,
            query_fact_scores,
            instruction=llm_filter_instruction,
        )
        raw_top30_triples = rerank_log.get("facts_before_rerank", None)
        filtered_triples = rerank_log.get("facts_after_rerank", None)

        rerank_end = time.time()
        self.rerank_time += (rerank_end - rerank_start)

        unique_subj_before = _count_unique_subjects(raw_top30_triples)
        unique_subj_after = _count_unique_subjects(filtered_triples)

        ppr_runnable = (len(top_k_facts) > 0)

        if ppr_runnable:
            ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.graph_search_with_fact_entities(
                query=query,
                link_top_k=self.global_config.linking_top_k,
                query_fact_scores=query_fact_scores,
                top_k_facts=top_k_facts,
                top_k_fact_indices=top_k_fact_indices,
                passage_node_weight=self.global_config.passage_node_weight,
            )
            ppr_top_docs = [
                self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"]
                for idx in ppr_sorted_doc_ids[:num_to_retrieve]
            ]
            final_docs_base = list(ppr_top_docs)
            final_scores = ppr_sorted_doc_scores[:num_to_retrieve]
            final_source = "graph"
        else:
            final_docs_base = list(dpr_top_docs)
            final_scores = dpr_sorted_doc_scores[:num_to_retrieve]
            final_source = "dpr_fallback"

        # base top5 before injection/guard
        base_top5 = list(final_docs_base[:5])
        while len(base_top5) < 5:
            base_top5.append("")

        # -------------------------
        # C) evidence injection (only meaningful when evidence is provided)
        #    MUST happen before guard; protects evidence in subsequent guard.
        # -------------------------
        ev = [x for x in (evidence or []) if x]
        ev = _dedup_preserve(ev)[:5]
        protected_set = set(ev)

        top5_after_inj, ev_log = apply_evidence_injection_top5(
            top5=base_top5,
            evidence_docs=ev if ev else None,
            base_ranked_docs=final_docs_base,
        )
        # refresh protected_set using ev (not depending on injection result)
        protected_set = set(ev)

        # -------------------------
        # D) guard (tail replace) using THIS round dpr1, never replacing evidence
        # -------------------------
        top5_after_guard, guard_log = apply_guard_top5_protect_evidence(
            top5=top5_after_inj,
            dpr1=dpr1,
            protected_docs=protected_set,
        )

        # -------------------------
        # E) final normalize: dedup + refill from base ranked docs (does not "inject" extra evidence)
        # -------------------------
        final_top5 = _dedup_preserve(top5_after_guard)
        if len(final_top5) < 5:
            seen = set(final_top5)
            for x in (final_docs_base or []):
                if len(final_top5) == 5:
                    break
                if x and x not in seen:
                    final_top5.append(x)
                    seen.add(x)
        final_top5 = final_top5[:5]
        while len(final_top5) < 5:
            final_top5.append("")

        # E2) align scores with final_top5 after injection/guard
        if final_scores is not None:
            aligned_top5_scores = _align_scores_to_top5(final_top5, final_docs_base, final_scores)
            final_scores = [float(x) for x in list(final_scores)]
            if len(final_scores) < len(final_docs_base):
                final_scores += [float("-inf")] * (len(final_docs_base) - len(final_scores))
            final_scores[:5] = aligned_top5_scores
        

        # pad final_docs to 5-only
        final_docs = list(final_docs_base)
        if len(final_docs) < 5:
            final_docs = final_docs + [""] * (5 - len(final_docs))
        final_docs[:5] = final_top5

        return {
            "query": query,

            "final_top5": final_top5,
            "final_docs": final_docs,
            "final_scores": final_scores,

            # round DPR logs
            "dpr_top5": dpr_top5,
            "dpr1": dpr1,

            # graph logs
            "graph_top5": (ppr_top_docs[:5] if ppr_top_docs else None),
            "graph_runnable": bool(ppr_runnable),
            "final_source": final_source,

            # filter logs
            "subject_cap": int(subject_cap),
            "filter_instruction": llm_filter_instruction,
            "raw_top30_triples": raw_top30_triples,
            "filtered_triples": filtered_triples,
            "unique_subject_count_before": unique_subj_before,
            "unique_subject_count_after": unique_subj_after,

            # injection logs
            "evidence_docs": (ev if ev else None),
            "evidence_injected": bool(ev_log.get("evidence_used", False)),
            "evidence_missing": ev_log.get("evidence_missing", []),
            "evidence_missing_cnt": int(ev_log.get("evidence_missing_cnt", 0)),
            "evidence_replaced_cnt": int(ev_log.get("replaced_cnt", 0)),

            # guard logs
            "guard_triggered@5": bool(guard_log.get("guard_triggered", False)),
            "guard_replaced_idx@5": guard_log.get("guard_replaced_idx", None),
        }

    def retrieve(
        self,
        queries: list[str],
        gold_docs: list[list[str]] | None = None,
        *,
        num_to_retrieve: int | None = None,
        result_save_root: str | None = None,
        eval_k_list: tuple[int, ...] = (1, 2, 5, 30),
        dataset_name: str = "unknown_dataset",
        subject_cap: int = 4,
        judge_concurrency: int = 100,
        judge_batch_size: int = 100,
        judge_model_name: str = "deepseek-reasoner",
        keep_miss_distribution: bool = True,   # whether to keep/save miss distribution
    ) -> list["QuerySolution"] | tuple[list["QuerySolution"], dict]:
        """
        Iterative retrieval (max 2 rounds):
        Phase-1: all queries run Round-1 FULL (sequential, heavy local)
        Phase-2: batched concurrent judge on round1 top5 (DeepSeek)
        Phase-3: only bridge_possible run Round-2 FULL (sequential)
        Final: keep original query's docs = round1 docs, but overwrite top5 with:
                - if round2 executed: use round2 result top5 (already injected evidence inside retrieve_full_once)
                - else: round1 top5
        """
        import os, json, time
        from collections import defaultdict
        from tqdm import tqdm

        # -------------------------
        # setup
        # -------------------------
        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k
        if num_to_retrieve < 5:
            num_to_retrieve = 5

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        out_dir = None
        if result_save_root is not None:
            out_dir = os.path.join(result_save_root, dataset_name, f"ITER_cap{int(subject_cap)}")
            os.makedirs(out_dir, exist_ok=True)

        metrics_summary_path = os.path.join(out_dir, f"{dataset_name}_metrics_summary.json") if out_dir else None

        # -------------------------
        # helpers (metrics)
        # -------------------------
        def avg_recall_at_k(golds: list[list[str]], retrieved_topk_list: list[list[str]]) -> float:
            n = len(golds)
            s = 0.0
            denom = 0
            for i in range(n):
                g = golds[i] or []
                if len(g) == 0:
                    continue
                denom += 1
                rset = set(retrieved_topk_list[i])
                hit = sum(1 for x in g if x in rset)
                s += hit / len(g)
            return s / denom if denom > 0 else 0.0

        def all_recall_at_k(golds: list[list[str]], retrieved_topk_list: list[list[str]]) -> float:
            n = len(golds)
            hit = 0
            denom = 0
            for i in range(n):
                g = golds[i] or []
                if len(g) == 0:
                    continue
                denom += 1
                rset = set(retrieved_topk_list[i])
                if all(x in rset for x in g):
                    hit += 1
            return hit / denom if denom > 0 else 0.0

        def missing_list(gold: list[str], topk_docs: list[str]) -> list[str]:
            rset = set(topk_docs)
            return [g for g in (gold or []) if g not in rset]

        def hit_at_1(gold: list[str], ranked: list[str]) -> int:
            if not gold or not ranked:
                return 0
            return 1 if ranked[0] in set(gold) else 0

        def mrr_first_hit(gold: list[str], ranked: list[str]) -> float:
            if not gold:
                return 0.0
            gset = set(gold)
            for idx, doc in enumerate(ranked):
                if doc in gset:
                    return 1.0 / (idx + 1)
            return 0.0

        # -------------------------
        # run
        # -------------------------
        retrieval_results: list["QuerySolution"] = []
        final_top5_all: list[list[str]] = []

        # only allocate miss containers if needed
        miss_bucket_top5: dict[str, list[dict]] | None = defaultdict(list) if keep_miss_distribution else None
        miss_bucket_counts_top5 = defaultdict(int) if keep_miss_distribution else None

        second_round_cnt = 0
        hit1_sum = 0
        mrr_sum = 0.0

        t0 = time.time()

        # Phase-1: round1 for all
        round1_pack = [None] * len(queries)
        round1_top5 = [None] * len(queries)

        for q_idx, query in tqdm(
            enumerate(queries),
            desc=f"IterRetrieve-R1[{dataset_name}|cap{int(subject_cap)}]",
            total=len(queries),
        ):
            r1 = self.retrieve_full_once(
                query=query,
                num_to_retrieve=num_to_retrieve,
                subject_cap=subject_cap,
                evidence=None,
            )
            round1_pack[q_idx] = r1
            round1_top5[q_idx] = r1["final_top5"]

        # Phase-2: judge for all (batched concurrent)
        judge1_all = self.judge_answerability_and_bridge(
            judge_model_name=judge_model_name,
            queries=queries,
            top5_docs_list=round1_top5,
            judge_concurrency=judge_concurrency,
            judge_batch_size=judge_batch_size,
        )

        # Phase-3: round2 only for bridge_possible
        for q_idx, query in tqdm(
            enumerate(queries),
            desc=f"IterRetrieve-R2[{dataset_name}|cap{int(subject_cap)}]",
            total=len(queries),
        ):
            r1 = round1_pack[q_idx]
            top5_r1 = round1_top5[q_idx]
            judge1 = judge1_all[q_idx]

            can_answer = bool(judge1.get("can_answer", False))
            evidence_docs_r1 = judge1.get("evidence_docs", []) or []

            final_top5 = top5_r1
            round_used = 1
            r2 = None

            if (not can_answer) and bool(judge1.get("bridge_possible", False)):
                bridge_q = str(judge1.get("bridge_question", "") or "").strip()
                if bridge_q:
                    second_round_cnt += 1
                    round_used = 2

                    r2 = self.retrieve_full_once(
                        query=bridge_q,
                        num_to_retrieve=num_to_retrieve,
                        subject_cap=subject_cap,
                        evidence=evidence_docs_r1,
                    )
                    final_top5 = r2["final_top5"]

            # keep docs container from r1 for compatibility; overwrite top5
            final_docs = list(r1["final_docs"])
            final_docs[:5] = final_top5
            retrieval_results.append(QuerySolution(question=query, docs=final_docs, doc_scores=r1["final_scores"]))
            final_top5_all.append(final_top5)

            gold = gold_docs[q_idx] if gold_docs is not None else None
            if gold is not None:
                hit1_sum += hit_at_1(gold, final_top5)
                mrr_sum += mrr_first_hit(gold, final_top5)

                if keep_miss_distribution:
                    miss5 = missing_list(gold, final_top5)
                    miss5_cnt = len(miss5)
                    miss_bucket_counts_top5[miss5_cnt] += 1  # type: ignore

                    bucket_key = f"miss_{miss5_cnt}"
                    miss_bucket_top5[bucket_key].append({  # type: ignore
                        "qid": q_idx,
                        "query": query,
                        "gold_docs": gold,

                        "round_used": int(round_used),
                        "second_round_used": bool(round_used == 2),

                        "final_top5_docs": final_top5,
                        "missing_gold_at5": miss5,
                        "missing_count_at5": miss5_cnt,
                        "is_all_recall_at5": (miss5_cnt == 0),

                        "round1": {
                            "final_top5_docs": top5_r1,
                            "dpr_top5_docs": r1.get("dpr_top5"),
                            "graph_top5_docs": r1.get("graph_top5"),
                            "graph_runnable": r1.get("graph_runnable"),
                            "guard_triggered@5": r1.get("guard_triggered@5"),
                            "final_source": r1.get("final_source"),
                            "raw_top30_triples": r1.get("raw_top30_triples"),
                            "filtered_triples": r1.get("filtered_triples"),
                            "unique_subject_count_before": r1.get("unique_subject_count_before"),
                            "unique_subject_count_after": r1.get("unique_subject_count_after"),
                        },
                        "judge1": judge1,

                        "round2": (None if r2 is None else {
                            "bridge_question": str(judge1.get("bridge_question", "") or ""),
                            "final_top5_docs": r2.get("final_top5"),
                            "final_source": r2.get("final_source"),
                            "graph_runnable": r2.get("graph_runnable"),
                            "evidence_docs_forced": evidence_docs_r1,
                        }),
                    })

        t1 = time.time()
        self.all_retrieval_time += (t1 - t0)

        if gold_docs is None:
            return retrieval_results

        total_q = len(gold_docs)

        metrics = {
            "dataset": dataset_name,
            "setting": f"ITER_FULL_cap{int(subject_cap)}_max2",
            "subject_cap": int(subject_cap),
            "total_queries": int(total_q),
            "time_sec": float(t1 - t0),

            "second_round_cnt": int(second_round_cnt),
            "second_round_ratio": float(second_round_cnt / total_q) if total_q > 0 else 0.0,

            "recall@5": float(avg_recall_at_k(gold_docs, final_top5_all)),
            "all_recall@5": float(all_recall_at_k(gold_docs, final_top5_all)),
            "Hit@1": float(hit1_sum / total_q) if total_q > 0 else 0.0,
            "MRR@5": float(mrr_sum / total_q) if total_q > 0 else 0.0,

            "judge_concurrency": int(judge_concurrency),
            "judge_batch_size": int(judge_batch_size),
            "judge_model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            "judge_base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
        }

        if keep_miss_distribution:
            metrics["miss_bucket@5_counts"] = {
                f"miss_{m}": int(c)
                for m, c in sorted(miss_bucket_counts_top5.items(), key=lambda x: x[0])  # type: ignore
            }
            metrics["miss_bucket@5_rates"] = {
                f"miss_{m}": (float(c) / total_q if total_q > 0 else 0.0)
                for m, c in sorted(miss_bucket_counts_top5.items(), key=lambda x: x[0])  # type: ignore
            }

        if out_dir:
            if metrics_summary_path:
                with open(metrics_summary_path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, ensure_ascii=False, indent=2)

            # only save miss bucket files when enabled
            if keep_miss_distribution:
                for bucket, items in (miss_bucket_top5 or {}).items():
                    path = os.path.join(out_dir, f"{bucket}_top5.json")
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(items, f, ensure_ascii=False, indent=2)

        logger.info(f"=== Iterative Top-5 Summary [{dataset_name} | cap{int(subject_cap)} | max2] ===")
        logger.info(
            f"recall@5={metrics['recall@5']:.4f} | all_recall@5={metrics['all_recall@5']:.4f} | "
            f"Hit@1={metrics['Hit@1']:.4f} | MRR@5={metrics['MRR@5']:.4f} | "
            f"second_round={metrics['second_round_cnt']}/{total_q} ({metrics['second_round_ratio']:.4f}) | "
            f"judge_conc={metrics['judge_concurrency']} batch={metrics['judge_batch_size']}"
        )
        if keep_miss_distribution:
            logger.info(f"miss buckets @5: {metrics.get('miss_bucket@5_counts', {})}")

        return retrieval_results, metrics

    def rag_qa(self,
               queries: List[str|QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None,
               dataset_name: str = "unknown_dataset") -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        Performs retrieval-augmented generation enhanced QA using the SRgraphrag 2 framework.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it returns answers only or additionally evaluate retrieval and answer quality using
        recall @ k, exact match and F1 score metrics.

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): A list of lists containing gold-standard documents for
                each query. This is used if document-level evaluation is to be performed. Default is None.
            gold_answers (Optional[List[List[str]]]): A list of lists containing gold-standard answers for
                each query. Required if evaluation of question answering (QA) answers is enabled. Default
                is None.

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: A tuple that always includes:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
                If evaluation is enabled, the tuple also includes:
                - A dictionary with overall results from the retrieval phase (if applicable).
                - A dictionary with overall QA evaluation metrics (exact match and F1 scores).

        """
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        # Retrieving (if necessary)
        overall_retrieval_result = None

        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve(queries=queries,gold_docs=gold_docs,eval_k_list=[5],dataset_name=dataset_name)
            else:
                queries = self.retrieve(queries=queries)

        # Performing QA
        queries_solutions, all_response_message, all_metadata = self.qa(queries)

        # Evaluating QA
        if gold_answers is not None:
            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)

            # round off to 4 decimal places for QA results
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # Save retrieval and QA results
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results
        else:
            return queries_solutions, all_response_message, all_metadata

    def retrieve_dpr(self,
                     queries: List[str],
                     num_to_retrieve: int = None,
                     gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Performs retrieval using a DPR framework, which consists of several steps:
        - Dense passage scoring

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                A list of lists containing gold-standard documents corresponding to each query. Required
                if retrieval performance evaluation is enabled (`do_eval_retrieval` in global configuration).

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                If retrieval performance evaluation is not enabled, returns a list of QuerySolution objects, each containing
                the retrieved documents and their scores for the corresponding query. If evaluation is enabled, also returns
                a dictionary containing the evaluation metrics computed over the retrieved results.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """
        retrieve_start_time = time.time()  # Record start time

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        retrieval_results = []

        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            logger.info('No facts found after reranking, return DPR results')
            sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)

            top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in
                          sorted_doc_ids[:num_to_retrieve]]

            retrieval_results.append(
                QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve]))

        retrieve_end_time = time.time()  # Record end time

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")

        # Evaluate retrieval
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(
                gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results],
                k_list=k_list)
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def rag_qa_dpr(self,
               queries: List[str|QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None) -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        Performs retrieval-augmented generation enhanced QA using a standard DPR framework.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it returns answers only or additionally evaluate retrieval and answer quality using
        recall @ k, exact match and F1 score metrics.

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): A list of lists containing gold-standard documents for
                each query. This is used if document-level evaluation is to be performed. Default is None.
            gold_answers (Optional[List[List[str]]]): A list of lists containing gold-standard answers for
                each query. Required if evaluation of question answering (QA) answers is enabled. Default
                is None.

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: A tuple that always includes:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
                If evaluation is enabled, the tuple also includes:
                - A dictionary with overall results from the retrieval phase (if applicable).
                - A dictionary with overall QA evaluation metrics (exact match and F1 scores).

        """
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        # Retrieving (if necessary)
        overall_retrieval_result = None

        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve_dpr(queries=queries, gold_docs=gold_docs)
            else:
                queries = self.retrieve_dpr(queries=queries)

        # Performing QA
        queries_solutions, all_response_message, all_metadata = self.qa(queries)

        # Evaluating QA
        if gold_answers is not None:
            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)

            # round off to 4 decimal places for QA results
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # Save retrieval and QA results
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results
        else:
            return queries_solutions, all_response_message, all_metadata

    def qa(self, queries: List[QuerySolution]):
        all_qa_messages = []
        bad_cases = []  # 用于记录被拦截的样本

        for idx, query_solution in enumerate(tqdm(queries, desc="Collecting QA prompts")):
            retrieved_passages = query_solution.docs[:self.global_config.qa_top_k]

            prompt_user = ''
            for passage in retrieved_passages:
                prompt_user += f'Wikipedia Title: {passage}\n\n'
            prompt_user += 'Question: ' + query_solution.question + '\nThought: '

            if self.prompt_template_manager.is_template_name_valid(name=f'rag_qa_{self.global_config.dataset}'):
                prompt_dataset_name = self.global_config.dataset
            else:
                logger.debug(
                    f"rag_qa_{self.global_config.dataset} does not have a customized prompt template. Using MUSIQUE's prompt template instead."
                )
                prompt_dataset_name = 'musique'

            msg = self.prompt_template_manager.render(
                name=f'rag_qa_{prompt_dataset_name}', prompt_user=prompt_user
            )
            all_qa_messages.append(msg)

        # 逐条推理，避免一条拦截导致整体中断
        all_response_message = []
        all_metadata = []
        all_cache_hit = []

        for i, qa_messages in enumerate(tqdm(all_qa_messages, desc="QA Reading")):
            try:
                resp_msg, meta, cache_hit = self.llm_model.infer(qa_messages)
            except BadRequestError as e:
                # 关键：400 内容风险不重试，记录并跳过
                logger.warning(f"[QA] BadRequestError at idx={i}: {str(e)}")

                # 记录触发样本（注意别把超长内容全塞日志，写文件更好）
                q = queries[i].question if i < len(queries) else None
                docs = queries[i].docs[:self.global_config.qa_top_k] if i < len(queries) else None
                bad_cases.append({
                    "idx": i,
                    "question": q,
                    "docs_topk": docs,
                    "error": str(e),
                })

                # 给一个“占位返回”，保证后续解析不崩
                resp_msg, meta, cache_hit = "", {"skipped": True, "reason": "BadRequestError_ContentRisk"}, False
            except Exception as e:
                # 其他异常你也可以选择跳过，避免跑一半断掉
                logger.exception(f"[QA] Unexpected error at idx={i}: {str(e)}")
                bad_cases.append({
                    "idx": i,
                    "question": queries[i].question if i < len(queries) else None,
                    "docs_topk": queries[i].docs[:self.global_config.qa_top_k] if i < len(queries) else None,
                    "error": str(e),
                })
                resp_msg, meta, cache_hit = "", {"skipped": True, "reason": "UnexpectedError"}, False

            all_response_message.append(resp_msg)
            all_metadata.append(meta)
            all_cache_hit.append(cache_hit)

        # 把 bad cases 落盘，方便你回查是哪条触发
        if len(bad_cases) > 0:
            save_dir = getattr(self.global_config, "output_dir", ".")
            os.makedirs(save_dir, exist_ok=True)
            bad_path = os.path.join(save_dir, "qa_bad_cases.jsonl")
            with open(bad_path, "a", encoding="utf-8") as f:
                for item in bad_cases:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.warning(f"[QA] Saved {len(bad_cases)} bad cases to: {bad_path}")

        # 解析答案
        queries_solutions = []
        for query_solution_idx, query_solution in tqdm(enumerate(queries), desc="Extraction Answers from LLM Response"):
            response_content = all_response_message[query_solution_idx] or ""
            try:
                pred_ans = response_content.split('Answer:')[1].strip()
            except Exception as e:
                logger.warning(f"Error in parsing the answer from the raw LLM QA inference response: {str(e)}!")
                pred_ans = response_content  # 空字符串也OK

            query_solution.answer = pred_ans
            queries_solutions.append(query_solution)

        return queries_solutions, all_response_message, all_metadata

    def add_fact_edges(self, chunk_ids: List[str], chunk_triples: List[Tuple]):
        """
        Adds fact edges from given triples to the graph.

        The method processes chunks of triples, computes unique identifiers
        for entities and relations, and updates various internal statistics
        to build and maintain the graph structure. Entities are uniquely
        identified and linked based on their relationships.

        Parameters:
            chunk_ids: List[str]
                A list of unique identifiers for the chunks being processed.
            chunk_triples: List[Tuple]
                A list of tuples representing triples to process. Each triple
                consists of a subject, predicate, and object.

        Raises:
            Does not explicitly raise exceptions within the provided function logic.
        """

        if "name" in self.graph.vs:
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        logger.info(f"Adding OpenIE triples to graph.")

        for chunk_key, triples in tqdm(zip(chunk_ids, chunk_triples)):
            entities_in_chunk = set()

            if chunk_key not in current_graph_nodes:
                for triple in triples:
                    triple = tuple(triple)

                    node_key = compute_mdhash_id(content=triple[0], prefix=("entity-"))
                    node_2_key = compute_mdhash_id(content=triple[2], prefix=("entity-"))

                    self.node_to_node_stats[(node_key, node_2_key)] = self.node_to_node_stats.get(
                        (node_key, node_2_key), 0.0) + 1
                    self.node_to_node_stats[(node_2_key, node_key)] = self.node_to_node_stats.get(
                        (node_2_key, node_key), 0.0) + 1

                    entities_in_chunk.add(node_key)
                    entities_in_chunk.add(node_2_key)

                for node in entities_in_chunk:
                    self.ent_node_to_chunk_ids[node] = self.ent_node_to_chunk_ids.get(node, set()).union(set([chunk_key]))

    def add_passage_edges(self, chunk_ids: List[str], chunk_triple_entities: List[List[str]]):
        """
        Adds edges connecting passage nodes to phrase nodes in the graph.

        This method is responsible for iterating through a list of chunk identifiers
        and their corresponding triple entities. It calculates and adds new edges
        between the passage nodes (defined by the chunk identifiers) and the phrase
        nodes (defined by the computed unique hash IDs of triple entities). The method
        also updates the node-to-node statistics map and keeps count of newly added
        passage nodes.

        Parameters:
            chunk_ids : List[str]
                A list of identifiers representing passage nodes in the graph.
            chunk_triple_entities : List[List[str]]
                A list of lists where each sublist contains entities (strings) associated
                with the corresponding chunk in the chunk_ids list.

        Returns:
            int
                The number of new passage nodes added to the graph.
        """

        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        num_new_chunks = 0

        logger.info(f"Connecting passage nodes to phrase nodes.")

        for idx, chunk_key in tqdm(enumerate(chunk_ids)):

            if chunk_key not in current_graph_nodes:
                for chunk_ent in chunk_triple_entities[idx]:
                    node_key = compute_mdhash_id(chunk_ent, prefix="entity-")

                    self.node_to_node_stats[(chunk_key, node_key)] = 1.0

                num_new_chunks += 1

        return num_new_chunks

    def add_synonymy_edges(self):
        """
        Adds synonymy edges between similar nodes in the graph to enhance connectivity by identifying and linking synonym entities.

        This method performs key operations to compute and add synonymy edges. It first retrieves embeddings for all nodes, then conducts
        a nearest neighbor (KNN) search to find similar nodes. These similar nodes are identified based on a score threshold, and edges
        are added to represent the synonym relationship.

        Attributes:
            entity_id_to_row: dict (populated within the function). Maps each entity ID to its corresponding row data, where rows
                              contain `content` of entities used for comparison.
            entity_embedding_store: Manages retrieval of texts and embeddings for all rows related to entities.
            global_config: Configuration object that defines parameters such as `synonymy_edge_topk`, `synonymy_edge_sim_threshold`,
                           `synonymy_edge_query_batch_size`, and `synonymy_edge_key_batch_size`.
            node_to_node_stats: dict. Stores scores for edges between nodes representing their relationship.

        """
        logger.info(f"Expanding graph with synonymy edges")

        self.entity_id_to_row = self.entity_embedding_store.get_all_id_to_rows()
        entity_node_keys = list(self.entity_id_to_row.keys())

        logger.info(f"Performing KNN retrieval for each phrase nodes ({len(entity_node_keys)}).")

        entity_embs = self.entity_embedding_store.get_embeddings(entity_node_keys)

        # Here we build synonymy edges only between newly inserted phrase nodes and all phrase nodes in the storage to reduce cost for incremental graph updates
        query_node_key2knn_node_keys = retrieve_knn(query_ids=entity_node_keys,
                                                    key_ids=entity_node_keys,
                                                    query_vecs=entity_embs,
                                                    key_vecs=entity_embs,
                                                    k=self.global_config.synonymy_edge_topk,
                                                    query_batch_size=self.global_config.synonymy_edge_query_batch_size,
                                                    key_batch_size=self.global_config.synonymy_edge_key_batch_size)

        num_synonym_triple = 0
        synonym_candidates = []  # [(node key, [(synonym node key, corresponding score), ...]), ...]

        for node_key in tqdm(query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys)):
            synonyms = []

            entity = self.entity_id_to_row[node_key]["content"]

            if len(re.sub('[^A-Za-z0-9]', '', entity)) > 2:
                nns = query_node_key2knn_node_keys[node_key]

                num_nns = 0
                for nn, score in zip(nns[0], nns[1]):
                    if score < self.global_config.synonymy_edge_sim_threshold or num_nns > 100:
                        break

                    nn_phrase = self.entity_id_to_row[nn]["content"]

                    if nn != node_key and nn_phrase != '':
                        sim_edge = (node_key, nn)
                        synonyms.append((nn, score))
                        num_synonym_triple += 1

                        self.node_to_node_stats[sim_edge] = score  # Need to seriously discuss on this
                        num_nns += 1

            synonym_candidates.append((node_key, synonyms))

    def load_existing_openie(self, chunk_keys: List[str]) -> Tuple[List[dict], Set[str]]:
        """
        Loads existing OpenIE results from the specified file if it exists and combines
        them with new content while standardizing indices. If the file does not exist or
        is configured to be re-initialized from scratch with the flag `force_openie_from_scratch`,
        it prepares new entries for processing.

        Args:
            chunk_keys (List[str]): A list of chunk keys that represent identifiers
                                     for the content to be processed.

        Returns:
            Tuple[List[dict], Set[str]]: A tuple where the first element is the existing OpenIE
                                         information (if any) loaded from the file, and the
                                         second element is a set of chunk keys that still need to
                                         be saved or processed.
        """

        # combine openie_results with contents already in file, if file exists
        chunk_keys_to_save = set()

        if not self.global_config.force_openie_from_scratch and os.path.isfile(self.openie_results_path):
            openie_results = json.load(open(self.openie_results_path))
            all_openie_info = openie_results.get('docs', [])

            #Standardizing indices for OpenIE Files.

            renamed_openie_info = []
            for openie_info in all_openie_info:
                openie_info['idx'] = compute_mdhash_id(openie_info['passage'], 'chunk-')
                renamed_openie_info.append(openie_info)

            all_openie_info = renamed_openie_info

            existing_openie_keys = set([info['idx'] for info in all_openie_info])

            for chunk_key in chunk_keys:
                if chunk_key not in existing_openie_keys:
                    chunk_keys_to_save.add(chunk_key)
        else:
            all_openie_info = []
            chunk_keys_to_save = chunk_keys

        return all_openie_info, chunk_keys_to_save

    def merge_openie_results(self,
                             all_openie_info: List[dict],
                             chunks_to_save: Dict[str, dict],
                             ner_results_dict: Dict[str, NerRawOutput],
                             triple_results_dict: Dict[str, TripleRawOutput]) -> List[dict]:
        """
        Merges OpenIE extraction results with corresponding passage and metadata.

        This function integrates the OpenIE extraction results, including named-entity
        recognition (NER) entities and triples, with their respective text passages
        using the provided chunk keys. The resulting merged data is appended to
        the `all_openie_info` list containing dictionaries with combined and organized
        data for further processing or storage.

        Parameters:
            all_openie_info (List[dict]): A list to hold dictionaries of merged OpenIE
                results and metadata for all chunks.
            chunks_to_save (Dict[str, dict]): A dict of chunk identifiers (keys) to process
                and merge OpenIE results to dictionaries with `hash_id` and `content` keys.
            ner_results_dict (Dict[str, NerRawOutput]): A dictionary mapping chunk keys
                to their corresponding NER extraction results.
            triple_results_dict (Dict[str, TripleRawOutput]): A dictionary mapping chunk
                keys to their corresponding OpenIE triple extraction results.

        Returns:
            List[dict]: The `all_openie_info` list containing dictionaries with merged
            OpenIE results, metadata, and the passage content for each chunk.

        """

        for chunk_key, row in chunks_to_save.items():
            passage = row['content']
            try:
                chunk_openie_info = {'idx': chunk_key, 'passage': passage,
                                 'extracted_entities': ner_results_dict[chunk_key].unique_entities,
                                 'extracted_triples': triple_results_dict[chunk_key].triples}
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_key}: {e}")
                chunk_openie_info = {'idx': chunk_key, 'passage': passage,
                                 'extracted_entities': [],
                                 'extracted_triples': []}
            all_openie_info.append(chunk_openie_info)

        return all_openie_info

    def save_openie_results(self, all_openie_info: List[dict]):
        """
        Computes statistics on extracted entities from OpenIE results and saves the aggregated data in a
        JSON file. The function calculates the average character and word lengths of the extracted entities
        and writes them along with the provided OpenIE information to a file.

        Parameters:
            all_openie_info : List[dict]
                List of dictionaries, where each dictionary represents information from OpenIE, including
                extracted entities.
        """

        sum_phrase_chars = sum([len(e) for chunk in all_openie_info for e in chunk['extracted_entities']])
        sum_phrase_words = sum([len(e.split()) for chunk in all_openie_info for e in chunk['extracted_entities']])
        num_phrases = sum([len(chunk['extracted_entities']) for chunk in all_openie_info])

        if len(all_openie_info) > 0:
            # Avoid division by zero if there are no phrases
            if num_phrases > 0:
                avg_ent_chars = round(sum_phrase_chars / num_phrases, 4)
                avg_ent_words = round(sum_phrase_words / num_phrases, 4)
            else:
                avg_ent_chars = 0
                avg_ent_words = 0
                
            openie_dict = {
                'docs': all_openie_info,
                'avg_ent_chars': avg_ent_chars,
                'avg_ent_words': avg_ent_words
            }
            
            with open(self.openie_results_path, 'w') as f:
                json.dump(openie_dict, f)
            logger.info(f"OpenIE results saved to {self.openie_results_path}")

    def augment_graph(self):
        """
        Provides utility functions to augment a graph by adding new nodes and edges.
        It ensures that the graph structure is extended to include additional components,
        and logs the completion status along with printing the updated graph information.
        """

        self.add_new_nodes()
        self.add_new_edges()

        logger.info(f"Graph construction completed!")
        print(self.get_graph_info())

    def add_new_nodes(self):
        """
        Adds new nodes to the graph from entity and passage embedding stores based on their attributes.

        This method identifies and adds new nodes to the graph by comparing existing nodes
        in the graph and nodes retrieved from the entity embedding store and the passage
        embedding store. The method checks attributes and ensures no duplicates are added.
        New nodes are prepared and added in bulk to optimize graph updates.
        """

        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}

        entity_to_row = self.entity_embedding_store.get_all_id_to_rows()
        passage_to_row = self.chunk_embedding_store.get_all_id_to_rows()

        node_to_rows = entity_to_row
        node_to_rows.update(passage_to_row)

        new_nodes = {}
        for node_id, node in node_to_rows.items():
            node['name'] = node_id
            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        if len(new_nodes) > 0:
            self.graph.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)

    def add_new_edges(self):
        """
        Processes edges from `node_to_node_stats` to add them into a graph object while
        managing adjacency lists, validating edges, and logging invalid edge cases.
        """

        graph_adj_list = defaultdict(dict)
        graph_inverse_adj_list = defaultdict(dict)
        edge_source_node_keys = []
        edge_target_node_keys = []
        edge_metadata = []
        for edge, weight in self.node_to_node_stats.items():
            if edge[0] == edge[1]: continue
            graph_adj_list[edge[0]][edge[1]] = weight
            graph_inverse_adj_list[edge[1]][edge[0]] = weight

            edge_source_node_keys.append(edge[0])
            edge_target_node_keys.append(edge[1])
            edge_metadata.append({
                "weight": weight
            })

        valid_edges, valid_weights = [], {"weight": []}
        current_node_ids = set(self.graph.vs["name"])
        for source_node_id, target_node_id, edge_d in zip(edge_source_node_keys, edge_target_node_keys, edge_metadata):
            if source_node_id in current_node_ids and target_node_id in current_node_ids:
                valid_edges.append((source_node_id, target_node_id))
                weight = edge_d.get("weight", 1.0)
                valid_weights["weight"].append(weight)
            else:
                logger.warning(f"Edge {source_node_id} -> {target_node_id} is not valid.")
        self.graph.add_edges(
            valid_edges,
            attributes=valid_weights
        )

    def save_igraph(self):
        logger.info(
            f"Writing graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges"
        )
        self.graph.write_pickle(self._graph_pickle_filename)
        logger.info(f"Saving graph completed!")

    def get_graph_info(self) -> Dict:
        """
        Obtains detailed information about the graph such as the number of nodes,
        triples, and their classifications.

        This method calculates various statistics about the graph based on the
        stores and node-to-node relationships, including counts of phrase and
        passage nodes, total nodes, extracted triples, triples involving passage
        nodes, synonymy triples, and total triples.

        Returns:
            Dict
                A dictionary containing the following keys and their respective values:
                - num_phrase_nodes: The number of unique phrase nodes.
                - num_passage_nodes: The number of unique passage nodes.
                - num_total_nodes: The total number of nodes (sum of phrase and passage nodes).
                - num_extracted_triples: The number of unique extracted triples.
                - num_triples_with_passage_node: The number of triples involving at least one
                  passage node.
                - num_synonymy_triples: The number of synonymy triples (distinct from extracted
                  triples and those with passage nodes).
                - num_total_triples: The total number of triples.
        """
        graph_info = {}

        # get # of phrase nodes
        phrase_nodes_keys = self.entity_embedding_store.get_all_ids()
        graph_info["num_phrase_nodes"] = len(set(phrase_nodes_keys))

        # get # of passage nodes
        passage_nodes_keys = self.chunk_embedding_store.get_all_ids()
        graph_info["num_passage_nodes"] = len(set(passage_nodes_keys))

        # get # of total nodes
        graph_info["num_total_nodes"] = graph_info["num_phrase_nodes"] + graph_info["num_passage_nodes"]

        # get # of extracted triples
        graph_info["num_extracted_triples"] = len(self.fact_embedding_store.get_all_ids())

        num_triples_with_passage_node = 0
        passage_nodes_set = set(passage_nodes_keys)
        num_triples_with_passage_node = sum(
            1 for node_pair in self.node_to_node_stats
            if node_pair[0] in passage_nodes_set or node_pair[1] in passage_nodes_set
        )
        graph_info['num_triples_with_passage_node'] = num_triples_with_passage_node

        graph_info['num_synonymy_triples'] = len(self.node_to_node_stats) - graph_info[
            "num_extracted_triples"] - num_triples_with_passage_node

        # get # of total triples
        graph_info["num_total_triples"] = len(self.node_to_node_stats)

        return graph_info

    def prepare_retrieval_objects(self):
        """
        Prepares various in-memory objects and attributes necessary for fast retrieval processes, such as embedding data and graph relationships, ensuring consistency
        and alignment with the underlying graph structure.
        """

        logger.info("Preparing for fast retrieval.")

        logger.info("Loading keys.")
        self.query_to_embedding: Dict = {'triple': {}, 'passage': {}}

        self.entity_node_keys: List = list(self.entity_embedding_store.get_all_ids()) # a list of phrase node keys
        self.passage_node_keys: List = list(self.chunk_embedding_store.get_all_ids()) # a list of passage node keys
        self.fact_node_keys: List = list(self.fact_embedding_store.get_all_ids())

        # Check if the graph has the expected number of nodes
        expected_node_count = len(self.entity_node_keys) + len(self.passage_node_keys)
        actual_node_count = self.graph.vcount()
        
        if expected_node_count != actual_node_count:
            logger.warning(f"Graph node count mismatch: expected {expected_node_count}, got {actual_node_count}")
            # If the graph is empty but we have nodes, we need to add them
            if actual_node_count == 0 and expected_node_count > 0:
                logger.info(f"Initializing graph with {expected_node_count} nodes")
                self.add_new_nodes()
                self.save_igraph()

        # Create mapping from node name to vertex index
        try:
            igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)} # from node key to the index in the backbone graph
            self.node_name_to_vertex_idx = igraph_name_to_idx
            
            # Check if all entity and passage nodes are in the graph
            missing_entity_nodes = [node_key for node_key in self.entity_node_keys if node_key not in igraph_name_to_idx]
            missing_passage_nodes = [node_key for node_key in self.passage_node_keys if node_key not in igraph_name_to_idx]
            
            if missing_entity_nodes or missing_passage_nodes:
                logger.warning(f"Missing nodes in graph: {len(missing_entity_nodes)} entity nodes, {len(missing_passage_nodes)} passage nodes")
                # If nodes are missing, rebuild the graph
                self.add_new_nodes()
                self.save_igraph()
                # Update the mapping
                igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)}
                self.node_name_to_vertex_idx = igraph_name_to_idx
            
            self.entity_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.entity_node_keys] # a list of backbone graph node index
            self.passage_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.passage_node_keys] # a list of backbone passage node index
        except Exception as e:
            logger.error(f"Error creating node index mapping: {str(e)}")
            # Initialize with empty lists if mapping fails
            self.node_name_to_vertex_idx = {}
            self.entity_node_idxs = []
            self.passage_node_idxs = []

        logger.info("Loading embeddings.")
        self.entity_embeddings = np.array(self.entity_embedding_store.get_embeddings(self.entity_node_keys))
        self.passage_embeddings = np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys))

        self.fact_embeddings = np.array(self.fact_embedding_store.get_embeddings(self.fact_node_keys))

        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])

        self.proc_triples_to_docs = {}

        for doc in all_openie_info:
            triples = flatten_facts([doc['extracted_triples']])
            for triple in triples:
                if len(triple) == 3:
                    proc_triple = tuple(text_processing(list(triple)))
                    self.proc_triples_to_docs[str(proc_triple)] = self.proc_triples_to_docs.get(str(proc_triple), set()).union(set([doc['idx']]))

        if self.ent_node_to_chunk_ids is None:
            ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

            # Check if the lengths match
            if not (len(self.passage_node_keys) == len(ner_results_dict) == len(triple_results_dict)):
                logger.warning(f"Length mismatch: passage_node_keys={len(self.passage_node_keys)}, ner_results_dict={len(ner_results_dict)}, triple_results_dict={len(triple_results_dict)}")
                
                # If there are missing keys, create empty entries for them
                for chunk_id in self.passage_node_keys:
                    if chunk_id not in ner_results_dict:
                        ner_results_dict[chunk_id] = NerRawOutput(
                            chunk_id=chunk_id,
                            response=None,
                            metadata={},
                            unique_entities=[]
                        )
                    if chunk_id not in triple_results_dict:
                        triple_results_dict[chunk_id] = TripleRawOutput(
                            chunk_id=chunk_id,
                            response=None,
                            metadata={},
                            triples=[]
                        )

            # prepare data_store
            chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in self.passage_node_keys]

            self.node_to_node_stats = {}
            self.ent_node_to_chunk_ids = {}
            self.add_fact_edges(self.passage_node_keys, chunk_triples)

        self.ready_to_retrieve = True

    def get_query_embeddings(self, queries: List[str] | List[QuerySolution]):
        """
        Retrieves embeddings for given queries and updates the internal query-to-embedding mapping. The method determines whether each query
        is already present in the `self.query_to_embedding` dictionary under the keys 'triple' and 'passage'. If a query is not present in
        either, it is encoded into embeddings using the embedding model and stored.

        Args:
            queries List[str] | List[QuerySolution]: A list of query strings or QuerySolution objects. Each query is checked for
            its presence in the query-to-embedding mappings.
        """

        all_query_strings = []
        for query in queries:
            if isinstance(query, QuerySolution) and (
                    query.question not in self.query_to_embedding['triple'] or query.question not in
                    self.query_to_embedding['passage']):
                all_query_strings.append(query.question)
            elif query not in self.query_to_embedding['triple'] or query not in self.query_to_embedding['passage']:
                all_query_strings.append(query)

        if len(all_query_strings) > 0:
            # get all query embeddings
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_fact.")
            query_embeddings_for_triple = self.embedding_model.batch_encode(all_query_strings,
                                                                            instruction=get_query_instruction('query_to_fact'),
                                                                            norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_triple):
                self.query_to_embedding['triple'][query] = embedding

            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_passage.")
            query_embeddings_for_passage = self.embedding_model.batch_encode(all_query_strings,
                                                                             instruction=get_query_instruction('query_to_passage'),
                                                                             norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                self.query_to_embedding['passage'][query] = embedding

    def get_fact_scores(self, query: str) -> np.ndarray:
        """
        Retrieves and computes normalized similarity scores between the given query and pre-stored fact embeddings.

        Parameters:
        query : str
            The input query text for which similarity scores with fact embeddings
            need to be computed.

        Returns:
        numpy.ndarray
            A normalized array of similarity scores between the query and fact
            embeddings. The shape of the array is determined by the number of
            facts.

        Raises:
        KeyError
            If no embedding is found for the provided query in the stored query
            embeddings dictionary.
        """
        query_embedding = self.query_to_embedding['triple'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_fact'),
                                                                norm=True)

        # Check if there are any facts
        if len(self.fact_embeddings) == 0:
            logger.warning("No facts available for scoring. Returning empty array.")
            return np.array([])
            
        try:
            query_fact_scores = np.dot(self.fact_embeddings, query_embedding.T) # shape: (#facts, )
            query_fact_scores = np.squeeze(query_fact_scores) if query_fact_scores.ndim == 2 else query_fact_scores
            query_fact_scores = min_max_normalize(query_fact_scores)
            return query_fact_scores
        except Exception as e:
            logger.error(f"Error computing fact scores: {str(e)}")
            return np.array([])

    def dense_passage_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Conduct dense passage retrieval to find relevant documents for a query.

        This function processes a given query using a pre-trained embedding model
        to generate query embeddings. The similarity scores between the query
        embedding and passage embeddings are computed using dot product, followed
        by score normalization. Finally, the function ranks the documents based
        on their similarity scores and returns the ranked document identifiers
        and their scores.

        Parameters
        ----------
        query : str
            The input query for which relevant passages should be retrieved.

        Returns
        -------
        tuple : Tuple[np.ndarray, np.ndarray]
            A tuple containing two elements:
            - A list of sorted document identifiers based on their relevance scores.
            - A numpy array of the normalized similarity scores for the corresponding
              documents.
        """
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_passage'),
                                                                norm=True)
        query_doc_scores = np.dot(self.passage_embeddings, query_embedding.T)
        query_doc_scores = np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores
        query_doc_scores = min_max_normalize(query_doc_scores)

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
        return sorted_doc_ids, sorted_doc_scores


    def get_top_k_weights(self,
                          link_top_k: int,
                          all_phrase_weights: np.ndarray,
                          linking_score_map: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        This function filters the all_phrase_weights to retain only the weights for the
        top-ranked phrases in terms of the linking_score_map. It also filters linking scores
        to retain only the top `link_top_k` ranked nodes. Non-selected phrases in phrase
        weights are reset to a weight of 0.0.

        Args:
            link_top_k (int): Number of top-ranked nodes to retain in the linking score map.
            all_phrase_weights (np.ndarray): An array representing the phrase weights, indexed
                by phrase ID.
            linking_score_map (Dict[str, float]): A mapping of phrase content to its linking
                score, sorted in descending order of scores.

        Returns:
            Tuple[np.ndarray, Dict[str, float]]: A tuple containing the filtered array
            of all_phrase_weights with unselected weights set to 0.0, and the filtered
            linking_score_map containing only the top `link_top_k` phrases.
        """
        # choose top ranked nodes in linking_score_map
        linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])

        # only keep the top_k phrases in all_phrase_weights
        top_k_phrases = set(linking_score_map.keys())
        top_k_phrases_keys = set(
            [compute_mdhash_id(content=top_k_phrase, prefix="entity-") for top_k_phrase in top_k_phrases])

        for phrase_key in self.node_name_to_vertex_idx:
            if phrase_key not in top_k_phrases_keys:
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
                if phrase_id is not None:
                    all_phrase_weights[phrase_id] = 0.0

        assert np.count_nonzero(all_phrase_weights) == len(linking_score_map.keys())
        return all_phrase_weights, linking_score_map

    def graph_search_with_fact_entities(self, query: str,
                                        link_top_k: int,
                                        query_fact_scores: np.ndarray,
                                        top_k_facts: List[Tuple],
                                        top_k_fact_indices: List[str],
                                        passage_node_weight: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes document scores based on fact-based similarity and relevance using personalized
        PageRank (PPR) and dense retrieval models. This function combines the signal from the relevant
        facts identified with passage similarity and graph-based search for enhanced result ranking.

        Parameters:
            query (str): The input query string for which similarity and relevance computations
                need to be performed.
            link_top_k (int): The number of top phrases to include from the linking score map for
                downstream processing.
            query_fact_scores (np.ndarray): An array of scores representing fact-query similarity
                for each of the provided facts.
            top_k_facts (List[Tuple]): A list of top-ranked facts, where each fact is represented
                as a tuple of its subject, predicate, and object.
            top_k_fact_indices (List[str]): Corresponding indices or identifiers for the top-ranked
                facts in the query_fact_scores array.
            passage_node_weight (float): Default weight to scale passage scores in the graph.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                - The first array corresponds to document IDs sorted based on their scores.
                - The second array consists of the PPR scores associated with the sorted document IDs.
        """

        #Assigning phrase weights based on selected facts from previous steps.
        linking_score_map = {}  # from phrase to the average scores of the facts that contain the phrase
        phrase_scores = {}  # store all fact scores for each phrase regardless of whether they exist in the knowledge graph or not
        phrase_weights = np.zeros(len(self.graph.vs['name']))
        passage_weights = np.zeros(len(self.graph.vs['name']))
        number_of_occurs = np.zeros(len(self.graph.vs['name']))

        phrases_and_ids = set()

        for rank, f in enumerate(top_k_facts):
            subject_phrase = f[0].lower()
            predicate_phrase = f[1].lower()
            object_phrase = f[2].lower()
            fact_score = query_fact_scores[
                top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores

            for phrase in [subject_phrase, object_phrase]:
                phrase_key = compute_mdhash_id(
                    content=phrase,
                    prefix="entity-"
                )
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)

                if phrase_id is not None:
                    weighted_fact_score = fact_score

                    if len(self.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                        weighted_fact_score /= len(self.ent_node_to_chunk_ids[phrase_key]) # “罕见实体更重要”的 IDF-ish 修正

                    phrase_weights[phrase_id] += weighted_fact_score
                    number_of_occurs[phrase_id] += 1

                phrases_and_ids.add((phrase, phrase_id))

        phrase_weights /= number_of_occurs

        for phrase, phrase_id in phrases_and_ids:
            if phrase not in phrase_scores:
                phrase_scores[phrase] = []

            phrase_scores[phrase].append(phrase_weights[phrase_id])

        # calculate average fact score for each phrase
        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        if link_top_k:
            phrase_weights, linking_score_map = self.get_top_k_weights(link_top_k,
                                                                           phrase_weights,
                                                                           linking_score_map)  # at this stage, the length of linking_scope_map is determined by link_top_k

        #Get passage scores according to chosen dense retrieval model
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
            passage_dpr_score = normalized_dpr_sorted_scores[i]
            passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
            passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
            passage_node_text = self.chunk_embedding_store.get_row(passage_node_key)["content"]
            linking_score_map[passage_node_text] = passage_dpr_score * passage_node_weight

        #Combining phrase and passage scores into one array for PPR
        node_weights = phrase_weights + passage_weights

        #Recording top 30 facts in linking_score_map
        if len(linking_score_map) > 30:
            linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30])

        assert sum(node_weights) > 0, f'No phrases found in the graph for the given facts: {top_k_facts}'

        #Running PPR algorithm based on the passage and phrase weights previously assigned
        ppr_start = time.time()
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(node_weights, damping=self.global_config.damping)
        ppr_end = time.time()

        self.ppr_time += (ppr_end - ppr_start)

        assert len(ppr_sorted_doc_ids) == len(
            self.passage_node_idxs), f"Doc prob length {len(ppr_sorted_doc_ids)} != corpus length {len(self.passage_node_idxs)}"

        return ppr_sorted_doc_ids, ppr_sorted_doc_scores


    def rerank_facts(
        self,
        query: str,
        query_fact_scores: np.ndarray,
        instruction: str | None = None,   # ✅ NEW
    ) -> Tuple[List[int], List[Tuple], dict]:
        """
        Returns:
            top_k_fact_indices: indices in fact_node_keys
            top_k_facts: list of triples (Tuple)
            rerank_log: {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
        """
        link_top_k: int = self.global_config.linking_top_k

        if query_fact_scores is None or len(query_fact_scores) == 0 or len(self.fact_node_keys) == 0:
            logger.warning("No facts available for reranking. Returning empty lists.")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': []}

        try:
            # top-k candidate by score
            if len(query_fact_scores) <= link_top_k:
                candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
            else:
                candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()

            real_candidate_fact_ids = [self.fact_node_keys[idx] for idx in candidate_fact_indices]
            fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)
            candidate_facts = [eval(fact_row_dict[_id]["content"]) for _id in real_candidate_fact_ids]

            # ✅ rerank by LLM (instruction 透传)
            top_k_fact_indices, top_k_facts, reranker_dict = self.rerank_filter.rerank(
                query=query,
                candidate_items=candidate_facts,
                candidate_indices=candidate_fact_indices,
                len_after_rerank=link_top_k,
                instruction=instruction,   # ✅ NEW
            )

            rerank_log = {
                "facts_before_rerank": candidate_facts,
                "facts_after_rerank": top_k_facts,
            }
            # 可选：把 reranker_dict 也塞进 log，便于 debug
            if isinstance(reranker_dict, dict):
                rerank_log.update({f"reranker_{k}": v for k, v in reranker_dict.items()})

            return top_k_fact_indices, top_k_facts, rerank_log

        except Exception as e:
            logger.error(f"Error in rerank_facts: {str(e)}")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': [], 'error': str(e)}  

    def run_ppr(self,
                reset_prob: np.ndarray,
                damping: float =0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs Personalized PageRank (PPR) on a graph and computes relevance scores for
        nodes corresponding to document passages. The method utilizes a damping
        factor for teleportation during rank computation and can take a reset
        probability array to influence the starting state of the computation.

        Parameters:
            reset_prob (np.ndarray): A 1-dimensional array specifying the reset
                probability distribution for each node. The array must have a size
                equal to the number of nodes in the graph. NaNs or negative values
                within the array are replaced with zeros.
            damping (float): A scalar specifying the damping factor for the
                computation. Defaults to 0.5 if not provided or set to `None`.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays. The
                first array represents the sorted node IDs of document passages based
                on their relevance scores in descending order. The second array
                contains the corresponding relevance scores of each document passage
                in the same order.
        """

        if damping is None: damping = 0.5 # for potential compatibility
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )

        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores
    
    def rag_retrieval_eval(self,
                        queries: List[str],
                        gold_docs: List[List[str]],
                        num_to_retrieve: int = None,
                        k_list: List[int] = None,
                        return_retrieval_results: bool = False
                        ) -> Dict | Tuple[List[QuerySolution], Dict]:
        """
        Retrieval-only pipeline: retrieve -> compute Recall@k.
        No QA / no LLM API.
        """
        if k_list is None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]

        retrieval_results, overall_retrieval_result = self.retrieve(
            queries=queries,
            num_to_retrieve=num_to_retrieve,
            gold_docs=gold_docs
        )
        overall_retrieval_result = {k: round(float(v), 4) for k, v in overall_retrieval_result.items()}
        logger.info(f"Retrieval-only evaluation (Recall@k): {overall_retrieval_result}")

        return (retrieval_results, overall_retrieval_result) if return_retrieval_results else overall_retrieval_result