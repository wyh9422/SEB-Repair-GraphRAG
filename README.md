# SEB-Repair GraphRAG

<p align="center">
  <img src="assets/fig_method_overview.png" width="95%">
</p>

SEB-Repair GraphRAG is an enhanced implementation of the GraphRAG system for multi-hop QA, with a core focus on **Single Evidence Break (SEB):** Even in high recall settings, the inference chain may break due to the absence of a single crucial bridging piece of evidence. The project provides a basic workflow of **indexing/iterative retrieval/retrieval + question answering**, and supports the use of judge LLM to determine the answerability of the top-5 evidence and generate bridging questions, thereby triggering a second round of retrieval.

---

## Figures

<p align="center">
  <img src="assets/fig_intro.jpeg" width="95%">
</p>

<p align="center">
  <img src="assets/fig_seb_case.png" width="95%">
</p>

<p align="center">
  <img src="assets/fig_miss_dist.png" width="75%">
</p>

---

## Installation

```bash
conda create -n sebgraphrag python=3.10 -y
conda activate sebgraphrag
pip install -r requirements.txt
```

---

## Environment Variables

```bash
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# HuggingFace cache
export HF_HOME=<path to Huggingface home directory>
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

# DeepSeek（OpenAI-compatible）
export DEEPSEEK_BASE_URL=https://api.deepseek.com
export DEEPSEEK_MODEL=deepseek-reasoner
export DEEPSEEK_API_KEY=YOUR_KEY
```

---

## Data Format

`main.py` By default, it reads from the following path:

- Corpus：`reproduce/dataset/{dataset_name}_corpus.json`
- Question Set：`reproduce/dataset/{dataset_name}.json`

### Corpus JSON (`*_corpus.json`)

```json
[
  {"title": "Doc Title", "text": "Doc content", "idx": 0},
  {"title": "Another Title", "text": "Another content", "idx": 1}
]
```

---

## Judge Prompt

The judge prompt uses a JSON configuration file:：

`src/SRgraphrag/prompts/dspy_prompts/judge_prompt.json`

Example Structure：

```json
{
  "system_prompt": "...",
  "user_prompt_template": "...",
  "doc_max_chars": 200000,
  "query_max_chars": 2000
}
```
---

## Quick Start

### 1) Retrieval Only

```bash
python main.py   --dataset 2wikimultihopqa   --mode retrieve   --llm_base_url https://api.deepseek.com/v1   --llm_name deepseek-chat   --judge_llm_name deepseek-reasoner   --embedding_name nvidia/NV-Embed-v2   --save_dir outputs   --result_save_root result_outputs   --test_n 1000   --subject_level_Entity_Cap 4
```

### 2) Retrieval + QA

```bash
python main.py   --dataset 2wikimultihopqa   --mode qa   --llm_base_url https://api.deepseek.com/v1   --llm_name deepseek-chat   --judge_llm_name deepseek-reasoner   --embedding_name nvidia/NV-Embed-v2   --save_dir outputs   --test_n 1000   --subject_level_Entity_Cap 4
```

---

## Outputs

- `outputs/{dataset}/`: Index and runtime artifacts (controlled by `BaseConfig(save_dir=...)`)

- QA results (currently in main.py): `outputs/{dataset}/rag_qa_{dataset}.json`

- If `--result_save_root` is enabled: Retrieval evaluations and miss buckets will be written to this directory

---

