# Thesis experiment suite

The opt-in entry point is `scripts/run_thesis_experiments.py`. Production retrieval defaults are unchanged. The only change to the production judge is recording provider usage and request-attempt metadata; neither is used in its decisions.

## Frozen protocol

- The existing 20 development questions are excluded. Main question indices are `[20,1000)` for MuSiQue, 2WikiMultiHopQA and HotpotQA, in that order. Historical use of the fixed dataset must still be disclosed; this is not an official unseen test split.
- Each question's first round, judge, second-round filtered facts, fact scores and graph request are frozen on disk. Gold labels are read only by the downstream report function, never by retrieval or QA.
- Existing corpus, graph and three embedding stores are required. No `index()` call is made. The script verifies cached passage IDs against the corpus and checks embedding size/mtime before and after.
- All methods use the same original SEB first round, subject cap 4, Top-5 and reader. The graph retrieval limit remains 200. API model aliases are `deepseek-chat` and `deepseek-reasoner`, with their response metadata and run dates retained.

## Methods

| ID | Configuration | Scope |
| --- | --- | --- |
| round1 | Frozen first-round result, no repair | 980 questions per dataset |
| ppr | Original second-round PPR | Same 980 |
| rerank | PPR top 30 candidates, each at most 4000 characters, one LLM passage selection | Same 980 |
| static | Fixed breadth-first exploration, up to two fixed source inspections, one LLM path-set selection | Same 980 |
| agent | Original adaptive Agent, PPR/DPR fallback | Same 980 |
| agent_no_fallback | Same Agent attempt, retain round1 on failure | Same 980; counterfactual diagnostic |
| agent_unprotected_merge | Same committed paths, re-fuse without protecting newly selected path sources | Same 980; offline fusion diagnostic, not a new search ablation |
| agent_mask_predicate | Hide structured predicate/raw-triple predicate fields, retain source text semantics | Fixed first 200 main questions |
| agent_steps4 | At most four Agent rounds | Same first 200 |
| agent_steps12 | At most twelve Agent rounds | Same first 200 |

Controls are injected only within the sequential experiment call using a scoped dispatcher patch. The static control uses the same graph tools and capacity validator; unknown choices fail closed and fall back to PPR. It is a bounded BFS control, not an official ToG reproduction. All failures remain in the denominator.

The one-shot prompt explicitly separates `selectable_ids` from context-only IDs. Development replay exposed a protected passage outside PPR Top-30 being selected because the original instruction said only “listed IDs.” The v2 prompt removes that ambiguity without enlarging the candidate pool or silently accepting invalid selections. Validation failures retain their specific reason in the trace.

The baseline Agent budget remains 8 rounds, 16 tool calls, 64 returned edges, 8 neighbors, 4 hops, 60 seconds, 2 retries; cumulative token cutoff is disabled. The 4/12-round variants retain all other limits, so termination reasons must be reported when another limit binds first. One-shot controls use one call with a 1024-token output cap; no prose/fenced JSON or invented IDs are silently repaired.

## Execution and recovery

Without `--live`, the command prints the proposed manifest and does not invoke models. Credentials are inherited from the environment and never saved in the manifest.

```bash
python scripts/run_thesis_experiments.py --run-dir result_outputs/thesis-main --live
python scripts/run_thesis_experiments.py --run-dir result_outputs/thesis-main --live --resume
```

Launch the command inside tmux on the server. The supervisor loads only one dataset/model process at a time. Run and GPU locks prevent accidental duplicates. Results are atomically checkpointed per question and method. Resume rejects source/config/data fingerprint changes; changed code requires a new run directory. A failed judge stops the job instead of silently treating a provider error as a no-trigger decision. Existing completed records are not rerun.

Each dataset has `status.json`, `questions/`, `reports/` and `summary.json`. The run root has the manifest, supervisor status, dataset logs and cgroup/GPU samples every five seconds. Intermediate summaries update every 25 questions. At completion, report generation performs 10,000 paired bootstrap resamples for recall, full recall, EM and F1 on exactly matched question IDs.

## Interpretation

Shared-stage logical cost is not the experiment suite's actual bill. Raw model events and cache flags are retained. Judge provider usage is recorded; failures/SDK internal retries may have unavailable token counts. Stage latency is recorded with frozen/cache-reused inputs, not mislabeled as cold standalone end-to-end latency.

Every successful path receives structural/source checks against final Top-5. This does not establish semantic correctness. Manual source-support/relevance annotation and a fair external Agent-method adaptation remain separate tasks. Full Recall measures completeness of annotated support passages, not correctness of a model reasoning chain. The reader receives only passage text, never an extra path rendering.
