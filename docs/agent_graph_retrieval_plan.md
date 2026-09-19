# Agent 引导图路径检索修改计划

本文针对 `/Users/wyh/Workspace/SEB-Repair-GraphRAG` 的 `dev` 分支。目标是在现有 SEB 受控迭代检索上增加可验证、可重放的图路径探索能力，并能与原 PPR 检索公平比较。

本文保留最初确定的 P0—P4 方案与验收目标。后续已按此方案实现基线修复、检索契约、关系索引、Agent 控制器及第二轮接入，没有改成新的检索方案。**代码 MVP 与冒烟验证已完成，不代表完整数据集效果实验已经完成。** 具体文件、限制和实测结果见 [代码概览](codebase_overview.md) 和 [实现与验证记录](implementation_status.md)。下文“建议/待修改/验收”保留原计划语境，不作为当前完成状态清单。

## 当前代码基础

本次整理将原 `src/SRgraphrag/SRgraphrag.py` 中的以下逻辑移入独立模块，类方法和检索主链仍作为兼容入口，具体检查结果以本次交付记录为准：

| 模块 | 职责 |
| --- | --- |
| `retrieval/facts.py` | 主语数量限制提示词、主语计数 |
| `retrieval/evidence.py` | Top-5 证据注入、保护、去重和分数对齐 |
| `retrieval/metrics.py` | 原检索指标计算 |
| `retrieval/judge.py` | 原批量可回答性判断、桥接问题和证据解析；保留类方法代理 |
| `main.py` | 将模型等重依赖导入移至 CLI 参数解析后 |

索引由 `index` 完成；QA 的检索调用方向为 `rag_qa → retrieve → retrieve_full_once`，检索完成后再调用 `qa` 生成答案。单轮内部执行 DPR、事实过滤、PPR、证据保护；外层判断能否回答，并在需要时用桥接问题进行第二轮检索。

图数据有两层：

- `add_fact_edges`、`add_passage_edges`、`add_synonymy_edges` 建立用于扩散的加权图，`add_new_edges` 只存储边权重。该图没有完整的谓词、方向和篇章来源信息，不能直接作为事实路径验证依据。
- `merge_openie_results` 保存原始三元组和来源篇章，`prepare_retrieval_objects` 构建 `proc_triples_to_docs`。可从这些数据建立关系索引，保留现有 PPR 图。

审查时本地 `outputs/2wikimultihopqa/openie_results_ner_deepseek-chat.json` 有 6,119 篇文档、65,693 条原始三元组，其中 6,030 篇含三元组，可用于关系索引的离线验证。未发现可直接运行完整检索的 graph.pickle 和 embedding 成品；端到端验收依赖这些缓存或已配置的模型环境。

## 开发顺序与通过条件

| 阶段 | 交付目标 | 前置依赖 | 进入下一阶段的条件 |
| --- | --- | --- | --- |
| P0 | 修复并固定基线正确性 | 当前模块整理 | 离线边界测试通过，变更后的结果字段语义明确 |
| P1 | 统一请求、结果和 PPR 适配 | P0 | 新旧 PPR 路径对同一输入结果一致 |
| P2 | 有向关系索引和确定性图工具 | P1 的数据契约 | 路径、来源、方向、预算可由离线测试验证 |
| P3 | JSON action Agent、预算和轨迹 | P2；LLM adapter | 模拟 Agent 可完整重放，错误动作不能绕过约束 |
| P4 | 第二轮接入及对比评测 | P0—P3；运行环境 | 全量逐题输出、质量与成本统计可复核 |

阶段按验收推进，不作绝对工期承诺。P2 的关系索引样例可在 P1 接口确定后独立开发，但不能跳过 P0 后直接报告 Agent 优于原方法。

## P0：先固定基线正确性

目标：使原有 PPR 两轮流程的输入、输出和实验记录可信。此阶段的修复会单独记录，不将质量变化归因于 Agent。

建议文件：`main.py`、`SRgraphrag.py`、`retrieval/evidence.py`、`retrieval/judge.py`、`retrieval/metrics.py`、`utils/misc_utils.py`，以及相应的 `tests/`。

待修改项：

1. **QA 参数透传。** `run_pipeline(mode="qa")` 当前只传部分参数给 `rag_qa`；后者调用 `retrieve` 时也没有统一传入主语限制、judge 模型、检索数和结果目录。为 `rag_qa` 增加显式的检索参数或配置对象，两种 CLI 模式使用相同配置。已经给定 `QuerySolution` 时不得重复检索。
2. **第二轮分数对齐。** `retrieve` 使用第二轮 Top-5 覆盖文档，却保留第一轮分数。按最终篇章 ID 对齐分数，同时记录来源轮次；旧第一轮尾部若保留，要明确其语义。为注入但无当前策略分数的篇章定义缺失值，避免用无来源的数字冒充排序得分，并使 JSON 序列化合法。
3. **judge 元数据准确。** 实际 judge 使用 `judge_model_name`，汇总却从 `DEEPSEEK_MODEL` 读取模型名称。应由实际执行配置生成模型、端点、提示词版本和采样参数记录，避免实验标签与真实调用不一致。
4. **数值与空输入。** `graph_search_with_fact_entities` 的全数组除法会对未出现实体产生 0/0；默认 Top-k 清零通常消除了这些值，但关闭该过滤时会影响后续断言。改为受保护除法和有限值校验，显式处理缺失实体、零种子、空事实、空语料。`rag_qa` 的 `queries[0]` 访问、空查询批次的 judge 和指标除数也应有确定行为；空批次不能发起模型调用。
5. **图状态校验。** 区分“节点存在”“边存在”“映射一致”和“可运行 PPR”。现有自动补节点不能证明边和缓存有效；缺少必要缓存时应返回明确错误或已定义回退原因。
6. **judge 异常归属与解析类型。** `_judge_batch` 当前把任意任务异常写到 `i = start`，可能覆盖批首的成功结果，真正失败项却变成 `none_result`。在携带原始索引的任务内部捕获异常，保证一题一结果；解析 JSON 后校验结果为对象，处理 `[]`、`null` 等合法 JSON 但非预期结构。验收覆盖“批首先成功、后续样本再失败”，确保只标记实际失败项。
7. **运行依赖可复现。** 补齐直接依赖声明，并按在线/离线、embedding 后端拆分可选依赖及导入；未选择的后端不能强制要求 vLLM 等库。保留无模型环境下的 CLI 帮助和策略测试，在完整环境中单独验证主包导入与小图检索。

接口要求：保持现有公开方法默认行为与返回形状；增加参数时使用默认值和关键字参数。元数据反映实际运行，缺失分数不能影响文档和分数一一对应。

验收：使用模拟检索器和 judge 验证 retrieve/qa 参数一致、第二轮文档与分数对应、证据保护不被覆盖；覆盖空批次、空事实、缺节点和 `link_top_k=0`。涉及 PPR 的测试使用小图和固定向量。对依赖不足而未运行的检查明确记录，不能用语法检查替代端到端效果验证。

依赖：无 Agent、无新图数据，不要求外部模型调用。完整基线指标复跑需要已有 embedding、graph 和模型环境。

## P1：统一检索契约与 PPR 适配

目标：把“外层是否进入第二轮”和“本轮如何检索”分开，让不同图搜索策略共用种子、证据选择和输出格式。

建议文件：新增 `retrieval/types.py`、`retrieval/seeds.py`、`retrieval/ppr.py`；修改 `SRgraphrag.py` 和 `utils/config_utils.py`。

建议内部接口：

```python
GraphSearchRequest(
    original_query, retrieval_query, round_index,
    seed_fact_ids, seed_entity_ids, seed_scores,
    protected_passage_ids, retrieval_limit, evidence_limit, budget,
)

GraphSearchResult(
    ranked_passage_ids, scores, score_sources,
    selected_paths, stop_reason, fallback_reason, usage, trace,
)

strategy.search(request) -> GraphSearchResult
```

- 从 `graph_search_with_fact_entities` 抽离种子权重构建，让 Agent 与 PPR 使用相同候选事实和稳定 ID。
- 在 `retrieve_full_once` 调用 `graph_search_with_fact_entities` 的位置增加策略适配；旧方法保留兼容代理，默认策略仍为 `ppr`。
- 稳定 hash ID 与 igraph vertex index 严格区分；vertex index 只存在于 PPR adapter 内部。
- `run_ppr` 当前只返回篇章分数。如 hybrid 需要节点先验，新增内部完整节点分数结果，不改变旧调用者的二元组返回契约。
- 将本轮查询与原始问题同时传递，避免第二轮 Agent 只看到桥接问题而丢失整体目标。

验收：同一固定小图、种子和参数下，新旧 PPR 文档排序和分数一致；ID 往返转换无误；缺种子和回退状态结构化；不选择 Agent 时不新增 LLM 调用。基线正确性修复以 P0 后版本作为比较对象。

依赖：P0 通过；PPR 对照需要 igraph/numpy 及固定测试向量。

## P2：有向关系 sidecar 与图探索工具

目标：为 Agent 提供可以精确选择、验证和回溯来源的关系记录。sidecar 是独立于旧 PPR 图的关系索引，不覆盖旧缓存。

建议文件：新增 `graph/schema.py`、`graph/relation_index.py`、`graph/tools.py`、`scripts/build_relation_index.py` 和 `tests/test_relation_index.py`、`tests/test_graph_tools.py`。

最小结构：

```text
EntityRecord: entity_id, canonical_text, display_labels
FactRecord: fact_id, subject_id, predicate, object_id,
            raw_triple_variants, passage_ids
RelationIndex: facts_by_id, outgoing_fact_ids, incoming_fact_ids,
               fact_ids_by_passage, passages_by_id
IndexManifest: schema_version, normalization_version,
               corpus_fingerprint, source_fingerprint
PathStep: fact_id, from_entity_id, to_entity_id, traversal_direction
```

构建规则：

- 从 OpenIE 的原始三元组和篇章构建，保留原始显示文本，规范化和 hash 与现有 store 一致。规范化规则版本化，不能在本次接入中静默改掉实体 ID。
- 同一实体对上的多个谓词必须分别存在；同一事实在多个篇章出现时聚合来源。
- 允许沿入边探索，但关系仍保持原始主谓宾方向；不能把逆向遍历解释成谓词含义反转。
- 只索引当前语料范围中的篇章。语料、抽取内容或规范化改变时判定缓存失效；排序固定，重复构建结果可复现。
- 原始记录没有可靠置信度，首版不编造该字段。相似边、共现边不作为事实证据边。

工具接口：

```text
expand_entity(entity_id, direction, limit, cursor)
inspect_passage(passage_id)
validate_path(steps)
commit_paths(paths, protected_passage_ids, evidence_limit)
```

工具只接受已返回且存在于索引的 ID，限制邻居数量、返回文本和展开总量；工具调用失败返回结构化状态。`validate_path` 检查边身份、方向、连续性和来源；`commit_paths` 检查最终篇章能否覆盖已选路径。验证通过表示索引一致，不代表抽取事实一定正确或证据一定足够回答问题。

验收：覆盖多谓词、多来源、逆向遍历、未知 ID、空三元组、规范化重复、来源缺失、游标和预算限制；用本地 OpenIE 缓存完成只读构建检查。相同输入重复构建的 manifest、计数和稳定排序一致。没有 embedding 和模型也能运行这些测试。

依赖：P1 的 ID 和结果契约；OpenIE 缓存。第一版无需原生工具调用协议，也无需重跑抽取。

## P3：JSON action Agent、预算、轨迹与路径保护

目标：Agent 从工具提供的有限候选中选择探索动作，构建有来源的路径；程序负责执行动作和约束预算。

建议文件：新增 `retrieval/agent.py`、`retrieval/agent_state.py`、`retrieval/agent_actions.py`、`prompts/templates/agent_graph_search.py`；扩展 `retrieval/evidence.py`；按需要新增独立 LLM adapter 和缓存模块。

建议协议：

```json
{"action": "expand_entity", "arguments": {"entity_id": "entity-...", "direction": "out", "limit": 8}}
```

采用 `plan/expand/inspect/commit/stop` 中的有限动作集合；计划只记录短目标和可审计的选择理由，不要求输出隐藏推理过程。严格解析 JSON 和参数 schema，不执行模型生成的代码。首版可通过现有文本推理接口返回 JSON，再由本地 dispatcher 调工具。

`CacheOpenAI.infer` 当前只提取 `message.content`，连同元数据和缓存命中标记返回，不传回 `tool_calls`；还要求 content 为字符串，不能直接假定支持原生工具调用。如另增原生协议，需独立适配及测试。Agent 缓存应包含模型、采样参数、提示词/动作 schema 版本、工具观察和索引版本，避免改变策略后误用旧输出；记录缓存命中与实际调用成本。

Agent 状态至少包含原始问题、本轮问题、候选种子、frontier、已访问节点/边、选中路径、保护篇章、剩余预算和停止原因。种子可能含噪声，允许舍弃不相关种子，不强迫将所有种子连接成一条路径；比较型问题允许多条证据分支。

预算覆盖最大 LLM 轮数、工具调用数、单步邻居数、总展开节点/边数、最大路径长度、Token 和时间。重试同样消耗预算。未知动作、格式错误、循环、工具失败、预算耗尽均有确定退出或回退规则。

路径保护是本阶段必要部分：当前 Top-5 注入和 DPR1 guard 只保护既有篇章，可能拆断 Agent 新选的多篇路径。最终选择要同时考虑首轮保护证据和路径所需的篇章集合；DPR1 保护、去重和补齐不能拆掉已提交路径。若保护集合已占满 Top-5 或路径无法容纳，应返回容量不足，不能静默挤掉旧证据或声称路径仍完整。

轨迹记录输入 ID、动作、工具返回 ID、预算变化、选中事实和篇章、退出/回退原因及调用用量。可在不调用模型的情况下重放验证。

验收：模拟 Agent 覆盖成功补链、噪声种子、不可达、非法动作、格式修复、重复展开、预算耗尽、证据容量冲突；逐步重放得到同一提交路径。检查最终 Top-5 中仍存在每条已提交路径的全部必要来源篇章。真实模型小样本联调作为单独检查记录。

依赖：P2 图工具通过；P1 结果契约。真实模型联调需要可用 LLM 配置；离线模拟测试不依赖 API。

## P4：第二轮集成与对比评测

目标：首先只把 Agent 接入现有门控触发的第二轮，验证收益、失败情况和成本，再决定是否扩大到第一轮。

建议文件：修改 `SRgraphrag.retrieve`、`retrieve_full_once`、`main.py`、`utils/config_utils.py`；新增或扩展 `evaluation/path_eval.py`、`evaluation/repair_eval.py`、实验运行与汇总脚本。

配置建议：将 `graph_search_mode = ppr | agent | hybrid` 与 `agent_apply_to = round2` 分开。`ppr` 保持基线；`agent` 使用相同种子在关系索引探索；`hybrid` 使用 PPR 节点先验/有限候选区域引导探索。三者的回退策略必须显式配置和记录，报告 Agent 真正执行的比例以及回退样本成绩。

hybrid 的候选区域不能仅取 Top-5 篇章中的实体，否则缺失桥接证据可能在搜索开始前就被剪掉。定义候选节点数和有限边界扩展预算，单独评价候选区域覆盖率及无候选路径的失败比例。

对照至少包括：

| 实验 | 第一轮 | 第二轮 | 目的 |
| --- | --- | --- | --- |
| PPR 单轮 | PPR | 不执行 | 测量基础检索 |
| PPR 两轮 | PPR | 原桥接问题 + PPR | 对照已有受控迭代方法 |
| Agent 修复 | 相同 PPR | 门控后 Agent | 测量显式图探索收益 |
| Hybrid 修复 | 相同 PPR | 门控后 PPR 先验 + Agent | 测量先验约束收益 |

固定数据划分、抽取缓存、种子候选、第一轮结果、模型、Top-5 证据预算和门控配置。Agent 与 PPR 两轮比较时优先复用同一门控输出，避免把门控差异混入检索策略差异。工具预算和 Token 成本均需公开记录；额外执行 PPR 的 hybrid 成本也计入。

逐题保存首轮、最终结果和调用轨迹，计算：AvgRecall@5、Full Recall@5、EM/F1、Miss-1→Miss-0 修复率、Miss-0 破坏率、完整路径保留率、结构合法路径比例、非法动作率、平均/分位延迟、展开量、LLM/工具调用量、Token 和每次成功修复成本。结构合法率不能替代答案或证据覆盖指标；数据集没有标注路径或桥接实体时，不虚构这类 gold 指标。

在线检索、门控、Agent、工具和回退逻辑**不能读取 gold docs、gold answers、gold bridge entities 或真实 Miss 状态**。这些字段只交给输出冻结后的离线评估；Miss-1 用于事后分组，不能作为线上触发条件。

验收：先在固定小样本跑通四种策略，再按选定数据划分完成配对比较和成本汇总。失败及回退样本不得删除；分母、置信区间/配对统计方法明确。只有实验支持后才在论文中给出效果和贡献判断。

依赖：P0—P3 通过；完整检索所需的图、embedding、模型环境；确定的实验数据划分和预算配置。

## MVP 边界与暂缓项

首个 MVP 范围：现有英文多跳问答语料、复用 OpenIE 抽取、事实有向关系 sidecar、固定闭集图工具、JSON action 控制器、第二轮接入、有限预算、稳定轨迹、Top-5 路径集合保护，以及 ppr/agent/hybrid 可切换评测。P0 修复是 MVP 前置工作。

暂缓：训练/RL、多 Agent 协作、全图无界探索、第一轮全面替换 PPR、自动重做实体消歧、改变现有规范化或重建全部 embedding、将相似/共现边当作事实边、动态大规模增删索引、通用 Agent 框架迁移和原生工具调用适配。受限 `connect_frontiers` 双向搜索可在基础 expand/inspect 工具稳定后追加，其展开量必须计入预算。

本计划不预先认定“PPR + Agent”具有新颖性，也不承诺会提高准确率。后续论文论证应基于明确的方法约束、相关工作核对和可复现对照结果。
