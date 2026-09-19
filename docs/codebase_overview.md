# 代码结构与实现状态

开发基于 `dev` 分支的 `6fa8e2c`。先完成兼容性整理，再按原 [P0—P4 计划](agent_graph_retrieval_plan.md) 修复基线并实现 Agent MVP。默认仍为 PPR 两轮流程；Agent/Hybrid 只在显式选择后应用于门控触发的第二轮。代码与冒烟验证已完成，不代表完整数据集效果实验已完成。

## 模块职责

以下路径相对于仓库根目录。没有移动数据、模型缓存、既有实验结果或提示词配置文件。

| 路径 | 当前职责 | 本次变化 |
| --- | --- | --- |
| `main.py` | CLI、数据加载、运行模式选择、结果保存 | 重依赖导入延后到参数解析之后，使 `--help` 可以离线使用 |
| `src/SRgraphrag/SRgraphrag.py` | 索引、图构建、检索编排、PPR、QA | 保留原调用入口；增加可选策略参数与基线修复 |
| `src/SRgraphrag/retrieval/facts.py` | 主语数量限制提示词、候选主语计数 | 从单轮检索的局部函数提取 |
| `src/SRgraphrag/retrieval/evidence.py` | Top-5 证据注入、DPR1 保护、去重、分数对齐 | 从单轮检索的局部函数提取 |
| `src/SRgraphrag/retrieval/metrics.py` | 平均召回、完整召回、缺失证据、Hit@1、MRR | 从迭代检索的局部函数提取 |
| `src/SRgraphrag/retrieval/judge.py` | 批量可回答性判断、桥接问题、证据解析 | 从类方法提取；原类方法保留代理入口 |
| `src/SRgraphrag/retrieval/types.py`、`seeds.py`、`ppr.py`、`dispatch.py` | 稳定 ID 契约、事实种子、PPR 适配、策略选择与回退 | 新增 |
| `src/SRgraphrag/retrieval/agent.py`、`agent_actions.py`、`agent_llm.py` | JSON 动作、预算、轨迹、无模型重放、独立文本请求缓存 | 新增 |
| `src/SRgraphrag/graph/` | 有向关系索引、来源追溯、闭集图工具、路径验证与证据容量 | 新增，不改原 igraph |
| `src/SRgraphrag/embedding_store.py`、`embedding_model/` | 向量存储与编码后端 | 后端工厂改为按需加载，旧缓存格式不变 |
| `src/SRgraphrag/information_extraction/`、`llm/` | 抽取与模型接口 | 后端按需加载 |
| `src/SRgraphrag/prompts/` | 事实过滤及 Agent 提示词 | 原过滤提示词不变，新增 Agent 文本动作提示词 |
| `src/SRgraphrag/evaluation/repair_eval.py` | 冻结输出后的修复/破坏率、配对统计、策略用量统计 | 新增 |
| `tests/` | 标准库回归与依赖可用时的真实小图集成测试 | 覆盖 CLI、judge、路径、预算、回放、数值及编排 |

`StandardRAG.py` 和原有 DPR 专用入口仍保留。主类中的 PPR 兼容入口现通过共享种子与策略适配器执行；底层仍使用原加权无向 PPR。

## 实际执行关系

`main.py` 加载语料和问题，创建配置与实例，先完成 `index(docs)`，再按模式调用检索或 QA。

### 索引与数据

`index` 复用或生成 OpenIE 结果，规范化实体与三元组，维护 chunk/entity/fact 三类向量库，然后建立实体事实边、篇章关联边和实体相似边。

需要区分三种数据：

- **OpenIE 缓存**：按篇章保存实体和原始三元组，可追溯来源；`prepare_retrieval_objects` 还会构建 `proc_triples_to_docs`。
- **向量库**：用于查询与篇章、实体、事实之间的语义匹配，保存稳定内容 ID 和对应内容。
- **igraph 加权图**：用于 PPR 扩散，包含实体与篇章节点；现有边最终只保存权重，不保留完整谓词、来源和语义方向。实体主体/客体的双向连接不能等同于有向事实路径。

Agent 的有向关系索引从三元组及其来源构建，不从仅含权重的 igraph 边反推事实。稳定内容 ID 与 igraph 内部 vertex index 严格区分；关系索引在重新索引、删除、语料范围或来源文件变化后失效。

### 单轮检索：`retrieve_full_once`

固定顺序为：DPR 篇章检索 → 事实候选打分与 LLM 过滤 → 有事实时 PPR、否则 DPR 回退 → 首轮证据注入 → 本轮 DPR1 保护 → Top-5 去重补齐与分数对齐。

其中 `subject_cap` 限制为 1—5，提示词内容沿用原实现。证据先于 DPR1 注入，后续替换不能覆盖受保护证据。输出仍为原字典，新增 `graph_search`、`path_protected_docs` 和 `repair_applied`。未知得分与 Agent 路径得分为 `None`，序列化为合法 JSON `null`。

第二轮可选 `agent` 或 `hybrid`。Agent 从相同种子开始，只操作工具已返回的 ID；Hybrid 以 PPR 节点分数选取有限区域并加入一层有预算的边界，不只取 Top-5 文档。当前 Hybrid 使用区域约束，不改变种子权重。提交路径的必要来源和首轮保护证据必须共同装入 Top-5，DPR1 不能挤掉它们。

路径验证只检查索引身份、方向、连续性与来源覆盖，不保证抽取事实正确或回答充分。Agent 可以忽略噪声种子并提交多分支。回退选择为 `ppr`、`dpr`、`none`，实际执行及原因均记录；`none` 表示失败时保留第一轮，不应用不完整修复。

### 迭代检索：`retrieve`

1. 全部问题完成第一轮 `retrieve_full_once`。
2. 批量 judge 判断 Top-5 是否可回答，并在需要时生成桥接问题。
3. 仅在不可回答、允许桥接且桥接问题非空时进行第二轮；把首轮 judge 选中的证据交给单轮检索保护。
4. 将第二轮 Top-5 写入最终结果；未触发则保留第一轮。最多两轮，不是任意次数的 Agent 循环。

不传 `gold_docs` 时返回 `list[QuerySolution]`；传入时返回 `(results, metrics)`。gold 只用于指标与缺失证据分桶，不用于第二轮门控、图工具或 Agent。最终 Top-5 使用实际轮次中对应文档的分数；第一轮尾部保留其自身得分。

### QA：`rag_qa`

输入问题字符串时调用 `retrieve`，输入已有 `QuerySolution` 时直接使用检索结果；随后调用 `qa` 生成答案，并在提供 gold answers 时评估。实际调用方向是 `rag_qa → retrieve → retrieve_full_once`。QA 与 retrieve 共用完整 CLI 检索配置；空查询不初始化检索模型。

## 兼容性范围与检查

保持 PPR 算法与权重公式、图缓存格式、内容 ID、事实过滤提示词、Top-5 替换顺序和第二轮门控；P0 修复后不再保留 NaN、分数错位和异常错归行为。同权重实体按稳定 ID 打破并列，避免原 set 顺序的不确定性。新增选项均有兼容默认值；`judge_answerability_and_bridge` 的类方法仍可按原方式调用。

在仓库根目录执行：

```bash
python3 -S main.py --help
python3 -S -m unittest discover -s tests -v
git diff --check
```

`-S` 禁用 site-packages，测试直接加载 `retrieval` 叶子包，不会初始化模型或调用 API。主包本身仍有重依赖，不能据此认为完整系统已可在无依赖环境运行。

以下为**首次纯整理阶段的历史验证记录**，不代表后续 P0 修复版本仍与旧缺陷逐项相同：

- 19 项标准库单元测试通过，覆盖 CLI 帮助、证据保护与注入顺序、重复与空结果、五槽容量、分数按文档对齐、主语上限及指标语义。
- `main.py`、`src/`、`tests/` 下共 57 个 Python 文件通过语法解析与编译检查；CLI 帮助和 `git diff --check` 通过。
- 相对整理前提交，14 个提取函数的 AST 在统一函数重命名、移除文档字符串及 judge 的 `self` 参数后完全一致；32 个类方法的参数签名保持一致，未重构的 29 个方法体保持一致。
- 一次性差分检查使用固定随机种子 `20260919`，以模拟向量排序和事实过滤结果运行旧/新单轮函数：1,000 组输入的输出及后端调用参数一致。
- 另有 48 组模拟迭代场景覆盖有/无 gold、初始化状态、门控分支、空批次、检索数和分桶开关；旧/新返回值及调用一致，忽略实际耗时字段。空批次使用的是模拟 judge，不代表真实 judge 已支持无调用退出。

AST 与差分检查是本次针对基线的专项验证，并非额外的常驻单元测试。它们不验证真实模型输出、PPR 数值计算、HTTP 协议或效果指标。

本地默认 Python 为 3.9.6，缺少科学计算与模型依赖；服务器为 Python 3.12.3，有现成运行依赖和图缓存。完整依赖环境运行 `python -m unittest discover -s tests -v` 会增加真实类与小图集成测试。`scripts/validate_agent_runtime.py` 检查现有 OpenIE/图，只有显式 `--live` 才调用 API，不加载 embedding 模型、不修改旧图缓存。最新验证范围与结果见 [实现与验证记录](implementation_status.md)。

## 已修复问题与评测边界

原代码中以下问题已纳入 P0 修复：

- QA 参数漏传与实验配置不一致。
- 第二轮 Top-5 文档与分数错位、无来源分数输出 Infinity。
- judge 非对象 JSON、异步异常错归批首、空批次与错误模型元数据；回放时使用冻结输出的实际元数据。
- PPR 非有限种子、缺失实体、空输入、缺边/缺映射/非法边权；不再只补节点就保存成“已修复”的图。
- 在线/离线模型后端强制导入与直接依赖缺失。

提供 `result_save_root` 时，无论是否有 gold 都保存 `retrieval_trace.jsonl`。默认目录为 `DATASET/ITER_capN`，Agent/Hybrid 追加 `_agent`/`_hybrid`；同目录重复运行会覆盖同名结果，正式实验应使用独立输出根目录。

`--round1_replay_path` 固定首轮证据和 judge，仅读取允许的检索字段，拒绝问题顺序不一致。`scripts/evaluate_repair.py` 在输出冻结后读取独立 gold，计算修复/破坏率、路径来源保留、配对 bootstrap 和轨迹中已有的策略用量。不能将缺少路径 gold 的结构指标叫作语义路径正确率，不能将未记录的全链路成本报为零。

尚未完成完整数据集的四组配对效果实验、全部后端版本组合测试及真实 embedding→检索→QA 的大规模运行；当前冒烟验证不构成 Agent 优于 PPR 的证据。
