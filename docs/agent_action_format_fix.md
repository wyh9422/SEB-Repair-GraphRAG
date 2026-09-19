# Agent 动作协议修复与取消累计 Token 截断

日期：2026-09-19。起点：`eb4d677`。最终被测代码：`1998c60`。

## 修改范围

- 提示词版本更新为 `agent-graph-search-v4`，六种动作均提供可被解析器接受的完整 `action + arguments` 示例。
- 在已有可用种子/锚点时优先调用图工具，避免不必要的 `plan` 回合。仍保留 plan/validate/stop，不强制提交路径。
- 错误反馈明确说明 `arguments` 封装和允许字段。扁平、混合、未知字段、重复键等错误仍严格拒绝，没有静默补造参数或放松实体/关系校验。
- 按用户要求，累计 Token 截断改为默认关闭：CLI `--agent_max_tokens none`，Python `max_tokens=None`。Token 使用量照常记录。
- 步数 8、工具调用 16、扩展边数 64、每次邻居 8、路径 4 跳、总时间 60 秒、重试 2 次保持不变。单次 JSON 动作输出仍最多 1,024 token，服务提供方上下文限制仍生效。
- 可显式传入 `--agent_max_tokens 12000` 恢复旧的累计限制；提前停止时 trace 记录已用预算、剩余预算、下次输入估算和估算方法。程序接口的 `max_tokens=0` 仍表示零 Token 预算，不等于关闭限制。
- 第一轮 PPR、事实过滤、judge 门控、原图、语料 embedding、路径来源与 Top-5 保护约束、PPR 回退不变。

## 问题与修复验证

原始三数据集各前 20 条测试中，2Wiki 有 3 条、MuSiQue 有 10 条触发 Agent；HotpotQA 没有触发。37 个 Agent 模型响应都缺少 `arguments`，工具调用为 0，13 条全部回退到 PPR。

第一次仅修复格式（v2）后，非法动作归零，真实工具调用 17 次，但仍无成功路径。精简提示词、优先工具（v3）后，工具调用达到 24 次，11 条仍被累计 Token 预检查截断，另外 2 条主动停止。

旧预检查把 UTF-8 字节数作为下一次输入的保守 Token 上界。因此 `token_budget` 并不表示已实际用完 12,000 token。例如一次停止时实际累计使用 1,809 token、剩余 10,191，但下次输入字节估算为 10,507，于是直接停止。按照用户新要求取消这一累计截断后，Agent 能继续探索和提交证据。

## 冻结输入复测结果

复用原测试的第一轮结果和 judge 输出，在 2Wiki、MuSiQue 各前 20 条上仅重新执行原来的 13 条第二轮请求。脚本校验：问题顺序、触发位置、第一轮和 judge 完全一致；Agent 初始上下文除提示词版本及 Token 限制外一致。HotpotQA 原先无触发，未重复运行。

| 数据集 | Agent 触发问题数 | 成功提交路径的问题数 | 回退问题数 | 格式错误次数 | 工具调用：成功 / 总尝试 |
|---|---:|---:|---:|---:|---:|
| 2WikiMultihopQA | 3 | 3 | 0 | 0 | 15 / 15 |
| MuSiQue | 10 | 7 | 3 | 0 | 43 / 48 |
| 合计 | 13 | 10 | 3 | 0 | 58 / 63 |

10 个成功问题的路径与来源全部通过无模型重放，路径来源和受保护文档均保留在最终 Top-5。没有通过扩大证据容量或允许访问未知节点来提高提交率。

MuSiQue 剩余回退（从 0 开始的样本索引）：

- 5、14：模型尝试访问未被工具暴露的实体，被 `unexposed_id` 拒绝后主动停止。
- 7：两次提交均因 `evidence_capacity_exceeded` 被拒绝，受保护文档不允许为新路径腾出位置，随后主动停止。

这三条最后的终止原因均为 `agent_stop`，不是 `invalid_action` 或 `token_budget`。没有强行修改原保护策略。

| 数据集（各 20 条） | 原测试最终平均 Recall@5 | 本次最终平均 Recall@5 | 原 / 本次完整召回率@5 |
|---|---:|---:|---:|
| 2WikiMultihopQA | 100% | 100% | 100% / 100% |
| MuSiQue | 75% | 79.17% | 55% / 65% |

这些是固定前 20 条的工程回归结果，不是完整 benchmark 或泛化提升证明。成功提交说明路径满足当前图和来源约束，不等于事实抽取或答案必然正确。

## “成功提交路径”的含义

`expand_entity` / `inspect_passage` 只让 Agent 观察关系和语料。Agent 必须调用 `commit_paths` 选定路径，再通过方向、连续性、已暴露 ID、来源与容量检查，才会成为检索结果。没有提交不等于图里没有路径，也不等于没有调用过工具。

## 成本、资源与测试

- 2Wiki：15 次 Agent 模型调用，96,631 token，含加载总耗时 43.6 秒。
- MuSiQue：54 次 Agent 模型调用，247,301 token，含加载总耗时 109.0 秒。
- 合计 343,932 个 Agent API 报告 token，Agent 本地响应缓存命中 0；不代表整个系统完整账单，也没有按价格换算费用。
- 两数据集串行运行，采样峰值内存约 23.37 GiB / 90 GiB，显存最高 17,003 MiB。资源每 5 秒采样，瞬时峰值可能未捕获。
- chunk/entity/fact parquet 的大小与纳秒 mtime 前后不变；原始 trace 的 SHA-256 不变。无须重新 embedding 语料。
- 137 项测试：服务器全部通过；本地标准库环境通过 128 项、明确跳过 9 项科学计算集成测试。语法编译及 `git diff --check` 通过。
- 新测试覆盖完整示例可解析、明确纠错、严格拒绝扁平动作、缓存隔离、无模型重放、关闭累计限制后继续提交、用量统计，以及关闭 Token 限制后步数/工具/重试约束仍有效。

## 服务器证据位置

项目根目录：`/root/autodl-tmp/SEB-Repair-GraphRAG-main`。

- 原始测试：`result_outputs/agent-smoke20.N9EAGZ`
- 格式修复 v2：`result_outputs/agent-format-fix.rVlk1v`
- 精简提示词 v3：`result_outputs/agent-format-fix-v3.2QxJoW`
- 最终关闭累计截断：`result_outputs/agent-no-token-cap.m0dS0f`
- 最终 tmux 会话：`seb-agent-no-token-m0dS0f`（已正常退出，保留 pane）。

每个运行目录均保留 `status.json`、监督脚本、日志、资源采样、每题检索 trace 和离线评估报告，未覆盖原实验。
