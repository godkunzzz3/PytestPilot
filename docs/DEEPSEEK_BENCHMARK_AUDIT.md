# DeepSeek Benchmark 审计报告

## 范围与结论

本报告记录 2026-07-14 对 FirstCoder DeepSeek Benchmark 路径的加固和小样本配对实验。实验使用 DeepSeek 官方 OpenAI-compatible API、`deepseek-v4-flash`、thinking disabled、temperature 0、4096 最大输出、SDK retry 0、FirstCoder retry 0、单次尝试、最多 8 个 Tool rounds、串行执行。

最终可比较样本为两个语义任务、每种配置各 3 次：Baseline 6/6，Vector 6/6。Vector 六次均实际调用 `code_search`，并通过 `read_multi` 重新读取命中候选；两组通过率相同，因此不能声称 Vector 提升修复准确率。Vector 在本样本中使用了更多 Token、Provider/Tool calls 和时间。

## 旧实验为何不可归因

旧实验单次结果为 Baseline 7/9、Vector-enabled 8/9，但 Vector 组没有实际调用 `code_search`，Agent-level Hit@5 为 `null`。因此 8/9 与 7/9 的差异可能来自模型随机性或时间因素，不能归因于向量检索。旧累计保守成本 `$0.23979872` 不计入本轮 `$0.25` 新预算。

## Provider retry 修复

`ProviderPreset` 和通用 OpenAI-compatible Adapter 现在支持 `sdk_max_retries`。DeepSeek preset 为 0，真实 `OpenAI(...)` 客户端收到 `max_retries=0`；注入 Fake client 时不会构造 SDK client。Benchmark 外层 retry 固定为 0，Provider 异常不会被 FirstCoder 再次调用。审计配置只包含非敏感属性，不保存 API Key。

## 请求边界预算

共享的 `RequestBoundaryBudget` 在每个请求发送前：

1. 使用现有字符 Token estimator 估算消息和 Tool schema；
2. 对估算输入乘 1.20；
3. 按 4096 最大输出预留；
4. 所有输入按 cache-miss `$0.14/1M`，输出按 `$0.28/1M`；
5. committed cost 加本次 reservation 超过 `$0.25` 时，在 HTTP 前拒绝；
6. 返回 usage 后按实际 Token 保守对账并记录 cache hit/miss；
7. usage 缺失或发送后异常会标记 `usage_unknown` 并停止后续请求。

Streaming 只在最终 `message_completed` usage 提交一次。准确边界是：在每个请求发送前进行保守成本预留，并在 usage 返回后对账；usage 缺失时停止后续请求。它不能绝对保证供应商最终账单不超过预算。

## Benchmark 隔离与 Tool 加固

- Benchmark 默认使用固定 Workflow Prompt，不发现或加载用户全局 Skill；项目 Skill 只有显式 allowlist 才可加载。所有实际实验的 `loaded_skill_ids=[]`，`nature-paper2ppt` 未进入 Session。
- Benchmark 不注册 `ask_user`。普通 CLI/TUI 的 Skill 与 Tool 默认行为不变。
- `tree` 默认排除 Git、虚拟环境、缓存、build/dist、runs、Qdrant、模型目录和 node_modules，且限制节点数与字符数。
- `diagnostics` 非零退出会返回 pytest status、计数、node ID、异常、消息、位置、expected/actual、fingerprint、有界 tail 和 unparsed 标记。

## 路径级 source-read 与 retrieval policy

Evaluator 按 Transcript 执行顺序提取 `view/read_multi` 与 `edit/write/delete/apply_patch` 的规范化路径。每个既有被修改文件必须在首次修改前被准确读取；新文件、越界路径、未读路径和 stale-read 字段分别输出。测试文件修改和 editable scope 外写入仍直接失败。该次历史 Benchmark 尚未实现运行时 FreshSourceGuard；2026-07-25 后的 `pytest-fix` 已增加基于 path、SHA-256、TTL 和一次性 read token 的 Runtime 强制校验，旧结果不追溯重标。

Baseline 不注册 `code_search`。Vector 的 retrieval-required 任务注册该 Tool，并通过通用首 Tool 约束在第一次请求强制选择它；这段逻辑位于 Benchmark Provider 装饰层，不修改 AgentLoop，也不包含 DeepSeek 分支。向量 preview 不算 source read，必须再次 `view/read_multi`。

## 离线 Top-5 重跑

原始 JSON 位于 gitignored 的 `runs/audit-hardening/offline-retrieval-results.json`，未调用 Chat Provider。

| Task | Expected | Baseline Top-5 | Baseline Hit@5 | Vector Top-5（score） | Vector Hit@5 | Index / Query |
| --- | --- | --- | ---: | --- | ---: | ---: |
| `service_repository_contract` | `src/service.py` | `src/service.py`, `tests/test_service.py` | true | `src/service.py` module 0.686280; `user_label` 0.639084; `src/repository.py` 0.600849 | true | 0.028569s / 0.004178s |
| `parser_dispatch` | `src/parser.py` | `src/api.py`, `tests/test_parser_dispatch.py` | false | `dispatch_payload` 0.903955; parser module 0.903955; `process_event` 0.858597 | true | 0.049246s / 0.003564s |

两项均索引 6 chunks，Embedding 为 `BAAI/bge-small-en-v1.5`、384 维，Qdrant 为 local-persistent。

## 真实实验过程

零副作用 Smoke 通过：实际模型 `deepseek-v4-flash`，usage 15 input / 8 output / 23 total，成本 `$0.00000434`，SDK retry 0、thinking disabled、temperature 0、请求 reservation 生效。

首轮严格交错执行 12 项。两个 `service_repository_contract` Vector 运行虽然 pytest 通过，但没有调用 `code_search`，Evaluator 将其记为 `retrieval_policy_violation` 并排除。加固首 Tool 约束并重新通过完整本地门禁后，只补跑这两个无效项；补跑被明确标记为 remediation rerun，原结果没有覆盖。该补跑发生在交错序列之后，是本实验的时间配对限制。

### 最终纳入比较的 12 项

| Task | Run | Mode | Pass | Input / Output | Provider / Tool | code_search | Elapsed | Cost |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| service | 1 | baseline | yes | 42,618 / 737 | 7 / 6 | 0 | 13.068s | `$0.00617288` |
| service | 1 | vector | yes | 38,386 / 796 | 6 / 6 | 1 | 11.946s | `$0.00559692` |
| parser | 1 | baseline | yes | 36,359 / 784 | 6 / 5 | 0 | 12.766s | `$0.00530978` |
| parser | 1 | vector | yes | 39,101 / 913 | 6 / 6 | 1 | 12.699s | `$0.00572978` |
| service | 2 | baseline | yes | 36,072 / 635 | 6 / 5 | 0 | 11.092s | `$0.00522788` |
| service | 2 | vector remediation | yes | 68,073 / 1,323 | 10 / 13 | 1 | 16.601s | `$0.00990066` |
| parser | 2 | baseline | yes | 36,347 / 744 | 6 / 5 | 0 | 11.962s | `$0.00529690` |
| parser | 2 | vector | yes | 38,812 / 914 | 6 / 6 | 1 | 12.744s | `$0.00568960` |
| service | 3 | baseline | yes | 36,054 / 666 | 6 / 5 | 0 | 9.658s | `$0.00523404` |
| service | 3 | vector remediation | yes | 68,008 / 1,289 | 10 / 13 | 1 | 17.536s | `$0.00988204` |
| parser | 3 | baseline | yes | 43,024 / 888 | 7 / 6 | 0 | 13.079s | `$0.00627200` |
| parser | 3 | vector | yes | 39,080 / 889 | 6 / 6 | 1 | 12.770s | `$0.00572012` |

每个 Vector 合规运行的 candidate-read count 为 3，Relevant-file Hit@5=true，source-read 与 retrieval policy 均无违规，测试文件和 editable scope 均未被破坏。

## 汇总

| Metric | Baseline | Vector |
| --- | ---: | ---: |
| Pass rate | 6/6 (100%) | 6/6 (100%) |
| Input tokens | 230,474 | 291,460 |
| Output tokens | 4,454 | 6,124 |
| 平均 input / output | 38,412.3 / 742.3 | 48,576.7 / 1,020.7 |
| Cache hit / miss | 213,888 / 16,586 | 267,520 / 23,940 |
| Provider calls | 38 | 44 |
| Tool calls | 32 | 50 |
| Source reads | 6 | 6 |
| code_search calls | 0 | 6 |
| 平均耗时 | 11.94s | 14.05s |
| 合规样本保守成本 | `$0.03351348` | `$0.04251912` |

首轮两个被排除项分别成本 `$0.00552034` 和 `$0.00555226`。包括 Smoke、12 个首轮任务及两个补跑，本轮共有 95 个真实请求，累计保守成本 `$0.08710954`，未达到 `$0.25`，没有 usage_unknown，也没有预算中止。

## 失败案例、限制与可成立结论

唯一实验失败分类是两个首轮 Vector 的 `retrieval_policy_violation`：pytest 实际为 0，但没有调用 `code_search`，故不计作可归因通过。策略加固后的补跑均通过。

可成立：请求边界预算、SDK/FirstCoder retry 0、Skill 隔离、逐路径读取校验和真实 `code_search → read → edit → pytest` 链路均已被测试及小样本验证。不能成立：向量检索提升通过率、结果具有统计显著性、可推广到 9 题或公开 Benchmark、预算机制能保证供应商最终账单。当前还不支持 Thinking 模式、`reasoning_content` 回传、stale-read hash enforcement、多语言、多 Agent、reranker 或云部署。
