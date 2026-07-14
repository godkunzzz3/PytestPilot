# FirstCoder 一周 MVP 目标

## 目标

在一周内交付一个仅面向 Python/pytest 仓库的本地 Coding Agent 闭环：从 pytest 失败日志提取结构化证据，结合确定性检索与本地向量检索定位源码，执行最小修改并以 pytest 结果验收，同时输出可复现的 Benchmark 指标。

## 必须完成

1. 最小 Benchmark 指标贯通：
   - `context_metrics`；
   - Provider 调用次数；
   - Tool 调用总数和按名称统计；
   - actual input/output/total token usage（缺失时为 `null`）；
   - estimated input token（与 actual usage 分开）；
   - 总耗时。
2. pytest 文本日志结构化解析，包括失败测试、异常或 Assertion、源码位置和稳定 fingerprint。
3. Python AST 代码切分，结果包含路径、symbol、行号、content hash 和有界内容。
4. 通过可替换接口使用 FastEmbed 生成本地向量；测试使用 Fake Embedding，不依赖网络。
5. 通过可替换接口使用 Qdrant local mode 建立本地持久化索引；测试可使用临时目录或 Fake Store。
6. 提供 `code_search` 语义检索 Tool。向量结果只作为候选，修改前仍必须读取当前源码；Qdrant 不可用时确定性检索继续工作。
7. 提供 pytest 修复 Skill 或 CLI 闭环：解析失败、检索、读取、最小修改、局部测试、完整测试、输出 diff 与指标。
8. 提供 8–10 个小型、离线、可复现的本地 Benchmark 任务。
9. 对相同任务记录 deterministic baseline 与 vector-enhanced 两种模式的结果，禁止改变题目或判分命令来提高通过率。
10. 更新 README、安装与运行命令、Fake Provider Benchmark 命令和演示说明。

## 本周不做

- 新的完整 Trace 子系统；
- `archive_read`（复用已有 `retrieve_archive`）；
- FreshSourceGuard；
- 专用 TUI 面板；
- 14 个 Benchmark；
- 多 Agent；
- 多语言；
- 多 Provider Embedding；
- Reranker；
- 云部署；
- 无关重构。

## Provider 限制

阿里百炼/Qwen 免费额度耗尽期间：

- 禁止真实 Provider 调用和自动重试；
- 单元测试、集成测试、开发和 Benchmark 全部使用 Fake Provider；
- 在用户明确确认额度恢复前，不运行真实 Smoke Test 或真实 Benchmark；
- FastEmbed 模型准备和 Qdrant local 可以在本地运行；
- 不读取、打印、复制或提交任何 API Key。

## 实施顺序与阶段门

### A. 代码和测试基线

- 阅读项目规范与相关设计文档；
- 记录现有未提交修改并保持不覆盖；
- 运行现有聚焦测试和完整测试，记录真实结果。

### B. 最小指标贯通

- 先补聚焦测试，再在 `firstcoder/eval/` 和 `benchmark/` 的最小职责范围实现；
- 所有 Provider 调用（包括隐藏摘要调用）和真实 Tool 调用必须在事件发生时累计，不从压缩上下文反推。

### C. pytest Parser

- 在 `firstcoder/ci/` 实现纯本地解析和数据模型；
- 使用真实风格 fixture 覆盖 Assertion、异常、多失败和路径位置。

### D. Qdrant、FastEmbed 与 code_search

- 在 `firstcoder/retrieval/` 实现 AST chunk、Embedding/Store 接口、索引与搜索；
- 在 `firstcoder/tools/` 只放 Tool schema、校验和执行适配；
- 索引记录 content hash，搜索结果长度有界且包含路径、symbol 和行号。

### E. pytest repair Skill/CLI

- 复用现有 Agent Runtime、Tool、权限和 Session；
- 不把检索实现或 pytest Parser 塞进 Agent Loop/TUI。

### F. Benchmark 与对比

- 扩充到 8–10 个固定任务；
- 使用 Fake Provider 对 deterministic baseline 和 vector-enhanced 运行相同题目与测试命令；
- 记录真实通过率、退出码、耗时、调用次数、token 字段、diff 和 transcript。

### G. 文档与最终验证

- 更新 README 和演示命令；
- 运行所有新增聚焦测试、相关测试集合和 `.venv/bin/python -m pytest tests -q`；
- 检查最终 diff，报告实际数据、缺失数据、限制和未完成项。

每个阶段完成后必须运行聚焦测试、检查 `git diff`、记录完成项与失败项，并保持仓库可运行。

## 验收标准

- 全部开发与测试不触发真实 Provider；
- 8–10 个任务可从干净临时目录重复生成并以声明的 pytest 命令判分；
- 两种检索模式的输出 schema 一致且结果可比较；
- 缺失的 actual usage 保持 `null`，estimated token 单独记录；
- Qdrant 或向量组件不可用时，pytest Parser、确定性检索、Tool 和 Runner 的本地测试仍可运行；
- 原始 Session 事实 append-only，compaction 仅改变 Provider 投影，tool call/result 配对、Replay 和 Resume 不变量不被破坏；
- 最终交付包含修改文件、测试命令与结果、实际 Benchmark 数据、未获得数据及原因、已知限制、未完成项和可用于简历的项目描述。

## 2026-07-14 实际状态

- A–F 已实现：指标贯通、pytest Parser、AST 切分、FastEmbed、Qdrant local、`code_search`、`pytest-fix` CLI 和 9 个 Benchmark 任务。
- 9 个任务均已在干净临时目录确认初始 pytest 非零退出；Evaluator 会拒绝测试文件修改和 editable scope 外写入。
- Fake Provider 工作流已覆盖 `view → edit → focused pytest → full pytest`；所有 Provider 自动重试设为 0。
- 本地缓存 `BAAI/bge-small-en-v1.5`（384 维）与 Qdrant local-persistent 已重跑。`parser_dispatch` 的 deterministic baseline Relevant-file Hit@5 为 false，vector 为 true；索引 0.049246 秒，查询 0.003564 秒。
- `service_repository_contract` 的两种模式 Relevant-file Hit@5 均为 true；vector 索引 0.028569 秒，查询 0.004178 秒。原始 Top-5 保存在 gitignored 的 `runs/audit-hardening/offline-retrieval-results.json`。
- 百炼仍未恢复；2026-07-14 另使用 DeepSeek 官方 API、`deepseek-v4-flash`、thinking disabled 完成真实验证。最终 Smoke 7/7 通过；9 题 baseline 7/9，9 题 vector-enabled 8/9，累计保守估算成本 `$0.23979872`。Vector 组没有实际调用 `code_search`，因此通过率差不能归因于向量检索，Agent-level Hit@5 为 `null`。
- DeepSeek 审计加固已关闭 SDK 隐式 retry，增加请求前 1.20 安全系数预算预留、usage 缺失停机、Benchmark Skill 隔离、tree/diagnostics 加固、逐路径 source-read 校验和 retrieval-required 策略。
- 两个语义任务的受控小样本中，Baseline 6/6、合规 Vector 6/6；Vector 实际调用 `code_search` 6 次并重新读取候选。两组通过率相同，不能声称检索提升准确率；Vector 平均 Token、Tool Call 与耗时更高。
- 初始 12 项中有两个 Vector 运行测试虽通过但未调用 `code_search`，被标记并排除；策略门加固后仅补跑这两项并通过。本轮 Smoke、12 项及 2 个补跑累计保守成本 `$0.08710954`，低于 `$0.25`，无 usage 缺失或预算中止。
- 加固后完整测试：`.venv/bin/python -m pytest tests -q` 得到 891 passed、2 skipped、0 failed（37.61 秒）；文档完成后仍需执行最终一次完整验证。
