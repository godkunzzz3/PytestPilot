# FirstCoder 开发规范

## 1. 项目定位

FirstCoder 是一个本地、可观测、可恢复的 Python Coding Agent。

当前项目只聚焦一个场景：

> 面向 Python/pytest 项目的 CI 失败诊断与自动修复。

系统目标是完成以下闭环：

```text
pytest 失败日志
→ 结构化提取错误证据
→ 确定性代码检索
→ 向量增强代码检索
→ 精确读取源码
→ 最小化修改
→ 局部测试
→ 完整测试
→ 输出 Diff、Trace 和运行指标
```

本项目不追求完整复刻 Claude Code 或 Codex。

核心特点：

* Agent 执行过程可观测；
* Session 事实 append-only；
* 上下文压缩可恢复；
* 向量检索与确定性检索结合；
* 任务结果由测试命令验证；
* Token、Tool Call、耗时和压缩效果可量化。

---

## 2. 当前范围

本阶段只支持：

* Python 仓库；
* pytest；
* pytest 文本日志；
* 仓库级代码索引；
* Qdrant 代码语义检索；
* FirstCoder 现有 Agent Runtime；
* 本地已配置的阿里百炼 OpenAI-Compatible Provider；
* 12–15 个可复现 Benchmark 任务。

明确不做：

* 多 Agent；
* 多语言代码修复；
* 通用长期记忆；
* 用户画像；
* Graph RAG；
* Reranker；
* Qdrant 集群；
* 云端平台；
* GitHub App；
* 自动创建 Pull Request；
* KV Cache 优化；
* 完整重写 CLI/TUI；
* 大规模 SWE-bench。

具体开发目标和验收标准以 `docs/MVP_GOAL.md` 为准。

---

## 3. 架构边界

### `firstcoder/app/`

负责：

* CLI/TUI；
* 应用装配；
* Slash command；
* Runtime 事件展示。

不得承载：

* Provider 协议；
* Tool 执行；
* 检索算法；
* Context 压缩；
* 权限策略。

### `firstcoder/agent/`

负责：

* Agent Loop；
* 模型与 Tool 循环；
* 暂停、恢复和停止条件；
* Turn 级协调。

不得承载：

* Provider SDK 细节；
* Qdrant 查询实现；
* pytest 日志解析；
* TUI 渲染。

### `firstcoder/context/`

负责：

* Append-only Session 事实；
* SessionView 重放；
* Provider 上下文投影；
* Token budget；
* Context compaction；
* Archive 和恢复；
* Checkpoint。

不得破坏性修改原始 Session 历史。

### `firstcoder/providers/`

负责：

* Provider 协议转换；
* Streaming 标准化；
* Tool Call 标准化；
* Provider 错误标准化；
* Token usage 标准化；
* Capability 声明。

不得在 `AgentLoop` 中加入阿里百炼专用分支。百炼兼容问题应在 OpenAI-Compatible Adapter 或 Capability 层解决。

### `firstcoder/tools/`

负责：

* Tool schema；
* 参数校验；
* Tool 执行；
* 结构化 ToolResult；
* 权限声明。

### `firstcoder/retrieval/`

负责：

* Python AST 代码切分；
* Embedding 接口；
* Qdrant 存储；
* 代码索引；
* 语义检索；
* 检索结果数据模型。

Retrieval 逻辑不得直接写入 Agent Loop。

### `firstcoder/ci/`

负责：

* pytest 日志清理；
* 失败测试解析；
* Assertion 和异常提取；
* Source location 提取；
* Failure fingerprint。

### `firstcoder/eval/` 和 `benchmark/`

负责：

* Benchmark Task；
* Agent Adapter；
* Evaluator；
* Trace；
* Metrics；
* 结果序列化。

---

## 4. 检索原则

代码检索使用分层策略：

```text
Stack Trace 明确路径
> 失败测试对应模块
> 精确 Symbol 或文本匹配
> Qdrant 向量相似度
```

必须遵守：

1. 向量检索只负责生成候选。
2. 模型修改代码前必须通过源码读取 Tool 获取当前真实内容。
3. 不得仅根据向量库中的代码副本修改文件。
4. Qdrant 不可用时，grep、view 和路径检索必须继续可用。
5. 检索结果必须包含路径、Symbol、行号和有界预览。
6. 检索输出必须有长度上限。
7. 索引内容必须带 content hash。
8. 文件变化后不得继续将旧 Chunk 当作当前源码证据。
9. 测试不得依赖外部 Embedding 网络服务。
10. Embedding 和 Vector Store 必须通过接口支持 Fake 实现。

---

## 5. Context 不变量

以下规则不得破坏：

1. 原始 Session 事实 append-only。
2. Context compaction 只修改 Provider 可见投影。
3. 原始证据不得因压缩而被删除。
4. 每个 Assistant Tool Call 必须保留对应 Tool Result。
5. `tool_call_id` 在压缩、Replay 和 Resume 后必须合法。
6. Provider 历史不得以孤立的 `role=tool` 开始。
7. 最新用户意图不得被有损压缩删除。
8. Fresh source read 不得被普通有损压缩删除。
9. 未知 Tool metadata 必须 fail-open，即保留证据。
10. 有损压缩前必须先 Archive 原文。
11. Archive retrieval 必须限制在当前 Session。
12. L1–L3 必须确定性、可重放。
13. L4 是唯一的模型语义摘要层。
14. Replay 和 Resume 必须幂等。
15. `PROMPT_TOO_LONG` 恢复必须有界。

不得将本项目的 Context Compaction 描述为 KV Cache 压缩。

---

## 6. Provider 规则

本地已经配置阿里百炼 OpenAI-Compatible Provider。

必须遵守：

1. 复用现有本地 Provider 配置。
2. 不新增重复配置体系。
3. 不读取、打印、复制或提交 API Key。
4. 不将凭证写入源码、测试、README、Trace 或 Benchmark 结果。
5. 单元测试必须使用 Fake Provider。
6. 真实百炼 API 只用于 Smoke Test 和 Benchmark。
7. 真实运行先执行 1 个 Smoke Task，再执行 3 个小任务。
8. 只有小任务稳定后，才运行完整 Benchmark。
9. 实际 Token usage 可用时进行累计。
10. Usage 缺失时使用 `null` 或独立 estimated 字段。
11. 不得将 estimated Token 标记为 actual usage。
12. Provider 不可用不得阻塞 Parser、Retrieval、Tool 和 Runner 的本地测试。

Smoke Test 至少验证：

* 非流式回复；
* 流式回复；
* 单次 Tool Calling；
* Tool Result 回填；
* 多轮请求；
* Usage 解析；
* Transcript 不包含凭证。

---

## 7. 指标要求

每个 Benchmark Task 至少记录：

* 是否通过；
* 测试命令和退出码；
* Provider 调用次数；
* Tool 调用总数；
* 按 Tool 名称统计；
* 实际输入、输出和总 Token；
* Estimated input Token；
* 总耗时；
* Context compaction 次数；
* 压缩前后 Token；
* Archive 数量；
* Archive retrieval 次数；
* 最终 Git Diff；
* Transcript 路径。

规则：

1. 所有 Provider 调用均需统计，包括隐藏调用。
2. 所有真实执行的 Tool 均需统计。
3. 不得根据压缩后的上下文反推 Tool 总数。
4. 不得伪造缺失指标。
5. 模型声称完成不代表任务成功。
6. 最终结果以声明的 pytest 命令为准。
7. Benchmark instrumentation 不得改变 Agent 行为。

---

## 8. 开发流程

每个非简单改动必须：

1. 阅读相关设计文档。
2. 找到拥有职责的最小模块。
3. 明确需要保持的不变量。
4. 先增加聚焦测试。
5. 进行最小实现。
6. 运行聚焦测试。
7. 运行相关测试集合。
8. 检查 Git Diff。
9. 移除无关修改。
10. 更新受影响文档。

不得：

* 大规模无关重构；
* 削弱测试；
* 修改 Benchmark 目标以提高通过率；
* 混入大范围格式化；
* 将业务逻辑放进 TUI；
* 将权限控制放进 Prompt；
* 删除原始 Session 证据。

---

## 9. 测试命令

使用 Python 3.11 或更高版本。

安装依赖：

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

完整测试：

```bash
.venv/bin/python -m pytest tests -q
```

不得在仓库根目录运行裸 `pytest`。

Benchmark 基础设施测试：

```bash
.venv/bin/python -m pytest \
  tests/test_local_pytest_benchmark.py \
  tests/test_eval_adapter.py -q
```

Context 相关测试：

```bash
.venv/bin/python -m pytest \
  tests/test_context_builder_new.py \
  tests/test_context_compaction_pipeline.py \
  tests/test_context_window_manager.py \
  tests/test_context_llm_compact.py \
  tests/test_context_resume.py \
  tests/test_context_archive.py -q
```

新增 Retrieval 和 CI 模块后，应增加对应的聚焦测试文件。

---

## 10. 依赖规则

本阶段允许引入满足 MVP 所需的：

* Qdrant Python Client；
* 一个本地 Embedding 实现或可替换 Embedding 接口。

新增生产依赖前必须：

1. 说明用途；
2. 检查现有依赖能否完成；
3. 避免引入重量级框架；
4. 提供不依赖真实网络的测试方式；
5. 更新 `pyproject.toml` 和 README。

不得为了代码检索引入完整 LangChain 或 LlamaIndex。

---

## 11. Codex 执行协议

修改前：

* 阅读本文件；
* 阅读 `docs/MVP_GOAL.md`；
* 检查当前代码和测试；
* 给出简短计划；
* 说明关键不变量。

执行过程中：

* 按 `docs/MVP_GOAL.md` 的阶段顺序推进；
* 每个阶段保持仓库可运行；
* 普通实现决策采用最简单、最可测试的方案；
* 不因非关键选择暂停；
* 遇到测试失败应继续定位和修复；
* 每个阶段完成后检查 Diff。

仅在以下情况暂停：

* 需要新的凭证；
* 需要破坏性数据操作；
* 需要改变项目核心架构；
* 需要明显扩大目标范围；
* 发生无法自行解决的外部阻塞。

完成后必须报告：

* 修改内容；
* 修改文件；
* 测试命令；
* 实际测试结果；
* 实际 Benchmark 数据；
* 未获得的数据及原因；
* 已知限制；
* 未完成项；
* 可用于简历的项目描述。

不得在必要测试未运行时声称任务完成。
