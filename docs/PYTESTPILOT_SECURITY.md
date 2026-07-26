# PytestPilot 执行安全与 Attempt 隔离

## 执行边界

`ExecutionBackend` 接收参数数组、工作目录和 `ResourceLimits`。当前实现：

- `LocalProcessBackend`：仅用于可信项目与单元测试；使用环境白名单、进程组超时、输出上限以及平台支持的 rlimit，但不提供网络或宿主文件系统隔离。
- `DockerSandboxBackend`：用于不可信仓库；关闭网络，根文件系统只读，worktree 单独以可写 bind mount 暴露，使用非 root 用户，丢弃 Linux capabilities，启用 `no-new-privileges`，限制 CPU、内存、pids、单文件大小、tmpfs、输出与墙钟时间。

Docker 使用 `--pull=never`，运行时不会联网拉取镜像。先构建 `docker/pytest-sandbox.Dockerfile`；若目标项目依赖 pytest 之外的第三方包，应制作预装且版本锁定的派生镜像。当前实现限制单文件大小，但 bind mount 的总磁盘配额仍由宿主文件系统或 Docker 运行环境负责。

相同 Backend 同时用于 Workflow 的 baseline/focused/full pytest，以及 Agent 的 `diagnostics`、`shell` 和 `python_exec`，避免验证命令被隔离但 Agent 内部命令仍在宿主机执行。

## FreshSourceGuard

启用 Guard 的读取返回：

```text
path + sha256 + size + read_at + read_token
```

已有文件的 `edit`、`write`、`delete` 和 `apply_patch` 在 mutation 前强制校验：

1. token 存在且属于同一路径；
2. 路径位于 `editable_paths`；
3. 当前内容 SHA-256 与 size 等于读取版本；
4. token 未超过 TTL；
5. 成功写入后 token 立即消费，不能重放。

新文件不需要伪造读取，但路径必须提前列入 `editable_paths`。Guard 在 Tool executor 内校验，失败时不会写文件；Transcript 审计继续保留为二次证据，不再是主要执行约束。

## Attempt 隔离

运行前要求用户 Git worktree 干净。每轮从同一个 `base_commit` 创建 detached Git worktree：

```text
baseline commit
├── attempt-1 worktree → patch + focused/full evidence
└── attempt-2 worktree → baseline + 显式的 attempt-1 patch/失败证据
```

第一轮文件状态不会隐式进入第二轮。只有 focused 和 full pytest 都通过的 Attempt 标记为 `selected`，随后仅将 `editable_paths` 回写原仓库；所有 Attempt 失败时原仓库保持不变。非 Git 目录只为本地 fixture 兼容，会先复制到临时 Git snapshot，再以相同 worktree 流程执行。

每轮结果记录 `base_commit`、`attempt_patch`、focused/full 结果、`introduced_failures` 和 `selected`。
