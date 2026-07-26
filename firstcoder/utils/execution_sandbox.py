"""Execution sandbox for local subprocess tools."""

from __future__ import annotations

import os
import shlex
from pathlib import Path

from firstcoder.agent.cancellation import current_cancellation_token
from firstcoder.execution import ExecutionBackend, ResourceLimits
from firstcoder.utils.sandbox_access import SandboxAccess
from firstcoder.utils.sandbox import PathSandbox
from firstcoder.utils.subprocess import CommandResult, run_command


_SENSITIVE_ENV_KEYWORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "COOKIE")


class ExecutionSandbox:
    """Small subprocess boundary layered above PathSandbox.

    This is intentionally not a policy engine. PermissionManager decides whether
    a command may run; this class constrains how approved subprocesses run.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        access: SandboxAccess | None = None,
        backend: ExecutionBackend | None = None,
        resource_limits: ResourceLimits | None = None,
    ) -> None:
        self.path_sandbox = PathSandbox(root, access=access)
        self.root = self.path_sandbox.root
        self.backend = backend
        self.resource_limits = resource_limits

    def resolve_cwd(self, cwd: str | Path | None = ".") -> Path:
        return self.path_sandbox.resolve_validated(cwd, expect="dir")

    def relative(self, path: str | Path) -> str:
        return self.path_sandbox.relative(path)

    def build_env(self, extra_env: dict[str, str] | None = None) -> dict[str, str]:
        env = {key: value for key, value in os.environ.items() if not _is_sensitive_env_key(key)}
        for key, value in (extra_env or {}).items():
            if not _is_sensitive_env_key(key):
                env[str(key)] = str(value)
        return env

    def run(
        self,
        command: list[str] | str,
        *,
        cwd: str | Path | None = ".",
        timeout_seconds: int = 30,
        max_output_chars: int = 20000,
        shell: bool = False,
        extra_env: dict[str, str] | None = None,
    ) -> CommandResult:
        try:
            workdir = self.resolve_cwd(cwd)
        except ValueError as exc:
            return CommandResult(
                exit_code=-1,
                stdout="",
                stderr="",
                stdout_truncated=False,
                stderr_truncated=False,
                ok=False,
                error=str(exc),
            )
        if self.backend is not None:
            args = ["/bin/sh", "-lc", command] if isinstance(command, str) and shell else (
                shlex.split(command) if isinstance(command, str) else list(command)
            )
            base = self.resource_limits or ResourceLimits()
            limits = ResourceLimits(
                timeout_seconds=timeout_seconds,
                cpu_count=base.cpu_count,
                memory_mb=base.memory_mb,
                pids=base.pids,
                max_output_chars=max_output_chars,
                max_file_size_mb=base.max_file_size_mb,
                tmpfs_mb=base.tmpfs_mb,
            )
            return self.backend.run(args, workdir, limits)
        return run_command(
            command,
            cwd=workdir,
            timeout_seconds=timeout_seconds,
            max_output_chars=max_output_chars,
            shell=shell,
            env=self.build_env(extra_env),
            cancellation_token=current_cancellation_token(),
        )


def _is_sensitive_env_key(key: str) -> bool:
    normalized = key.upper()
    return any(keyword in normalized for keyword in _SENSITIVE_ENV_KEYWORDS)
