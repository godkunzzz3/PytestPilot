"""Command execution backends for trusted and untrusted repositories."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from firstcoder.utils.subprocess import CommandResult


_DEFAULT_ENV_ALLOWLIST = (
    "LANG",
    "LC_ALL",
    "TZ",
)


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """Hard and host-side limits applied to one command."""

    timeout_seconds: float = 300.0
    cpu_count: float = 1.0
    memory_mb: int = 1024
    pids: int = 128
    max_output_chars: int = 100_000
    max_file_size_mb: int = 64
    tmpfs_mb: int = 256

    def __post_init__(self) -> None:
        for name in (
            "timeout_seconds",
            "cpu_count",
            "memory_mb",
            "pids",
            "max_output_chars",
            "max_file_size_mb",
            "tmpfs_mb",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")


class ExecutionBackend(Protocol):
    """Runs an argv command inside an explicit execution boundary."""

    def run(
        self,
        command: list[str],
        cwd: Path,
        limits: ResourceLimits,
    ) -> CommandResult:
        ...


class LocalProcessBackend:
    """Host process backend for trusted projects and unit tests only."""

    trusted_only = True

    def __init__(
        self,
        *,
        environment: dict[str, str] | None = None,
        env_allowlist: tuple[str, ...] = _DEFAULT_ENV_ALLOWLIST,
    ) -> None:
        source = os.environ if environment is None else environment
        self.environment = _whitelisted_environment(source, env_allowlist)
        self.environment["PYTHONDONTWRITEBYTECODE"] = "1"

    def run(self, command: list[str], cwd: Path, limits: ResourceLimits) -> CommandResult:
        return _run_bounded(
            command,
            cwd=cwd.resolve(),
            env=self.environment,
            limits=limits,
            local_resource_limits=True,
        )


class DockerSandboxBackend:
    """Docker boundary for running tests from an isolated attempt worktree."""

    trusted_only = False

    def __init__(
        self,
        *,
        image: str = "python:3.11-slim",
        docker_binary: str = "docker",
        environment: dict[str, str] | None = None,
        env_allowlist: tuple[str, ...] = _DEFAULT_ENV_ALLOWLIST,
        uid: int = 65532,
        gid: int = 65532,
    ) -> None:
        if not image.strip():
            raise ValueError("Docker image cannot be empty")
        self.image = image
        self.docker_binary = docker_binary
        source = os.environ if environment is None else environment
        self.environment = _whitelisted_environment(source, env_allowlist)
        self.environment["PYTHONDONTWRITEBYTECODE"] = "1"
        self.uid = uid
        self.gid = gid

    def run(self, command: list[str], cwd: Path, limits: ResourceLimits) -> CommandResult:
        root = cwd.resolve()
        container_command = _container_command(command)
        docker_command = [
            self.docker_binary,
            "run",
            "--rm",
            "--pull=never",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            f"--cpus={limits.cpu_count}",
            f"--memory={limits.memory_mb}m",
            "--memory-swap",
            f"{limits.memory_mb}m",
            f"--pids-limit={limits.pids}",
            f"--user={self.uid}:{self.gid}",
            f"--ulimit=fsize={limits.max_file_size_mb * 1024}:{limits.max_file_size_mb * 1024}",
            f"--tmpfs=/tmp:rw,noexec,nosuid,nodev,size={limits.tmpfs_mb}m",
            "--mount",
            f"type=bind,src={root},dst=/workspace,rw",
            "--workdir=/workspace",
        ]
        for key, value in sorted(self.environment.items()):
            docker_command.extend(["--env", f"{key}={value}"])
        docker_command.extend([self.image, *container_command])
        return _run_bounded(
            docker_command,
            cwd=root,
            env=_docker_client_environment(os.environ),
            limits=limits,
            local_resource_limits=False,
        )


def _container_command(command: list[str]) -> list[str]:
    if not command:
        raise ValueError("command cannot be empty")
    converted = list(command)
    if Path(converted[0]).resolve() == Path(sys.executable).resolve():
        converted[0] = "python"
    return converted


def _whitelisted_environment(source: dict[str, str], allowlist: tuple[str, ...]) -> dict[str, str]:
    return {key: source[key] for key in allowlist if key in source}


def _docker_client_environment(source: dict[str, str]) -> dict[str, str]:
    """Keep only variables needed to locate and talk to the Docker daemon."""

    keys = ("PATH", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG", "HOME")
    return {key: source[key] for key in keys if key in source}


def _run_bounded(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    limits: ResourceLimits,
    local_resource_limits: bool,
) -> CommandResult:
    """Stream output into bounded buffers and kill the whole process group on timeout."""

    preexec_fn = _local_preexec(limits) if local_resource_limits and os.name == "posix" else None
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=preexec_fn is None,
            preexec_fn=preexec_fn,
        )
    except OSError as exc:
        return CommandResult(
            exit_code=-1,
            stdout="",
            stderr="",
            stdout_truncated=False,
            stderr_truncated=False,
            ok=False,
            error=f"Command execution failed: {exc}",
        )

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    stdout_state = [0, False]
    stderr_state = [0, False]
    readers = [
        threading.Thread(
            target=_drain_stream,
            args=(process.stdout, stdout_parts, stdout_state, limits.max_output_chars),
            daemon=True,
        ),
        threading.Thread(
            target=_drain_stream,
            args=(process.stderr, stderr_parts, stderr_state, limits.max_output_chars),
            daemon=True,
        ),
    ]
    for reader in readers:
        reader.start()

    error: str | None = None
    try:
        process.wait(timeout=limits.timeout_seconds)
    except subprocess.TimeoutExpired:
        error = "Command execution timed out"
        _kill_process_group(process)
    for reader in readers:
        reader.join(timeout=1)

    stdout = "".join(stdout_parts)
    stderr = "".join(stderr_parts)
    returncode = process.returncode if process.returncode is not None else -1
    return CommandResult(
        exit_code=returncode,
        stdout=stdout,
        stderr=stderr,
        stdout_truncated=bool(stdout_state[1]),
        stderr_truncated=bool(stderr_state[1]),
        ok=returncode == 0 and error is None,
        error=error,
    )


def _drain_stream(stream, parts: list[str], state: list[int | bool], limit: int) -> None:
    if stream is None:
        return
    for chunk in iter(lambda: stream.read(4096), ""):
        remaining = limit - int(state[0])
        if remaining > 0:
            kept = chunk[:remaining]
            parts.append(kept)
            state[0] = int(state[0]) + len(kept)
        if len(chunk) > max(remaining, 0):
            state[1] = True
    stream.close()


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()
    process.wait(timeout=5)


def _local_preexec(limits: ResourceLimits):
    def apply_limits() -> None:
        import resource

        os.setsid()
        cpu_seconds = max(1, int(limits.timeout_seconds))
        _try_setrlimit(resource, resource.RLIMIT_CPU, cpu_seconds)
        memory_bytes = limits.memory_mb * 1024 * 1024
        _try_setrlimit(resource, resource.RLIMIT_AS, memory_bytes)
        _try_setrlimit(resource, resource.RLIMIT_NPROC, limits.pids)
        file_bytes = limits.max_file_size_mb * 1024 * 1024
        _try_setrlimit(resource, resource.RLIMIT_FSIZE, file_bytes)

    return apply_limits


def _try_setrlimit(resource_module, resource_kind: int, requested: int) -> None:
    """Apply the tightest supported limit without making trusted-local startup brittle."""

    try:
        _, hard = resource_module.getrlimit(resource_kind)
        value = requested if hard == resource_module.RLIM_INFINITY else min(requested, hard)
        resource_module.setrlimit(resource_kind, (value, value))
    except (OSError, ValueError):
        return
