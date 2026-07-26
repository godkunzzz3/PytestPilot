import sys
from pathlib import Path

import firstcoder.execution as execution
from firstcoder.execution import DockerSandboxBackend, LocalProcessBackend, ResourceLimits
from firstcoder.utils.subprocess import CommandResult


def test_local_backend_filters_environment_and_bounds_output(tmp_path: Path) -> None:
    backend = LocalProcessBackend(
        environment={"LANG": "C", "API_TOKEN": "secret", "UNLISTED": "value"},
        env_allowlist=("LANG",),
    )
    result = backend.run(
        [
            sys.executable,
            "-c",
            "import os; print(os.getenv('LANG')); print(os.getenv('API_TOKEN')); print('x' * 100)",
        ],
        tmp_path,
        ResourceLimits(timeout_seconds=5, max_output_chars=20),
    )

    assert result.exit_code == 0
    assert result.stdout.startswith("C\nNone\n")
    assert len(result.stdout) == 20
    assert result.stdout_truncated is True


def test_docker_backend_applies_isolation_and_resource_limits(monkeypatch, tmp_path: Path) -> None:
    seen = {}

    def fake_run(command, *, cwd, env, limits, local_resource_limits):
        seen.update(command=command, cwd=cwd, env=env, limits=limits, local=local_resource_limits)
        return CommandResult(0, "ok", "", False, False, True)

    monkeypatch.setattr(execution, "_run_bounded", fake_run)
    backend = DockerSandboxBackend(
        image="example/pytest:locked",
        environment={"LANG": "C", "API_TOKEN": "secret"},
    )
    limits = ResourceLimits(
        timeout_seconds=12,
        cpu_count=1.5,
        memory_mb=768,
        pids=32,
        max_file_size_mb=8,
        tmpfs_mb=16,
    )

    result = backend.run(["python", "-m", "pytest", "-q"], tmp_path, limits)

    command = seen["command"]
    assert result.ok is True
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert "--cpus=1.5" in command
    assert "--memory=768m" in command
    assert "--pids-limit=32" in command
    assert "--user=65532:65532" in command
    assert "--ulimit=fsize=8192:8192" in command
    assert "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=16m" in command
    assert f"type=bind,src={tmp_path.resolve()},dst=/workspace,rw" in command
    assert "LANG=C" in command
    assert all("secret" not in part for part in command)
    assert command[-5:] == ["example/pytest:locked", "python", "-m", "pytest", "-q"]
    assert seen["local"] is False


def test_resource_limits_reject_non_positive_values() -> None:
    try:
        ResourceLimits(memory_mb=0)
    except ValueError as exc:
        assert "memory_mb" in str(exc)
    else:
        raise AssertionError("invalid resource limits must be rejected")
