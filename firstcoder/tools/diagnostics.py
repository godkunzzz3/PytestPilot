"""`diagnostics` 工具。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from firstcoder.ci.pytest_parser import parse_pytest_output
from firstcoder.execution import ExecutionBackend, ResourceLimits
from firstcoder.tools.types import Tool, ToolResult, make_error_result, make_text_result
from firstcoder.utils.introspection import tool_from_function
from firstcoder.utils.execution_sandbox import ExecutionSandbox
from firstcoder.utils.sandbox_access import SandboxAccess


def create_diagnostics_tool(
    root: str | Path,
    *,
    access: SandboxAccess | None = None,
    execution_backend: ExecutionBackend | None = None,
    resource_limits: ResourceLimits | None = None,
) -> Tool:
    """创建项目诊断工具。"""

    sandbox = ExecutionSandbox(
        root,
        access=access,
        backend=execution_backend,
        resource_limits=resource_limits,
    )

    def diagnostics(command: str = "python -m pytest -q", timeout_seconds: int = 120, max_output_chars: int = 20000) -> ToolResult:
        """运行项目诊断命令，适合测试、lint、类型检查。"""

        if timeout_seconds <= 0:
            return make_error_result("diagnostics", "timeout_seconds 必须大于 0")
        if max_output_chars <= 0:
            return make_error_result("diagnostics", "max_output_chars 必须大于 0")

        normalized_command = (
            command
            if execution_backend is not None
            else command.replace("python", sys.executable, 1) if command.startswith("python ") else command
        )
        result = sandbox.run(
            normalized_command,
            cwd=".",
            timeout_seconds=timeout_seconds,
            max_output_chars=max_output_chars,
            shell=True,
        )

        data = {
            "command": command,
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "truncated": result.stdout_truncated or result.stderr_truncated,
        }

        combined_output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        if _is_pytest_command(command):
            report = parse_pytest_output(
                combined_output,
                command=command,
                exit_code=result.exit_code,
                max_raw_chars=min(max_output_chars, 20_000),
            ).to_dict()
            report["unparsed"] = bool(result.exit_code and not report["failures"] and report["status"] == "unknown")
            data["pytest_evidence"] = report
            data["output_tail"] = combined_output[-4000:].rstrip()

        if result.error:
            return make_error_result("diagnostics", result.error, **data)
        if not result.ok:
            summary = _failure_summary(data.get("pytest_evidence"))
            return make_error_result(
                "diagnostics",
                f"诊断命令退出码为 {result.exit_code}。{summary}".rstrip("。"),
                **data,
            )

        content = (result.stdout or result.stderr).strip() or "诊断通过。"
        return make_text_result("diagnostics", content, **data)

    return tool_from_function(diagnostics)


def _is_pytest_command(command: str) -> bool:
    parts = command.strip().split()
    return "pytest" in parts or any(part.endswith("/pytest") for part in parts)


def _failure_summary(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    failures = value.get("failures")
    if isinstance(failures, list) and failures and isinstance(failures[0], dict):
        failure = failures[0]
        node_id = str(failure.get("node_id") or "unknown")
        exception = str(failure.get("exception_type") or failure.get("failure_kind") or "failure")
        message = str(failure.get("message") or "")[:600]
        return f"pytest {value.get('status')}: {node_id} {exception}: {message}"
    return f"pytest {value.get('status', 'unparsed')}: unparsed output"
