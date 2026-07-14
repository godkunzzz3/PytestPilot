"""`tree` 工具。"""

from __future__ import annotations

from pathlib import Path

from firstcoder.tools.types import Tool, ToolResult, make_error_result, make_text_result
from firstcoder.utils.introspection import tool_from_function
from firstcoder.utils.sandbox import PathSandbox
from firstcoder.utils.sandbox_access import SandboxAccess


_ALWAYS_EXCLUDED_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "build",
    "dist",
    "qdrant",
    "qdrant_storage",
    "models",
    "runs",
}


def create_tree_tool(root: str | Path, *, access: SandboxAccess | None = None) -> Tool:
    """创建目录树查看工具。"""

    sandbox = PathSandbox(root, access=access)

    def tree(
        path: str = ".",
        max_depth: int = 3,
        max_entries: int = 200,
        max_output_chars: int = 12000,
        include_hidden: bool = False,
    ) -> ToolResult:
        """展示项目内目录树；适合快速了解结构。"""

        try:
            target = sandbox.resolve_validated(path, expect="dir")
        except ValueError as exc:
            return make_error_result("tree", str(exc))
        if max_depth <= 0:
            return make_error_result("tree", "max_depth 必须大于 0")
        if max_entries <= 0:
            return make_error_result("tree", "max_entries 必须大于 0")
        if max_output_chars <= 0:
            return make_error_result("tree", "max_output_chars 必须大于 0")

        lines: list[str] = []
        entries: list[str] = []
        state = {"truncated": False, "excluded": 0, "chars": 0}
        _walk_tree(
            sandbox,
            target,
            lines,
            entries,
            0,
            max_depth,
            max_entries,
            max_output_chars,
            include_hidden,
            state,
        )
        content = "\n".join(lines) if lines else "目录为空。"
        return make_text_result(
            "tree",
            content[:max_output_chars],
            entries=entries,
            truncated=bool(state["truncated"]),
            excluded_entries=int(state["excluded"]),
            include_hidden=include_hidden,
        )

    return tool_from_function(tree)


def _walk_tree(
    sandbox: PathSandbox,
    current: Path,
    lines: list[str],
    entries: list[str],
    depth: int,
    max_depth: int,
    max_entries: int,
    max_output_chars: int,
    include_hidden: bool,
    state: dict[str, int | bool],
) -> None:
    """递归构造目录树文本。"""

    if depth >= max_depth:
        return

    children = sorted(current.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
    for child in children:
        if len(entries) >= max_entries:
            state["truncated"] = True
            return

        if _excluded(child, include_hidden=include_hidden):
            state["excluded"] = int(state["excluded"]) + 1
            continue

        relative = sandbox.relative(child)
        display = f"{relative}/" if child.is_dir() else relative
        line = f"{'  ' * depth}{display}"
        added_chars = len(line) + (1 if lines else 0)
        if int(state["chars"]) + added_chars > max_output_chars:
            state["truncated"] = True
            return
        lines.append(line)
        state["chars"] = int(state["chars"]) + added_chars
        entries.append(display)

        if child.is_dir() and not child.is_symlink():
            _walk_tree(
                sandbox,
                child,
                lines,
                entries,
                depth + 1,
                max_depth,
                max_entries,
                max_output_chars,
                include_hidden,
                state,
            )
            if state["truncated"]:
                return


def _excluded(path: Path, *, include_hidden: bool) -> bool:
    name = path.name
    if name in _ALWAYS_EXCLUDED_DIRS:
        return True
    if name.startswith(".venv"):
        return True
    return not include_hidden and name.startswith(".")
