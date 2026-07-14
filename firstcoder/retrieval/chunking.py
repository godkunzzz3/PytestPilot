"""Python AST chunking with deterministic line-window fallback."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from firstcoder.retrieval.models import CodeChunk


IGNORED_DIRECTORY_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "build",
    "dist",
    ".firstcoder",
    "runs",
    "generated",
}


def iter_python_files(root: str | Path) -> list[Path]:
    project = Path(root).resolve()
    files: list[Path] = []
    for path in project.rglob("*.py"):
        relative = path.relative_to(project)
        if any(_ignored_part(part) for part in relative.parts[:-1]):
            continue
        if path.is_file():
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(project).as_posix())


def chunk_python_file(
    root: str | Path,
    path: str | Path,
    *,
    repo_id: str,
    window_lines: int = 120,
    max_content_chars: int = 4_000,
) -> list[CodeChunk]:
    project = Path(root).resolve()
    source_path = Path(path).resolve()
    if project != source_path and project not in source_path.parents:
        raise ValueError("Python source path escapes repository root")
    relative = source_path.relative_to(project).as_posix()
    text = source_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines:
        lines = [""]
    try:
        tree = ast.parse(text, filename=relative)
    except SyntaxError:
        return _window_chunks(
            lines,
            repo_id=repo_id,
            path=relative,
            kind="fallback",
            symbol="<syntax-error>",
            is_test=_is_test(relative, ""),
            window_lines=window_lines,
            max_content_chars=max_content_chars,
        )

    chunks = _window_chunks(
        lines,
        repo_id=repo_id,
        path=relative,
        kind="module",
        symbol="<module>",
        is_test=_is_test(relative, ""),
        window_lines=window_lines,
        max_content_chars=max_content_chars,
    )
    parents = _parent_map(tree)
    nodes = [node for node in ast.walk(tree) if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
    nodes.sort(key=lambda node: (int(node.lineno), int(getattr(node, "end_lineno", node.lineno))))
    for node in nodes:
        start = int(node.lineno)
        end = int(getattr(node, "end_lineno", node.lineno))
        parent = parents.get(node)
        if isinstance(node, ast.ClassDef):
            kind = "class"
            symbol = node.name
        elif isinstance(parent, ast.ClassDef):
            kind = "method"
            symbol = f"{parent.name}.{node.name}"
        elif node.name.startswith("test_"):
            kind = "test_function"
            symbol = node.name
        elif isinstance(node, ast.AsyncFunctionDef):
            kind = "async_function"
            symbol = node.name
        else:
            kind = "function"
            symbol = node.name
        chunks.extend(
            _window_chunks(
                lines[start - 1 : end],
                repo_id=repo_id,
                path=relative,
                kind=kind,
                symbol=symbol,
                is_test=_is_test(relative, symbol),
                window_lines=window_lines,
                max_content_chars=max_content_chars,
                line_offset=start - 1,
            )
        )
    return chunks


def _window_chunks(
    lines: list[str],
    *,
    repo_id: str,
    path: str,
    kind: str,
    symbol: str,
    is_test: bool,
    window_lines: int,
    max_content_chars: int,
    line_offset: int = 0,
) -> list[CodeChunk]:
    size = max(1, window_lines)
    chunks: list[CodeChunk] = []
    for index, offset in enumerate(range(0, len(lines), size), start=1):
        selected = lines[offset : offset + size]
        start_line = line_offset + offset + 1
        end_line = start_line + len(selected) - 1
        full_content = "\n".join(selected)
        window_symbol = symbol if len(lines) <= size else f"{symbol}#window-{index}"
        identity = {
            "repo_id": repo_id,
            "path": path,
            "kind": kind,
            "symbol": window_symbol,
            "start_line": start_line,
            "end_line": end_line,
        }
        chunk_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:32]
        content_hash = hashlib.sha256(full_content.encode("utf-8")).hexdigest()
        bounded = full_content[: max(0, max_content_chars)]
        chunks.append(
            CodeChunk(
                chunk_id=chunk_id,
                repo_id=repo_id,
                path=path,
                language="python",
                kind=kind,
                symbol=window_symbol,
                start_line=start_line,
                end_line=end_line,
                is_test=is_test,
                content_hash=content_hash,
                bounded_content=bounded,
            )
        )
    return chunks


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _ignored_part(part: str) -> bool:
    return part in IGNORED_DIRECTORY_NAMES or part == ".venv" or part.startswith(".venv-")


def _is_test(path: str, symbol: str) -> bool:
    parts = Path(path).parts
    return "tests" in parts or Path(path).name.startswith("test_") or symbol.startswith("test_")
