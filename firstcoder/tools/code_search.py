"""Semantic code search Tool; returned chunks are candidates, never source truth."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Protocol

from firstcoder.retrieval.models import CodeSearchHit, RetrievalUnavailableError
from firstcoder.tools.types import Tool, make_error_result, make_text_result
from firstcoder.utils.introspection import tool_from_function


class CodeSearchLike(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        path_prefix: str | None = None,
        include_tests: bool = True,
    ) -> list[CodeSearchHit]:
        ...


def create_code_search_tool(root: str | Path, *, search: CodeSearchLike) -> Tool:
    project = Path(root).resolve()

    def code_search(
        query: str,
        top_k: int = 5,
        path_prefix: str = "",
        include_tests: bool = True,
    ):
        """Search the local semantic code index for bounded candidates; read source before editing."""

        if not query.strip():
            return make_error_result("code_search", "query cannot be empty")
        if not 1 <= top_k <= 20:
            return make_error_result("code_search", "top_k must be between 1 and 20")
        normalized_prefix = path_prefix.replace("\\", "/").strip()
        if normalized_prefix:
            prefix_path = PurePosixPath(normalized_prefix)
            if prefix_path.is_absolute() or ".." in prefix_path.parts:
                return make_error_result("code_search", "path_prefix must stay inside the project")
            resolved = (project / normalized_prefix).resolve()
            if project != resolved and project not in resolved.parents:
                return make_error_result("code_search", "path_prefix must stay inside the project")
        try:
            hits = search.search(
                query,
                top_k=top_k,
                path_prefix=normalized_prefix or None,
                include_tests=include_tests,
            )
        except RetrievalUnavailableError as exc:
            return make_error_result(
                "code_search",
                f"Semantic retrieval unavailable: {exc}. Continue with grep/path/symbol/view.",
                unavailable=True,
                fallback_tools=["grep", "glob", "view", "read_multi"],
            )
        except (ValueError, OSError) as exc:
            return make_error_result("code_search", str(exc))
        payload = [hit.to_dict() for hit in hits]
        content = "\n\n".join(
            f"{hit.path}:{hit.start_line}-{hit.end_line} {hit.symbol} score={hit.score:.4f}\n"
            f"{hit.bounded_preview}"
            for hit in hits
        )
        return make_text_result(
            "code_search",
            (content or "No semantic code candidates found.")[:5_000],
            hits=payload,
            candidate_only=True,
            requires_source_read=True,
        )

    return tool_from_function(code_search)
