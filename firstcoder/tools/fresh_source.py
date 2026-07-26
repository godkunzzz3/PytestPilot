"""Runtime-enforced fresh-source revisions for mutation tools."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from firstcoder.utils.sandbox import PathSandbox


@dataclass(frozen=True, slots=True)
class SourceRevision:
    path: str
    sha256: str
    size: int
    read_at: datetime
    token: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size": self.size,
            "read_at": self.read_at.isoformat(),
            "token": self.token,
        }


class FreshSourceViolation(ValueError):
    """Raised before a mutation that lacks current, scoped source evidence."""


class FreshSourceGuard:
    """Issues short-lived revisions and validates them immediately before writes."""

    def __init__(
        self,
        root: str | Path,
        *,
        editable_paths: list[str] | tuple[str, ...],
        token_ttl_seconds: float = 900.0,
    ) -> None:
        if token_ttl_seconds <= 0:
            raise ValueError("token_ttl_seconds must be greater than zero")
        self.sandbox = PathSandbox(root)
        self.editable_paths = frozenset(self._normalize(path) for path in editable_paths)
        self.token_ttl = timedelta(seconds=token_ttl_seconds)
        self._revisions: dict[str, SourceRevision] = {}

    def issue(self, path: str | Path) -> SourceRevision:
        target = self.sandbox.resolve_validated(path, expect="file")
        relative = self.sandbox.relative(target)
        content = target.read_bytes()
        revision = SourceRevision(
            path=relative,
            sha256=hashlib.sha256(content).hexdigest(),
            size=len(content),
            read_at=datetime.now(timezone.utc),
            token=secrets.token_urlsafe(24),
        )
        self._revisions[revision.token] = revision
        return revision

    def validate_existing(self, path: str | Path, token: str) -> SourceRevision:
        relative = self._require_editable(path)
        revision = self._revisions.get(token)
        if revision is None:
            raise FreshSourceViolation("missing or unknown read_token")
        if revision.path != relative:
            raise FreshSourceViolation(
                f"read_token belongs to {revision.path}, not {relative}"
            )
        if datetime.now(timezone.utc) - revision.read_at > self.token_ttl:
            self._revisions.pop(token, None)
            raise FreshSourceViolation(f"read_token expired for {relative}")
        target = self.sandbox.resolve_validated(relative, expect="file")
        content = target.read_bytes()
        current_hash = hashlib.sha256(content).hexdigest()
        if current_hash != revision.sha256 or len(content) != revision.size:
            raise FreshSourceViolation(f"source changed after read: {relative}")
        return revision

    def validate_new(self, path: str | Path) -> str:
        relative = self._require_editable(path)
        target = self.sandbox.resolve(relative)
        if target.exists():
            raise FreshSourceViolation(f"existing file requires read_token: {relative}")
        return relative

    def consume(self, token: str) -> None:
        self._revisions.pop(token, None)

    def _require_editable(self, path: str | Path) -> str:
        relative = self._normalize(path)
        if relative not in self.editable_paths:
            raise FreshSourceViolation(f"path is outside editable_paths: {relative}")
        return relative

    def _normalize(self, path: str | Path) -> str:
        return self.sandbox.relative(self.sandbox.resolve(path))
