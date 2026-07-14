"""Stable failure fingerprints for grouping repeated pytest failures."""

from __future__ import annotations

import hashlib
import json

from firstcoder.ci.normalize import normalize_fingerprint_message, normalize_fingerprint_path


def failure_fingerprint(
    *,
    node_id: str,
    phase: str,
    exception_type: str | None,
    message: str,
    source_path: str | None,
) -> str:
    payload = {
        "node_id": normalize_fingerprint_path(node_id),
        "phase": phase,
        "exception_type": exception_type or "unknown",
        "message": normalize_fingerprint_message(message),
        "source_path": normalize_fingerprint_path(source_path or ""),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]
