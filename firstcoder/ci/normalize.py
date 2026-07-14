"""Normalization helpers for unstable pytest console text."""

from __future__ import annotations

import re


_ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_ADDRESS_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
_ISO_TIMESTAMP_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}[T ][0-9:.+-]+Z?\b")
_TMP_PREFIX_RE = re.compile(r"(?:(?:/private)?/tmp|/var/folders)/[^\s'\"]+?/(?=(?:src|tests|test|firstcoder)/)")
_DURATION_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:ms|s|seconds?)\b", re.IGNORECASE)


def clean_pytest_text(value: str) -> str:
    cleaned = _ANSI_RE.sub("", value).replace("\x00", "")
    return cleaned.replace("\r\n", "\n").replace("\r", "\n")


def normalize_path(path: str) -> str:
    normalized = path.strip().strip("'\"").replace("\\", "/")
    while "//" in normalized and not normalized.startswith("//"):
        normalized = normalized.replace("//", "/")
    return normalized


def normalize_fingerprint_message(value: str) -> str:
    normalized = clean_pytest_text(value)
    normalized = _ADDRESS_RE.sub("<address>", normalized)
    normalized = _ISO_TIMESTAMP_RE.sub("<timestamp>", normalized)
    normalized = _TMP_PREFIX_RE.sub("<tmp>/", normalized)
    normalized = _DURATION_RE.sub("<duration>", normalized)
    return " ".join(normalized.split())


def normalize_fingerprint_path(value: str) -> str:
    normalized = normalize_path(value)
    normalized = _TMP_PREFIX_RE.sub("<tmp>/", normalized)
    return normalized
