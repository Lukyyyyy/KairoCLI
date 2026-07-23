"""Redaction and bounding helpers for MCP diagnostics."""

from __future__ import annotations

from typing import Any

from ..text_safety import safe_text
from ..trace import redact_sensitive_text

MAX_MCP_RESOURCE_ERROR_CHARS = 4_000


def bounded_mcp_error(value: Any) -> str:
    redacted = redact_sensitive_text(
        safe_text(value, fallback=f"{type(value).__name__} message unavailable")
    )
    if len(redacted) <= MAX_MCP_RESOURCE_ERROR_CHARS:
        return redacted
    marker = f"...[MCP error truncated: original_chars={len(redacted)}]"
    available = max(0, MAX_MCP_RESOURCE_ERROR_CHARS - len(marker) - 1)
    return f"{redacted[:available]}\n{marker}"
