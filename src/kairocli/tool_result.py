from __future__ import annotations

from .json_boundary import decode_strict_json

MAX_TOOL_RESULT_JSON_BYTES = 800_000


def is_failed_tool_text(value: str) -> bool:
    """Classify tool text once for execution policy and every presentation surface."""

    try:
        parsed = decode_strict_json(
            value,
            max_bytes=MAX_TOOL_RESULT_JSON_BYTES,
            max_depth=32,
            max_nodes=100_000,
        )
    except (RecursionError, TypeError, UnicodeError, ValueError):
        if value.lstrip().startswith(("{", "[")):
            return True
        return value.casefold().startswith(
            ("error:", "tool execution failed", "mcp tool returned an error:")
        )
    if not isinstance(parsed, dict):
        return False
    error = parsed.get("error")
    return bool(
        ("error" in parsed and error is not None and error is not False)
        or parsed.get("isError")
        or parsed.get("approval_denied")
        or parsed.get("policy_denied")
    )
