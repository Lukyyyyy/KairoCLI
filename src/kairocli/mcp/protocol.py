"""JSON-RPC framing, SSE parsing, and tool output conversion for MCP."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from ..image import process_base64_image
from ..models import ToolOutput
from .safety import bounded_mcp_error

MAX_MCP_PROTOCOL_JSON_DEPTH = 32
MAX_MCP_PROTOCOL_JSON_NODES = 200_000


class McpProtocolError(RuntimeError):
    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


def format_tool_result(result: dict[str, Any]) -> str:
    return format_tool_output(result).text


def format_tool_output(result: dict[str, Any]) -> ToolOutput:
    content = result.get("content", [])
    if not isinstance(content, list):
        content = []
    parts: list[str] = []
    image_urls: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            parts.append(json.dumps(item, ensure_ascii=False, default=str))
            continue
        content_type = item.get("type")
        if content_type == "text":
            parts.append(str(item.get("text", "")))
        elif content_type == "image":
            data = str(item.get("data", ""))
            mime_type = str(item.get("mimeType", "application/octet-stream"))
            try:
                processed = process_base64_image(data, mime_type)
            except ValueError as exc:
                parts.append(
                    f"[MCP image: mimeType={mime_type}, base64Length={len(data)}, "
                    f"not attached: {bounded_mcp_error(exc)}; "
                    "use a text snapshot instead]"
                )
            else:
                image_urls.append(processed.data_url())
                prompt_metadata = processed.prompt_metadata()
                parts.append(
                    f"[MCP image: mimeType={processed.media_type}, "
                    f"base64Length={len(processed.data)}, {processed.metadata()}; "
                    "attached for the next model turn]"
                    + (f"\n{prompt_metadata}" if prompt_metadata else "")
                )
        elif content_type in {"resource", "resource_link"}:
            resource = item.get("resource", item)
            uri = resource.get("uri", "unknown") if isinstance(resource, dict) else "unknown"
            parts.append(f"[MCP resource: {uri}]")
        else:
            parts.append(json.dumps(item, ensure_ascii=False, default=str))
    if not parts:
        structured = result.get("structuredContent")
        if structured is not None:
            parts.append(json.dumps(structured, ensure_ascii=False, default=str))
        else:
            parts.append("[MCP tool returned no content]")
    text = "\n".join(part for part in parts if part)
    if result.get("isError"):
        text = f"MCP tool returned an error: {text}"
    return ToolOutput(text, tuple(image_urls))


def parse_sse_messages(value: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    data: list[str] = []

    def flush() -> None:
        if not data:
            return
        raw = "\n".join(data)
        data.clear()
        try:
            parsed = _decode_mcp_protocol_json(raw)
        except (OverflowError, RecursionError, UnicodeError, ValueError):
            return
        if isinstance(parsed, dict):
            messages.append(parsed)

    for line in value.splitlines():
        if not line:
            flush()
        elif line.startswith("data:"):
            data.append(line.removeprefix("data:").lstrip())
    flush()
    return messages


def _serialize_mcp_message(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise McpProtocolError(-32_600, "MCP request must be valid finite JSON") from exc


def _decode_mcp_protocol_json(value: str | bytes) -> Any:
    payload = json.loads(
        value,
        object_pairs_hook=_protocol_object_without_duplicates,
        parse_constant=_reject_protocol_json_constant,
    )
    _validate_mcp_protocol_json_shape(payload)
    return payload


def _protocol_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate MCP JSON key: {key}")
        result[key] = value
    return result


def _reject_protocol_json_constant(value: str) -> Any:
    raise ValueError(f"Non-standard JSON constant is not allowed: {value}")


def _validate_mcp_protocol_json_shape(value: Any) -> None:
    stack: list[tuple[Any, int]] = [(value, 0)]
    visited = 0
    while stack:
        current, depth = stack.pop()
        visited += 1
        if visited > MAX_MCP_PROTOCOL_JSON_NODES:
            raise ValueError("MCP protocol JSON exceeds the node limit")
        if depth > MAX_MCP_PROTOCOL_JSON_DEPTH:
            raise ValueError("MCP protocol JSON exceeds the nesting limit")
        children: Iterable[Any]
        if isinstance(current, dict):
            children = current.values()
        elif isinstance(current, list):
            children = current
        else:
            continue
        child_count = len(current)
        if visited + len(stack) + child_count > MAX_MCP_PROTOCOL_JSON_NODES:
            raise ValueError("MCP protocol JSON exceeds the node limit")
        stack.extend((child, depth + 1) for child in children)


def sanitize_schema(value: Any, max_description_chars: int = 1_000) -> dict[str, Any]:
    """Convert broad JSON Schema input into the conservative tool-schema subset."""
    if not isinstance(value, dict):
        return {"type": "object", "properties": {}}

    def clean(item: Any) -> Any:
        if isinstance(item, list):
            return [clean(child) for child in item]
        if not isinstance(item, dict):
            return item
        output = {
            key: clean(child)
            for key, child in item.items()
            if key not in {"$schema", "$id", "$ref"}
        }
        alternatives: list[str] = []
        for keyword in ("anyOf", "oneOf"):
            union = output.pop(keyword, None)
            if isinstance(union, list):
                kinds = [
                    str(option.get("type", option)) if isinstance(option, dict) else str(option)
                    for option in union
                ]
                alternatives.append(f"{keyword} options: {', '.join(kinds)}")
        if alternatives:
            output["type"] = "object"
            existing = str(output.get("description", "")).strip()
            note = "; ".join(alternatives)
            output["description"] = f"{existing} ({note})" if existing else note
        description = output.get("description")
        if isinstance(description, str) and len(description) > max_description_chars:
            output["description"] = description[:max_description_chars] + "..."
        return output

    cleaned = clean(value)
    if not isinstance(cleaned, dict):
        return {"type": "object", "properties": {}}
    cleaned.setdefault("type", "object")
    cleaned.setdefault("properties", {})
    return cleaned
