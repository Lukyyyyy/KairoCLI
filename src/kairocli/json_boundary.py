from __future__ import annotations

import json
from typing import Any


def decode_strict_json(
    raw: str | bytes | bytearray,
    *,
    max_bytes: int,
    max_depth: int,
    max_nodes: int,
) -> Any:
    if isinstance(raw, str):
        encoded_size = len(raw.encode("utf-8"))
    else:
        encoded_size = len(raw)
    if encoded_size > max_bytes:
        raise ValueError(f"JSON exceeds the {max_bytes} byte limit")
    payload = json.loads(
        raw,
        object_pairs_hook=_object_without_duplicates,
        parse_constant=_reject_constant,
    )
    _validate_tree(payload, max_depth=max_depth, max_nodes=max_nodes)
    return payload


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError(f"Invalid JSON constant: {value}")


def _validate_tree(root: Any, *, max_depth: int, max_nodes: int) -> None:
    nodes = 0
    stack: list[tuple[Any, int]] = [(root, 1)]
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > max_nodes:
            raise ValueError("JSON is too complex")
        if depth > max_depth:
            raise ValueError("JSON is too deeply nested")
        if isinstance(value, dict):
            stack.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            stack.extend((item, depth + 1) for item in value)
