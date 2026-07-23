"""Conservative JSON Schema validation for tool arguments."""

from __future__ import annotations

import math
import re
import shlex
from pathlib import Path
from typing import Any

from ..policy import PathGuard, PolicyDenied
from ..text_safety import safe_text


class ToolArgumentsError(ValueError):
    """Raised when tool arguments do not satisfy their declared schema."""


def _validate_tool_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> None:
    try:
        _validate_schema_definition(schema, "$", 0, [0])
        _validate_schema_value(schema, arguments, "$", 0, [0])
    except ToolArgumentsError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise ToolArgumentsError("Tool parameter schema is invalid") from exc


def _validate_schema_definition(
    schema: Any,
    path: str,
    depth: int,
    visited: list[int],
) -> None:
    visited[0] += 1
    if depth > 16 or visited[0] > 1_000:
        raise ToolArgumentsError("Tool parameter schema exceeds complexity limits")
    if not isinstance(schema, dict):
        raise ToolArgumentsError(f"Tool parameter schema at {path} must be an object")

    expected = schema.get("type")
    supported_types = {"object", "array", "string", "integer", "number", "boolean", "null"}
    if expected is not None:
        declared = [expected] if isinstance(expected, str) else expected
        if (
            not isinstance(declared, list)
            or not declared
            or not all(isinstance(item, str) and item in supported_types for item in declared)
            or len(set(declared)) != len(declared)
        ):
            raise ToolArgumentsError(f"Tool parameter schema at {path} has invalid type")

    enum = schema.get("enum")
    if enum is not None and (not isinstance(enum, list) or not enum):
        raise ToolArgumentsError(f"Tool parameter schema at {path} has invalid enum")

    for minimum_key, maximum_key in (
        ("minLength", "maxLength"),
        ("minItems", "maxItems"),
    ):
        minimum = schema.get(minimum_key)
        maximum = schema.get(maximum_key)
        for key, constraint in ((minimum_key, minimum), (maximum_key, maximum)):
            if constraint is not None and (type(constraint) is not int or constraint < 0):
                raise ToolArgumentsError(f"Tool parameter schema at {path} has invalid {key}")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ToolArgumentsError(
                f"Tool parameter schema at {path} has inconsistent {minimum_key}/{maximum_key}"
            )

    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    for key, constraint in (("minimum", minimum), ("maximum", maximum)):
        if constraint is not None and (
            not isinstance(constraint, (int, float))
            or isinstance(constraint, bool)
            or (isinstance(constraint, float) and not math.isfinite(constraint))
        ):
            raise ToolArgumentsError(f"Tool parameter schema at {path} has invalid {key}")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ToolArgumentsError(
            f"Tool parameter schema at {path} has inconsistent minimum/maximum"
        )

    required = schema.get("required", [])
    if (
        not isinstance(required, list)
        or not all(isinstance(item, str) for item in required)
        or len(set(required)) != len(required)
    ):
        raise ToolArgumentsError(f"Tool parameter schema at {path} has invalid required fields")

    properties = schema.get("properties", {})
    if not isinstance(properties, dict) or not all(isinstance(name, str) for name in properties):
        raise ToolArgumentsError(f"Tool parameter schema at {path} has invalid properties")
    for name, child in properties.items():
        _validate_schema_definition(child, f"{path}.properties.{name}", depth + 1, visited)

    items = schema.get("items")
    if items is not None:
        _validate_schema_definition(items, f"{path}.items", depth + 1, visited)
    additional = schema.get("additionalProperties", True)
    if not isinstance(additional, (bool, dict)):
        raise ToolArgumentsError(
            f"Tool parameter schema at {path} has invalid additionalProperties"
        )
    if isinstance(additional, dict):
        _validate_schema_definition(additional, f"{path}.additionalProperties", depth + 1, visited)


def _validate_schema_value(
    schema: Any,
    value: Any,
    path: str,
    depth: int,
    visited: list[int],
) -> None:
    visited[0] += 1
    if depth > 16 or visited[0] > 1_000:
        raise ToolArgumentsError("Tool arguments exceed schema validation complexity limits")
    if not isinstance(schema, dict):
        raise ToolArgumentsError("Tool parameter schema is invalid")

    expected = schema.get("type")
    if isinstance(expected, str):
        allowed_types = [expected]
    elif isinstance(expected, list) and all(isinstance(item, str) for item in expected):
        allowed_types = list(expected)
    elif expected is None:
        allowed_types = []
    else:
        raise ToolArgumentsError("Tool parameter schema has an invalid type constraint")
    if allowed_types and not any(_schema_type_matches(item, value) for item in allowed_types):
        rendered = " or ".join(allowed_types)
        raise ToolArgumentsError(f"{path} must be {rendered}")

    enum = schema.get("enum")
    if enum is not None:
        if not isinstance(enum, list):
            raise ToolArgumentsError("Tool parameter schema has an invalid enum")
        if not any(type(value) is type(item) and value == item for item in enum):
            raise ToolArgumentsError(f"{path} is not one of the allowed values")

    if isinstance(value, str):
        minimum = schema.get("minLength")
        maximum = schema.get("maxLength")
        if minimum is not None and len(value) < minimum:
            raise ToolArgumentsError(f"{path} is shorter than {minimum} characters")
        if maximum is not None and len(value) > maximum:
            raise ToolArgumentsError(f"{path} exceeds {maximum} characters")
    elif _schema_type_matches("number", value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ToolArgumentsError(f"{path} must be finite")
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and value < minimum:
            raise ToolArgumentsError(f"{path} must be at least {minimum}")
        if maximum is not None and value > maximum:
            raise ToolArgumentsError(f"{path} must be at most {maximum}")

    if isinstance(value, list):
        minimum_items = schema.get("minItems")
        maximum_items = schema.get("maxItems")
        if minimum_items is not None and len(value) < minimum_items:
            raise ToolArgumentsError(f"{path} must contain at least {minimum_items} items")
        if maximum_items is not None and len(value) > maximum_items:
            raise ToolArgumentsError(f"{path} must contain at most {maximum_items} items")
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(value):
                _validate_schema_value(item_schema, item, f"{path}[{index}]", depth + 1, visited)

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise ToolArgumentsError("Tool parameter schema has invalid properties")
        required = schema.get("required", [])
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            raise ToolArgumentsError("Tool parameter schema has invalid required fields")
        for name in required:
            if name not in value:
                raise ToolArgumentsError(f"{path}.{name} is required")
        additional = schema.get("additionalProperties", True)
        if not isinstance(additional, (bool, dict)):
            raise ToolArgumentsError("Tool parameter schema has invalid additionalProperties")
        for name, item in value.items():
            child_schema = properties.get(name)
            if child_schema is not None:
                _validate_schema_value(child_schema, item, f"{path}.{name}", depth + 1, visited)
            elif additional is False:
                raise ToolArgumentsError(f"{path}.{name} is not an allowed property")
            elif isinstance(additional, dict):
                _validate_schema_value(additional, item, f"{path}.{name}", depth + 1, visited)


def _schema_type_matches(expected: str, value: Any) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    raise ToolArgumentsError(f"Tool parameter schema uses unsupported type: {expected}")


def _validate_patch(patch: str, guard: PathGuard) -> list[str]:
    if not patch.strip():
        raise PolicyDenied("apply_patch requires a non-empty unified diff")
    if "\x00" in patch:
        raise PolicyDenied("apply_patch rejects NUL bytes")
    forbidden_prefixes = (
        "GIT binary patch",
        "Binary files ",
        "rename from ",
        "rename to ",
        "copy from ",
        "copy to ",
        "old mode ",
        "new mode ",
        "similarity index ",
        "dissimilarity index ",
    )
    paths: list[str] = []
    current_path: str | None = None
    old_marker_seen = False
    new_marker_seen = False
    old_marker_path: str | None = None
    new_marker_path: str | None = None
    in_hunks = False

    def finish_header() -> None:
        if current_path is None:
            return
        if not old_marker_seen or not new_marker_seen:
            raise PolicyDenied("Patch file is missing ---/+++ path markers")
        if old_marker_path is None and new_marker_path is None:
            raise PolicyDenied("Patch cannot use /dev/null for both path markers")
        if old_marker_path not in {None, current_path}:
            raise PolicyDenied("Patch --- path does not match diff --git header")
        if new_marker_path not in {None, current_path}:
            raise PolicyDenied("Patch +++ path does not match diff --git header")

    for line in patch.splitlines():
        if line.startswith(forbidden_prefixes):
            raise PolicyDenied(f"apply_patch rejects directive: {line.split(' ', 2)[0]}")
        if re.match(r"(?:new file|deleted file) mode (?:120000|160000)$", line) or re.match(
            r"index \S+ (?:120000|160000)$", line
        ):
            raise PolicyDenied("apply_patch rejects symlink and submodule changes")
        if line.startswith("diff --git "):
            finish_header()
            current_path = None
            old_marker_seen = False
            new_marker_seen = False
            old_marker_path = None
            new_marker_path = None
            in_hunks = False
            try:
                parts = shlex.split(line)
            except ValueError as exc:
                raise PolicyDenied("Invalid diff header: " + safe_text(exc)) from exc
            if len(parts) != 4:
                raise PolicyDenied("Invalid diff --git header")
            left = _patch_path(parts[2], "a/", guard)
            right = _patch_path(parts[3], "b/", guard)
            if left != right:
                raise PolicyDenied("apply_patch rejects rename/copy paths")
            current_path = left
            if left not in paths:
                paths.append(left)
            continue
        if current_path is None or in_hunks:
            continue
        if line.startswith("@@"):
            finish_header()
            in_hunks = True
            continue
        if line.startswith("--- "):
            if old_marker_seen:
                raise PolicyDenied("Patch file has duplicate --- path markers")
            old_marker_seen = True
            old_marker_path = _patch_marker_path(line, "---", "a/", guard)
            continue
        if line.startswith("+++ "):
            if new_marker_seen:
                raise PolicyDenied("Patch file has duplicate +++ path markers")
            new_marker_seen = True
            new_marker_path = _patch_marker_path(line, "+++", "b/", guard)
    finish_header()
    if not paths:
        raise PolicyDenied("apply_patch requires diff --git headers")
    if len(paths) > 100:
        raise PolicyDenied("apply_patch affects more than 100 files")
    return paths


def _patch_marker_path(line: str, marker: str, prefix: str, guard: PathGuard) -> str | None:
    try:
        parts = shlex.split(line)
    except ValueError as exc:
        raise PolicyDenied(f"Invalid {marker} path marker: " + safe_text(exc)) from exc
    if len(parts) != 2 or parts[0] != marker:
        raise PolicyDenied(f"Invalid {marker} path marker")
    if parts[1] == "/dev/null":
        return None
    return _patch_path(parts[1], prefix, guard)


def _patch_path(value: str, prefix: str, guard: PathGuard) -> str:
    if not value.startswith(prefix):
        raise PolicyDenied(f"Patch path must start with {prefix}")
    relative = value[len(prefix) :]
    if not relative or relative == "/dev/null":
        raise PolicyDenied("Invalid patch path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.parts[0] == ".git":
        raise PolicyDenied(f"Unsafe patch path: {relative}")
    resolved = guard.resolve_for_write(relative)
    return resolved.relative_to(guard.workspace).as_posix()
