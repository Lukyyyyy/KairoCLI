"""Schema builders for Kairo CLI built-in tools."""

from __future__ import annotations

from typing import Any

from .limits import (
    MAX_GREP_MAX_CHARS,
    MAX_PATCH_BYTES,
    MAX_READ_FILE_CHARS,
    MAX_READ_FILE_LINES,
)
from .shell import MAX_SHELL_COMMAND_BYTES


def _required(properties: dict[str, Any], *required: str) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(required)}


def _path_schema() -> dict[str, Any]:
    return _required(
        {
            "path": {"type": "string"},
            "offset": {"type": "integer", "minimum": 1},
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_READ_FILE_LINES,
            },
            "max_chars": {
                "type": "integer",
                "minimum": 1_000,
                "maximum": MAX_READ_FILE_CHARS,
            },
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"},
        },
        "path",
    )


def _lsp_inspect_schema() -> dict[str, Any]:
    return _required(
        {
            "path": {"type": "string"},
            "start_line": {"type": "integer", "minimum": 1},
            "end_line": {"type": "integer", "minimum": 1},
        },
        "path",
    )


def _lsp_workspace_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "max_files": {"type": "integer", "minimum": 1, "maximum": 500},
            "max_diagnostics": {
                "type": "integer",
                "minimum": 1,
                "maximum": 1_000,
            },
        },
        "additionalProperties": False,
    }


def _list_schema() -> dict[str, Any]:
    return _required(
        {
            "path": {"type": "string"},
            "offset": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": 1_000},
        },
        "path",
    )


def _write_schema() -> dict[str, Any]:
    return _required({"path": {"type": "string"}, "content": {"type": "string"}}, "path", "content")


def _glob_schema() -> dict[str, Any]:
    return _required(
        {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 200},
        },
        "pattern",
    )


def _patch_schema() -> dict[str, Any]:
    return _required(
        {
            "patch": {
                "type": "string",
                "description": (
                    "Unified Git diff with diff --git headers; binary, rename, copy and mode "
                    "changes are rejected."
                ),
                "maxLength": MAX_PATCH_BYTES,
            }
        },
        "patch",
    )


def _grep_schema() -> dict[str, Any]:
    return _required(
        {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
            "glob": {"type": "string"},
            "regex": {"type": "boolean"},
            "case_sensitive": {"type": "boolean"},
            "context_lines": {"type": "integer", "minimum": 0, "maximum": 5},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 200},
            "head_limit": {"type": "integer", "minimum": 1, "maximum": 50},
            "max_chars": {
                "type": "integer",
                "minimum": 1_000,
                "maximum": MAX_GREP_MAX_CHARS,
            },
        },
        "pattern",
    )


def _command_schema() -> dict[str, Any]:
    return _required(
        {
            "command": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_SHELL_COMMAND_BYTES,
            },
            "timeout": {"type": "number", "minimum": 0.1, "maximum": 300},
        },
        "command",
    )


def _shell_start_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"cwd": {"type": "string"}},
        "additionalProperties": False,
    }


def _shell_id_schema() -> dict[str, Any]:
    return _required(
        {
            "session_id": {
                "type": "string",
                "pattern": "^shell_[0-9a-f]{12}$",
            }
        },
        "session_id",
    )


def _shell_exec_schema() -> dict[str, Any]:
    schema = _shell_id_schema()
    schema["properties"].update(
        {
            "command": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_SHELL_COMMAND_BYTES,
            },
            "timeout": {"type": "number", "minimum": 0.1, "maximum": 300},
        }
    )
    schema["required"].append("command")
    return schema


def _create_schema() -> dict[str, Any]:
    return _required(
        {
            "path": {"type": "string", "minLength": 1, "maxLength": 1_024},
            "kind": {"type": "string", "enum": ["python", "node", "java"]},
        },
        "path",
    )


def _query_schema() -> dict[str, Any]:
    return _required({"query": {"type": "string"}}, "query")


def _code_graph_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "minLength": 1, "maxLength": 200},
        },
        "required": ["symbol"],
        "additionalProperties": False,
    }


def _web_search_schema() -> dict[str, Any]:
    return _required(
        {
            "query": {"type": "string", "minLength": 1, "maxLength": 2_000},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "query",
    )


def _url_schema() -> dict[str, Any]:
    return _required(
        {
            "url": {"type": "string"},
            "max_chars": {"type": "integer", "minimum": 1_000, "maximum": 100_000},
        },
        "url",
    )
