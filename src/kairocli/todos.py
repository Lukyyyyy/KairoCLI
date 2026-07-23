from __future__ import annotations

import asyncio
import builtins
import json
from pathlib import Path
from typing import Any

from .sessions import MAX_TODOS, SessionStore, TodoItem, TodoStatus
from .tools import ToolDefinition, ToolRegistry


class SessionTodoController:
    """Bind structured todos to whichever local conversation is currently active."""

    def __init__(self, store: SessionStore, workspace: Path) -> None:
        self.store = store
        self.workspace = workspace.resolve()
        self.session_id: str | None = None

    def attach(self, session_id: str) -> None:
        self.session_id = session_id

    def register(self, tools: ToolRegistry) -> None:
        tools.register(
            ToolDefinition(
                "read_todos",
                "Read the structured todo list for the active session. Use for a resumed "
                "multi-step task before changing its plan.",
                {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                self._read_tool,
            )
        )
        tools.register(
            ToolDefinition(
                "update_todos",
                "Atomically replace the active session's todo list. Use for substantive "
                "multi-step work, keep items concrete, and mark progress as it changes. "
                "There may be at most one in_progress item. Do not use for trivial tasks.",
                {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "maxItems": MAX_TODOS,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {
                                        "type": "string",
                                        "pattern": "^todo_[0-9a-f]{12}$",
                                    },
                                    "content": {
                                        "type": "string",
                                        "minLength": 1,
                                        "maxLength": 500,
                                    },
                                    "status": {
                                        "type": "string",
                                        "enum": [
                                            TodoStatus.PENDING.value,
                                            TodoStatus.IN_PROGRESS.value,
                                            TodoStatus.COMPLETED.value,
                                        ],
                                    },
                                },
                                "required": ["content", "status"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["items"],
                    "additionalProperties": False,
                },
                self._update_tool,
            )
        )

    async def list(self) -> builtins.list[TodoItem]:
        session_id = self._require_session()
        return await asyncio.to_thread(self.store.list_todos, session_id, self.workspace)

    async def replace(self, items: builtins.list[dict[str, Any]]) -> builtins.list[TodoItem]:
        session_id = self._require_session()
        return await asyncio.to_thread(self.store.replace_todos, session_id, self.workspace, items)

    async def command(self, payload: str | None) -> str:
        operation, _, argument = (payload or "list").strip().partition(" ")
        operation = operation.casefold() or "list"
        if operation in {"list", "status"}:
            return format_todos(await self.list())
        current = await self.list()
        if operation == "add":
            if not argument.strip():
                return "Usage: /todo add <description>"
            raw = [_as_dict(item) for item in current]
            raw.append({"content": argument.strip(), "status": "pending"})
            return format_todos(await self.replace(raw))
        if operation in {"start", "done", "reopen", "remove"}:
            identifier = argument.strip()
            if not identifier:
                return f"Usage: /todo {operation} <TODO_ID>"
            if not any(item.id == identifier for item in current):
                return f"Todo not found: {identifier}"
            if operation == "remove":
                updated = [_as_dict(item) for item in current if item.id != identifier]
            else:
                target_status = {
                    "start": TodoStatus.IN_PROGRESS,
                    "done": TodoStatus.COMPLETED,
                    "reopen": TodoStatus.PENDING,
                }[operation]
                updated = []
                for item in current:
                    status = item.status
                    if operation == "start" and status == TodoStatus.IN_PROGRESS:
                        status = TodoStatus.PENDING
                    if item.id == identifier:
                        status = target_status
                    updated.append(_as_dict(item, status))
            return format_todos(await self.replace(updated))
        if operation == "clear":
            target = argument.strip().casefold()
            if target == "completed":
                retained = [
                    _as_dict(item) for item in current if item.status != TodoStatus.COMPLETED
                ]
            elif target == "all":
                retained = []
            else:
                return "Usage: /todo clear <completed|all>"
            return format_todos(await self.replace(retained))
        return (
            "Usage: /todo [list|add TEXT|start ID|done ID|reopen ID|remove ID|"
            "clear completed|clear all]"
        )

    async def _read_tool(self, _arguments: dict[str, Any]) -> str:
        return json.dumps(_payload(await self.list()), ensure_ascii=False)

    async def _update_tool(self, arguments: dict[str, Any]) -> str:
        raw = arguments.get("items")
        if not isinstance(raw, list):
            raise ValueError("items must be a list")
        return json.dumps(_payload(await self.replace(raw)), ensure_ascii=False)

    def _require_session(self) -> str:
        if self.session_id is None:
            raise ValueError("No active local session is attached")
        return self.session_id


def format_todos(items: list[TodoItem]) -> str:
    if not items:
        return "No todos in the active session."
    symbols = {
        TodoStatus.PENDING: "○",
        TodoStatus.IN_PROGRESS: "▶",
        TodoStatus.COMPLETED: "✓",
    }
    completed = sum(item.status == TodoStatus.COMPLETED for item in items)
    lines = [f"Todos ({completed}/{len(items)} completed):"]
    lines.extend(f"{symbols[item.status]} {item.id} {item.content}" for item in items)
    return "\n".join(lines)


def _as_dict(item: TodoItem, status: TodoStatus | None = None) -> dict[str, str]:
    return {
        "id": item.id,
        "content": item.content,
        "status": (status or item.status).value,
    }


def _payload(items: list[TodoItem]) -> dict[str, Any]:
    return {
        "todos": [_as_dict(item) for item in items],
        "completed": sum(item.status == TodoStatus.COMPLETED for item in items),
        "total": len(items),
    }
