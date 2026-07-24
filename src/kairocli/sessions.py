from __future__ import annotations

import builtins
import html
import json
import os
import re
import sqlite3
import threading
import uuid
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from .models import Message, Role, ToolCall
from .paths import KairoPaths, reject_symlink_components

SESSION_ID_PATTERN = re.compile(r"session_[0-9a-f]{12}$")
TODO_ID_PATTERN = re.compile(r"todo_[0-9a-f]{12}$")
MAX_SESSION_MESSAGES = 1_000
MAX_SESSION_BYTES = 10 * 1024 * 1024
MAX_REASONING_CHARS = 100_000
MAX_SESSION_MESSAGE_CHARS = 1_000_000
MAX_SESSION_CONTENT_PARTS = 100
MAX_SESSION_TOOL_CALLS = 100
MAX_SESSION_TOOL_ARGUMENT_BYTES = 1024 * 1024
MAX_SESSION_COUNTER = 2**63 - 1
_EXPANDED_RESOURCE_RE = re.compile(
    r"<resource\b(?P<attributes>[^>]*?)(?:/\s*>|>.*?</resource\s*>)",
    re.IGNORECASE | re.DOTALL,
)
_EXPANDED_CONTEXT_RE = re.compile(
    r"<(?P<tag>file|directory|local-context|resource_error)\b[^>]*?"
    r"(?:/\s*>|>.*?</(?P=tag)\s*>)",
    re.IGNORECASE | re.DOTALL,
)
_RESOURCE_ATTRIBUTE_RE = re.compile(r'\b(server|uri)="([^"]*)"')
MAX_SESSION_JSON_DEPTH = 32
MAX_SESSION_JSON_NODES = 200_000
MAX_TODOS = 50
MAX_TODO_CONTENT_CHARS = 500
_SESSION_TOOL_VALUE = re.compile(r"[A-Za-z0-9_.:-]+$")
MAX_SESSION_EXPORT_BYTES = 20 * 1024 * 1024


class TodoStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class SessionConflictError(ValueError):
    """Raised when a stale process attempts to overwrite newer session messages."""


def write_session_export(paths: KairoPaths, markdown: str) -> Path:
    encoded = markdown.encode("utf-8")
    if len(encoded) > MAX_SESSION_EXPORT_BYTES:
        raise ValueError(f"Session export exceeds {MAX_SESSION_EXPORT_BYTES} bytes")
    directory = paths.export_dir
    reject_symlink_components(directory, "Session export directory")
    directory.mkdir(parents=True, exist_ok=True)
    reject_symlink_components(directory, "Session export directory")
    if os.name == "posix":
        directory.chmod(0o700)
    target = directory / (f"session-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}.md")
    if target.is_symlink():
        raise ValueError("Session export target cannot be a symlink")
    temporary = target.with_suffix(".tmp")
    if temporary.is_symlink():
        raise ValueError("Session export temporary file cannot be a symlink")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
        if os.name == "posix":
            target.chmod(0o600)
        return target
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class TodoItem:
    id: str
    content: str
    status: TodoStatus
    position: int
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class SessionMeta:
    id: str
    workspace: str
    provider: str
    model: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


@dataclass(frozen=True, slots=True)
class SessionState:
    meta: SessionMeta
    messages: tuple[Message, ...]
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    llm_calls: int = 0
    compactions: int = 0


@dataclass(frozen=True, slots=True)
class SessionSaveSnapshot:
    """A point-in-time Agent state safe to hand to a worker thread."""

    provider: str
    model: str
    messages: tuple[Message, ...]
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    llm_calls: int
    compactions: int
    sequence: int = 0

    @classmethod
    def capture(cls, agent: Any, *, sequence: int = 0) -> SessionSaveSnapshot:
        # Clone each message before crossing the thread boundary. Agent removes
        # historical image parts in place at the next turn, so a tuple alone would
        # not be a stable snapshot even though it isolates appends/replacements.
        return cls(
            provider=str(agent.llm.provider),
            model=str(agent.llm.model),
            messages=tuple(_without_images(message) for message in agent.history),
            input_tokens=int(agent.total_input_tokens),
            output_tokens=int(agent.total_output_tokens),
            cached_tokens=int(agent.total_cached_tokens),
            llm_calls=int(agent.llm_call_count),
            compactions=int(agent.compaction_count),
            sequence=sequence,
        )


class SessionStore:
    def __init__(self, database: Path) -> None:
        self.database = database
        self._message_revisions: dict[tuple[str, str], int] = {}
        # One store is shared by CLI/TUI turn and shutdown paths. Serialize its
        # revision-bearing operations while retaining cross-process optimistic CAS.
        self._state_lock = threading.RLock()
        self._next_save_sequence = 0
        self._committed_save_sequences: dict[tuple[str, str], int] = {}
        reject_symlink_components(self.database, "Session database")
        self.database.parent.mkdir(parents=True, exist_ok=True)
        reject_symlink_components(self.database.parent, "Session database")
        if os.name == "posix":
            self.database.parent.chmod(0o700)
        self._initialize()
        self._harden_database_files()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self._check_database_paths()
        connection = sqlite3.connect(self.database, timeout=10)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=10000")
            connection.execute("PRAGMA foreign_keys=ON")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
            self._harden_database_files()

    def _check_database_paths(self) -> None:
        reject_symlink_components(self.database.parent, "Session database")
        for path in self._database_files():
            if path.is_symlink():
                raise ValueError(f"Session database file cannot be a symlink: {path.name}")

    def _harden_database_files(self) -> None:
        if os.name != "posix":
            return
        for path in self._database_files():
            if path.is_symlink():
                raise ValueError(f"Session database file cannot be a symlink: {path.name}")
            if path.is_file():
                path.chmod(0o600)

    def _database_files(self) -> tuple[Path, Path, Path]:
        return (
            self.database,
            Path(str(self.database) + "-wal"),
            Path(str(self.database) + "-shm"),
        )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    workspace TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    messages_json TEXT NOT NULL,
                    message_count INTEGER NOT NULL DEFAULT 0,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    cached_tokens INTEGER NOT NULL DEFAULT 0,
                    llm_calls INTEGER NOT NULL DEFAULT 0,
                    compactions INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(sessions)").fetchall()
            }
            if "message_revision" not in columns:
                connection.execute(
                    "ALTER TABLE sessions ADD COLUMN message_revision INTEGER NOT NULL DEFAULT 0"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS sessions_workspace_updated "
                "ON sessions(workspace, updated_at DESC)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS session_todos (
                    session_id TEXT NOT NULL,
                    workspace TEXT NOT NULL,
                    id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('pending', 'in_progress', 'completed')
                    ),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, id),
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS session_todos_order "
                "ON session_todos(session_id, workspace, position)"
            )

    def create(self, workspace: Path, provider: str, model: str) -> SessionState:
        now = datetime.now(UTC).isoformat()
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (
                    id, workspace, provider, model, title, created_at, updated_at,
                    messages_json, message_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, '[]', 0)
                """,
                (
                    session_id,
                    str(workspace.resolve()),
                    provider,
                    model,
                    "New session",
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                DELETE FROM sessions
                WHERE workspace = ? AND id NOT IN (
                    SELECT id FROM sessions WHERE workspace = ?
                    ORDER BY updated_at DESC LIMIT 100
                )
                """,
                (str(workspace.resolve()), str(workspace.resolve())),
            )
        state = self.load(session_id, workspace)
        if state is None:  # pragma: no cover - defensive database invariant
            raise RuntimeError("Failed to create session")
        return state

    def save(
        self,
        session_id: str,
        workspace: Path,
        provider: str,
        model: str,
        messages: Sequence[Message],
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        llm_calls: int = 0,
        compactions: int = 0,
    ) -> None:
        with self._state_lock:
            self._next_save_sequence += 1
            self._save_locked(
                session_id,
                workspace,
                provider,
                model,
                messages,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
                llm_calls=llm_calls,
                compactions=compactions,
            )
            revision_key = (session_id, str(workspace.resolve()))
            self._committed_save_sequences[revision_key] = self._next_save_sequence

    def capture_snapshot(self, agent: Any) -> SessionSaveSnapshot:
        with self._state_lock:
            self._next_save_sequence += 1
            return SessionSaveSnapshot.capture(agent, sequence=self._next_save_sequence)

    def save_snapshot(
        self,
        session_id: str,
        workspace: Path,
        snapshot: SessionSaveSnapshot,
    ) -> None:
        _validate_session_id(session_id)
        revision_key = (session_id, str(workspace.resolve()))
        with self._state_lock:
            if snapshot.sequence <= self._committed_save_sequences.get(revision_key, -1):
                return
            self._save_locked(
                session_id,
                workspace,
                snapshot.provider,
                snapshot.model,
                snapshot.messages,
                input_tokens=snapshot.input_tokens,
                output_tokens=snapshot.output_tokens,
                cached_tokens=snapshot.cached_tokens,
                llm_calls=snapshot.llm_calls,
                compactions=snapshot.compactions,
            )
            self._committed_save_sequences[revision_key] = snapshot.sequence

    def _save_locked(
        self,
        session_id: str,
        workspace: Path,
        provider: str,
        model: str,
        messages: Sequence[Message],
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        llm_calls: int = 0,
        compactions: int = 0,
    ) -> None:
        _validate_session_id(session_id)
        serialized, normalized = _serialize_bounded(messages)
        title = _session_title(normalized)
        now = datetime.now(UTC).isoformat()
        expected_workspace = str(workspace.resolve())
        revision_key = (session_id, expected_workspace)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            expected_revision = self._message_revisions.get(revision_key)
            if expected_revision is None:
                current = connection.execute(
                    "SELECT message_revision FROM sessions WHERE id = ? AND workspace = ?",
                    (session_id, expected_workspace),
                ).fetchone()
                if current is None:
                    raise ValueError("Session does not exist in the current workspace")
                expected_revision = _session_counter(current["message_revision"])
            cursor = connection.execute(
                """
                UPDATE sessions SET
                    provider = ?, model = ?, title = ?, updated_at = ?,
                    messages_json = ?, message_count = ?, input_tokens = ?,
                    output_tokens = ?, cached_tokens = ?, llm_calls = ?, compactions = ?,
                    message_revision = message_revision + 1
                WHERE id = ? AND workspace = ? AND message_revision = ?
                """,
                (
                    provider,
                    model,
                    title,
                    now,
                    serialized,
                    len(normalized),
                    _session_counter(input_tokens),
                    _session_counter(output_tokens),
                    _session_counter(cached_tokens),
                    _session_counter(llm_calls),
                    _session_counter(compactions),
                    session_id,
                    expected_workspace,
                    expected_revision,
                ),
            )
            if cursor.rowcount != 1:
                exists = connection.execute(
                    "SELECT 1 FROM sessions WHERE id = ? AND workspace = ?",
                    (session_id, expected_workspace),
                ).fetchone()
                if exists is None:
                    raise ValueError("Session does not exist in the current workspace")
                raise SessionConflictError(
                    "Session changed in another Kairo CLI process; resume it before saving again"
                )
        self._message_revisions[revision_key] = expected_revision + 1

    def load(self, session_id: str, workspace: Path) -> SessionState | None:
        with self._state_lock:
            return self._load_locked(session_id, workspace)

    def _load_locked(self, session_id: str, workspace: Path) -> SessionState | None:
        _validate_session_id(session_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id,workspace,provider,model,title,created_at,updated_at,"
                "message_count,input_tokens,output_tokens,cached_tokens,llm_calls,compactions,"
                "message_revision,"
                "length(CAST(messages_json AS BLOB)) AS payload_bytes,"
                "CASE WHEN length(CAST(messages_json AS BLOB)) <= ? "
                "THEN messages_json ELSE NULL END AS bounded_messages_json "
                "FROM sessions WHERE id = ? AND workspace = ?",
                (MAX_SESSION_BYTES, session_id, str(workspace.resolve())),
            ).fetchone()
        if row is None:
            return None
        revision_key = (session_id, str(workspace.resolve()))
        revision = _session_counter(row["message_revision"])
        known_revision = self._message_revisions.get(revision_key)
        if known_revision is None or revision > known_revision:
            self._message_revisions[revision_key] = revision
        messages: list[Message] = []
        bounded_payload = row["bounded_messages_json"]
        if bounded_payload is not None:
            try:
                raw = json.loads(
                    str(bounded_payload),
                    parse_constant=_reject_json_constant,
                )
                _validate_session_json_shape(raw)
                messages = _repair_history(_deserialize_messages(raw))
            except (json.JSONDecodeError, OverflowError, RecursionError, TypeError, ValueError):
                messages = []
        return SessionState(
            _meta(row, len(messages)),
            tuple(messages),
            _session_counter(row["input_tokens"]),
            _session_counter(row["output_tokens"]),
            _session_counter(row["cached_tokens"]),
            _session_counter(row["llm_calls"]),
            _session_counter(row["compactions"]),
        )

    def latest(self, workspace: Path) -> SessionState | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id FROM sessions WHERE workspace = ? "
                "ORDER BY (message_count > 0) DESC, updated_at DESC LIMIT 1",
                (str(workspace.resolve()),),
            ).fetchone()
        return self.load(str(row["id"]), workspace) if row is not None else None

    def list(self, workspace: Path, limit: int = 20) -> list[SessionMeta]:
        maximum = min(max(limit, 1), 100)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM sessions WHERE workspace = ? ORDER BY updated_at DESC LIMIT ?",
                (str(workspace.resolve()), maximum),
            ).fetchall()
        return [_meta(row, int(row["message_count"])) for row in rows]

    def delete(self, session_id: str, workspace: Path) -> bool:
        with self._state_lock:
            return self._delete_locked(session_id, workspace)

    def delete_many(self, session_ids: Sequence[str], workspace: Path) -> int:
        """Atomically delete existing sessions from one workspace."""

        unique_ids = tuple(dict.fromkeys(session_ids))
        if not unique_ids:
            return 0
        if len(unique_ids) > 100:
            raise ValueError("Cannot delete more than 100 sessions at once")
        for session_id in unique_ids:
            _validate_session_id(session_id)

        expected_workspace = str(workspace.resolve())
        placeholders = ",".join("?" for _ in unique_ids)
        with self._state_lock:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                rows = connection.execute(
                    f"SELECT id FROM sessions WHERE workspace = ? AND id IN ({placeholders})",
                    (expected_workspace, *unique_ids),
                ).fetchall()
                found = {str(row["id"]) for row in rows}
                missing = [session_id for session_id in unique_ids if session_id not in found]
                if missing:
                    raise ValueError(
                        "Sessions were not found in the current workspace: " + ", ".join(missing)
                    )
                cursor = connection.execute(
                    f"DELETE FROM sessions WHERE workspace = ? AND id IN ({placeholders})",
                    (expected_workspace, *unique_ids),
                )
            self._forget_deleted_sessions(unique_ids, expected_workspace)
        return cursor.rowcount

    def delete_empty(self, workspace: Path, *, exclude_session_id: str) -> int:
        """Atomically delete empty sessions except the active session."""

        _validate_session_id(exclude_session_id)
        expected_workspace = str(workspace.resolve())
        with self._state_lock:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                rows = connection.execute(
                    "SELECT id FROM sessions WHERE workspace = ? AND message_count = 0 AND id != ?",
                    (expected_workspace, exclude_session_id),
                ).fetchall()
                deleted_ids = tuple(str(row["id"]) for row in rows)
                cursor = connection.execute(
                    "DELETE FROM sessions WHERE workspace = ? AND message_count = 0 AND id != ?",
                    (expected_workspace, exclude_session_id),
                )
            self._forget_deleted_sessions(deleted_ids, expected_workspace)
        return cursor.rowcount

    def _delete_locked(self, session_id: str, workspace: Path) -> bool:
        _validate_session_id(session_id)
        expected_workspace = str(workspace.resolve())
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM sessions WHERE id = ? AND workspace = ?",
                (session_id, expected_workspace),
            )
        if cursor.rowcount == 1:
            self._forget_deleted_sessions((session_id,), expected_workspace)
        return cursor.rowcount == 1

    def _forget_deleted_sessions(self, session_ids: Sequence[str], expected_workspace: str) -> None:
        for session_id in session_ids:
            self._message_revisions.pop((session_id, expected_workspace), None)
            self._committed_save_sequences.pop((session_id, expected_workspace), None)

    def list_todos(self, session_id: str, workspace: Path) -> builtins.list[TodoItem]:
        _validate_session_id(session_id)
        expected_workspace = str(workspace.resolve())
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM sessions WHERE id = ? AND workspace = ?",
                (session_id, expected_workspace),
            ).fetchone()
            if exists is None:
                raise ValueError("Session does not exist in the current workspace")
            rows = connection.execute(
                "SELECT * FROM session_todos WHERE session_id = ? AND workspace = ? "
                "ORDER BY position, id",
                (session_id, expected_workspace),
            ).fetchall()
        return [_todo_item(row) for row in rows]

    def replace_todos(
        self,
        session_id: str,
        workspace: Path,
        items: builtins.list[dict[str, Any]],
    ) -> builtins.list[TodoItem]:
        _validate_session_id(session_id)
        normalized = _normalize_todos(items)
        expected_workspace = str(workspace.resolve())
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM sessions WHERE id = ? AND workspace = ?",
                (session_id, expected_workspace),
            ).fetchone()
            if exists is None:
                raise ValueError("Session does not exist in the current workspace")
            old_rows = connection.execute(
                "SELECT id, created_at FROM session_todos WHERE session_id = ? AND workspace = ?",
                (session_id, expected_workspace),
            ).fetchall()
            created_by_id = {str(row["id"]): str(row["created_at"]) for row in old_rows}
            connection.execute(
                "DELETE FROM session_todos WHERE session_id = ? AND workspace = ?",
                (session_id, expected_workspace),
            )
            connection.executemany(
                """
                INSERT INTO session_todos (
                    session_id, workspace, id, position, content, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        session_id,
                        expected_workspace,
                        item["id"],
                        position,
                        item["content"],
                        item["status"],
                        created_by_id.get(item["id"], now),
                        now,
                    )
                    for position, item in enumerate(normalized)
                ],
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ? AND workspace = ?",
                (now, session_id, expected_workspace),
            )
            rows = connection.execute(
                "SELECT * FROM session_todos WHERE session_id = ? AND workspace = ? "
                "ORDER BY position, id",
                (session_id, expected_workspace),
            ).fetchall()
        return [_todo_item(row) for row in rows]


def apply_session(agent: Any, state: SessionState) -> None:
    agent.history = list(state.messages)
    agent.total_input_tokens = state.input_tokens
    agent.total_output_tokens = state.output_tokens
    agent.total_cached_tokens = state.cached_tokens
    agent.llm_call_count = state.llm_calls
    agent.compaction_count = state.compactions
    agent.last_context_tokens = agent.estimate_current_context_tokens()


def _serialize_bounded(messages: Sequence[Message]) -> tuple[str, list[Message]]:
    normalized = _repair_history([_without_images(message) for message in messages])
    normalized = normalized[-MAX_SESSION_MESSAGES:]
    normalized = _repair_history(normalized)
    while True:
        raw = [_message_to_dict(message) for message in normalized]
        _validate_session_json_shape(raw)
        serialized = json.dumps(
            raw,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        if len(serialized.encode()) <= MAX_SESSION_BYTES:
            return serialized, normalized
        if not normalized:
            raise ValueError("Session state exceeds the 10 MiB limit")
        normalized = _drop_oldest_round(normalized)


def _without_images(message: Message) -> Message:
    content = message.content
    if isinstance(content, list):
        retained = [
            dict(part)
            for part in content
            if isinstance(part, dict) and part.get("type") != "image_url"
        ]
        omitted = len(content) - len(retained)
        if omitted:
            retained.append(
                {
                    "type": "text",
                    "text": f"[Omitted {omitted} persisted image attachment(s).]",
                }
            )
        content = retained
    return Message(
        message.role,
        content,
        [ToolCall(call.id, call.name, dict(call.arguments)) for call in message.tool_calls],
        message.tool_call_id,
        (message.reasoning_content or "")[:MAX_REASONING_CHARS] or None,
    )


def _repair_history(messages: list[Message]) -> list[Message]:
    repaired: list[Message] = []
    pending: set[str] = set()
    pending_start = -1
    for message in messages:
        if not repaired and message.role != "user":
            continue
        if pending:
            if message.role != "tool" or message.tool_call_id not in pending:
                return repaired[:pending_start]
            pending.remove(str(message.tool_call_id))
            repaired.append(message)
            continue
        if message.role == "tool":
            continue
        repaired.append(message)
        if message.role == "assistant" and message.tool_calls:
            identifiers = {call.id for call in message.tool_calls if call.id}
            if len(identifiers) != len(message.tool_calls):
                repaired.pop()
                return repaired
            pending = identifiers
            pending_start = len(repaired) - 1
    if pending:
        return repaired[:pending_start]
    return repaired


def _drop_oldest_round(messages: list[Message]) -> list[Message]:
    if not messages:
        return []
    for index in range(1, len(messages)):
        if messages[index].role == "user":
            return _repair_history(messages[index:])
    return []


def _message_to_dict(message: Message) -> dict[str, Any]:
    return {
        "role": message.role,
        "content": message.content,
        "tool_calls": [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in message.tool_calls
        ],
        "tool_call_id": message.tool_call_id,
        "reasoning_content": message.reasoning_content,
    }


def _deserialize_messages(raw: Any) -> list[Message]:
    if not isinstance(raw, list):
        raise ValueError("Session messages must be a list")
    result: list[Message] = []
    for item in raw[-MAX_SESSION_MESSAGES:]:
        if not isinstance(item, dict) or item.get("role") not in {"user", "assistant", "tool"}:
            continue
        role: Role = item["role"]
        content = _safe_session_content(role, item.get("content", ""))
        raw_calls = item.get("tool_calls")
        calls: list[ToolCall] = []
        if role == "assistant" and isinstance(raw_calls, list):
            for call in raw_calls[:MAX_SESSION_TOOL_CALLS]:
                if not isinstance(call, dict):
                    continue
                identifier = str(call.get("id", ""))
                name = str(call.get("name", ""))
                arguments = call.get("arguments")
                if (
                    not identifier
                    or len(identifier) > 200
                    or not _SESSION_TOOL_VALUE.fullmatch(identifier)
                    or not name
                    or len(name) > 256
                    or not _SESSION_TOOL_VALUE.fullmatch(name)
                    or not isinstance(arguments, dict)
                    or not _bounded_json_object(arguments)
                ):
                    continue
                calls.append(ToolCall(identifier, name, arguments))
        tool_call_id = item.get("tool_call_id")
        if (
            not isinstance(tool_call_id, str)
            or not 0 < len(tool_call_id) <= 200
            or not _SESSION_TOOL_VALUE.fullmatch(tool_call_id)
        ):
            tool_call_id = None
        reasoning = item.get("reasoning_content")
        if role != "assistant" or not isinstance(reasoning, str):
            reasoning = None
        result.append(
            Message(
                role,
                content,
                calls,
                tool_call_id,
                reasoning[:MAX_REASONING_CHARS] if reasoning else None,
            )
        )
    return result


def _safe_session_content(role: str, value: Any) -> str | list[dict[str, Any]]:
    if isinstance(value, str):
        return _truncate_session_text(value)
    if role == "tool" or not isinstance(value, list):
        return _truncate_session_text(str(value) if value is not None else "")
    parts: list[dict[str, Any]] = []
    used = 0
    omitted_images = 0
    for raw_part in value[:MAX_SESSION_CONTENT_PARTS]:
        if not isinstance(raw_part, dict):
            continue
        if raw_part.get("type") == "image_url":
            omitted_images += 1
            continue
        if raw_part.get("type") != "text" or not isinstance(raw_part.get("text"), str):
            continue
        remaining = MAX_SESSION_MESSAGE_CHARS - used
        if remaining <= 0:
            break
        text = str(raw_part["text"])[:remaining]
        parts.append({"type": "text", "text": text})
        used += len(text)
    if omitted_images:
        marker = f"[Omitted {omitted_images} persisted image attachment(s).]"
        overflow = used + len(marker) - MAX_SESSION_MESSAGE_CHARS
        for part in reversed(parts):
            if overflow <= 0:
                break
            text = str(part["text"])
            trim = min(overflow, len(text))
            part["text"] = text[: len(text) - trim]
            overflow -= trim
        parts.append(
            {
                "type": "text",
                "text": marker[:MAX_SESSION_MESSAGE_CHARS],
            }
        )
    return parts


def _truncate_session_text(value: str) -> str:
    if len(value) <= MAX_SESSION_MESSAGE_CHARS:
        return value
    suffix = "\n[session message truncated]"
    return value[: MAX_SESSION_MESSAGE_CHARS - len(suffix)] + suffix


def _bounded_json_object(value: dict[str, Any]) -> bool:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (OverflowError, RecursionError, TypeError, ValueError):
        return False
    return len(encoded) <= MAX_SESSION_TOOL_ARGUMENT_BYTES


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"Non-standard JSON constant is not allowed: {value}")


def _validate_session_json_shape(value: Any) -> None:
    stack: list[tuple[Any, int]] = [(value, 0)]
    visited = 0
    while stack:
        current, depth = stack.pop()
        visited += 1
        if visited > MAX_SESSION_JSON_NODES:
            raise ValueError("Session JSON exceeds the node limit")
        if depth > MAX_SESSION_JSON_DEPTH:
            raise ValueError("Session JSON exceeds the nesting limit")
        children: Iterable[Any]
        if isinstance(current, dict):
            children = current.values()
            child_count = len(current)
        elif isinstance(current, list):
            children = current
            child_count = len(current)
        else:
            continue
        if visited + len(stack) + child_count > MAX_SESSION_JSON_NODES:
            raise ValueError("Session JSON exceeds the node limit")
        stack.extend((child, depth + 1) for child in children)


def _session_counter(value: Any) -> int:
    try:
        parsed = int(value)
    except (OverflowError, TypeError, ValueError):
        return 0
    return min(max(parsed, 0), MAX_SESSION_COUNTER)


def _session_title(messages: list[Message]) -> str:
    for message in messages:
        if message.role != "user":
            continue
        if isinstance(message.content, str):
            value = message.content
        else:
            value = " ".join(
                str(part.get("text", ""))
                for part in message.content
                if part.get("type") == "text"
                and not str(part.get("text", "")).startswith("[Omitted ")
            )
        normalized = _displayable_title_source(value)
        if normalized:
            return normalized[:80]
    return "New session"


def _displayable_title_source(value: str) -> str:
    """Remove injected context while retaining the user's visible mention."""

    def resource_mention(match: re.Match[str]) -> str:
        attributes = {
            name.casefold(): html.unescape(content)
            for name, content in _RESOURCE_ATTRIBUTE_RE.findall(match.group("attributes"))
        }
        server = attributes.get("server")
        uri = attributes.get("uri")
        return f"@{server}:{uri}" if server and uri else ""

    without_resources = _EXPANDED_RESOURCE_RE.sub(resource_mention, value)
    without_context = _EXPANDED_CONTEXT_RE.sub("", without_resources)
    return re.sub(r"\s+", " ", html.unescape(without_context)).strip()


def _meta(row: sqlite3.Row, message_count: int) -> SessionMeta:
    return SessionMeta(
        str(row["id"]),
        str(row["workspace"]),
        str(row["provider"]),
        str(row["model"]),
        str(row["title"]),
        str(row["created_at"]),
        str(row["updated_at"]),
        message_count,
    )


def _todo_item(row: sqlite3.Row) -> TodoItem:
    return TodoItem(
        str(row["id"]),
        str(row["content"]),
        TodoStatus(str(row["status"])),
        int(row["position"]),
        str(row["created_at"]),
        str(row["updated_at"]),
    )


def _normalize_todos(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not isinstance(items, list):
        raise ValueError("Todo items must be a list")
    if len(items) > MAX_TODOS:
        raise ValueError(f"A session may contain at most {MAX_TODOS} todo items")
    normalized: list[dict[str, str]] = []
    identifiers: set[str] = set()
    active = 0
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("Each todo item must be an object")
        content = re.sub(r"\s+", " ", str(raw.get("content", ""))).strip()
        if not content:
            raise ValueError("Todo content cannot be empty")
        if len(content) > MAX_TODO_CONTENT_CHARS:
            raise ValueError(f"Todo content cannot exceed {MAX_TODO_CONTENT_CHARS} characters")
        raw_status = str(raw.get("status", TodoStatus.PENDING.value))
        try:
            status = TodoStatus(raw_status)
        except ValueError as exc:
            raise ValueError(f"Invalid todo status: {raw_status}") from exc
        identifier = str(raw.get("id", "")).strip() or f"todo_{uuid.uuid4().hex[:12]}"
        if not TODO_ID_PATTERN.fullmatch(identifier):
            raise ValueError(f"Invalid todo ID: {identifier}")
        if identifier in identifiers:
            raise ValueError(f"Duplicate todo ID: {identifier}")
        identifiers.add(identifier)
        if status == TodoStatus.IN_PROGRESS:
            active += 1
        normalized.append({"id": identifier, "content": content, "status": status.value})
    if active > 1:
        raise ValueError("Only one todo item may be in progress")
    return normalized


def _validate_session_id(session_id: str) -> None:
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise ValueError("Invalid session ID")
