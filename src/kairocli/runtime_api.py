from __future__ import annotations

import asyncio
import hmac
import json
import os
import re
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Query
from fastapi.responses import PlainTextResponse

from .agent import Agent, AgentCanceled
from .brand import API_KEY_HEADER
from .models import Message
from .paths import reject_symlink_components
from .text_safety import bound_utf8
from .trace import safe_redacted_text
from .user_input import UserInputError, normalize_user_input

MAX_RUNTIME_CONTEXT_BYTES = 1024 * 1024
MAX_RUNTIME_EVENT_BYTES = 256 * 1024
MAX_RUNTIME_EVENT_RESPONSE_BYTES = 1024 * 1024
MAX_RUNTIME_EVENT_JSON_DEPTH = 32
MAX_RUNTIME_EVENT_JSON_NODES = 50_000
MAX_RUNTIME_EVENTS_PER_THREAD = 10_000
MAX_RUNTIME_TURNS_PER_THREAD = 100
MAX_RUNTIME_THREADS = 1_000
RUNTIME_DELTA_CHARS = 4_096
MAX_RUNTIME_API_KEY_CHARS = 1_024
RUNTIME_SHUTDOWN_GRACE_SECONDS = 0.5
MAX_SQLITE_INTEGER = 2**63 - 1
_DETACHED_RUNTIME_TASKS: set[asyncio.Task[Any]] = set()
_IDEMPOTENCY_KEY = re.compile(r"[\x21-\x7e]{1,200}\Z")
_BEARER_AUTHORIZATION = re.compile(r"(?i:Bearer) +([\x21-\x7e]{1,1024})\Z")
_THREAD_ID = re.compile(r"thread_[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_TURN_ID = re.compile(r"turn_[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_EVENT_CURSOR = re.compile(r"[0-9]{1,19}\Z")
_EVENT_TYPE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


@dataclass(slots=True)
class RuntimeEvent:
    id: int
    thread_id: str
    type: str
    data: dict[str, Any]
    timestamp: str


class RuntimeState:
    def __init__(self, agent_factory: Any, store: RuntimeThreadStore) -> None:
        self.agent_factory = agent_factory
        self.store = store
        self.active_agents: dict[str, Agent] = {}
        self.active_turn_ids: set[str] = set()
        self.active_turn_tasks: set[asyncio.Task[Any]] = set()
        self.owner_token = f"runtime_{uuid.uuid4().hex}"
        self.owner_pid = os.getpid()

    def event(self, thread_id: str, event_type: str, data: dict[str, Any]) -> None:
        self.store.append(thread_id, event_type, data)

    def refresh_owner_identity(self) -> None:
        current_pid = os.getpid()
        if current_pid == self.owner_pid:
            return
        self.owner_pid = current_pid
        self.owner_token = f"runtime_{uuid.uuid4().hex}"
        # A process created by fork inherits objects, not the parent's running tasks.
        self.active_agents.clear()
        self.active_turn_ids.clear()
        self.active_turn_tasks.clear()


class RuntimeThreadStore:
    def __init__(
        self,
        database: Path,
        *,
        max_events_per_thread: int = MAX_RUNTIME_EVENTS_PER_THREAD,
        max_turns_per_thread: int = MAX_RUNTIME_TURNS_PER_THREAD,
        max_threads: int = MAX_RUNTIME_THREADS,
    ) -> None:
        self.database = database
        self.max_events_per_thread = max(2, max_events_per_thread)
        self.max_turns_per_thread = max(2, max_turns_per_thread)
        self.max_threads = max(2, max_threads)
        self._default_owner_token = f"store_{uuid.uuid4().hex}"
        reject_symlink_components(database, "Runtime database")
        database.parent.mkdir(parents=True, exist_ok=True)
        reject_symlink_components(database.parent, "Runtime database")
        if os.name != "nt":
            database.parent.chmod(0o700)
        with self._connect() as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS threads (id TEXT PRIMARY KEY, created_at TEXT NOT NULL)"
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT, thread_id TEXT NOT NULL,
                type TEXT NOT NULL, data TEXT NOT NULL, timestamp TEXT NOT NULL,
                FOREIGN KEY(thread_id) REFERENCES threads(id))"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS turns (
                id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, prompt TEXT NOT NULL,
                idempotency_key TEXT, status TEXT NOT NULL, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, response TEXT, error TEXT,
                UNIQUE(thread_id, idempotency_key),
                FOREIGN KEY(thread_id) REFERENCES threads(id))"""
            )
            columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(turns)").fetchall()
            }
            if "response" not in columns:
                connection.execute("ALTER TABLE turns ADD COLUMN response TEXT")
            if "error" not in columns:
                connection.execute("ALTER TABLE turns ADD COLUMN error TEXT")
            if "owner_token" not in columns:
                connection.execute(
                    "ALTER TABLE turns ADD COLUMN owner_token TEXT NOT NULL DEFAULT ''"
                )
            if "owner_pid" not in columns:
                connection.execute(
                    "ALTER TABLE turns ADD COLUMN owner_pid INTEGER NOT NULL DEFAULT 0"
                )
            thread_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(threads)").fetchall()
            }
            if "owner_user_id" not in thread_columns:
                connection.execute(
                    "ALTER TABLE threads ADD COLUMN owner_user_id TEXT NOT NULL DEFAULT ''"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_thread_sequence "
                "ON events(thread_id, sequence)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_turns_thread_created "
                "ON turns(thread_id, created_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_threads_owner_user_id "
                "ON threads(owner_user_id)"
            )
            self._recover_stale_running(connection)
        if os.name != "nt":
            database.chmod(0o600)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self._check_database_paths()
        connection = sqlite3.connect(self.database, timeout=30)
        try:
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
        reject_symlink_components(self.database.parent, "Runtime database")
        for path in self._database_files():
            if path.is_symlink():
                raise ValueError(f"Runtime database file cannot be a symlink: {path.name}")

    def _harden_database_files(self) -> None:
        if os.name == "nt":
            return
        for path in self._database_files():
            if path.is_symlink():
                raise ValueError(f"Runtime database file cannot be a symlink: {path.name}")
            if path.is_file():
                path.chmod(0o600)

    def _database_files(self) -> tuple[Path, Path, Path]:
        return (
            self.database,
            Path(str(self.database) + "-wal"),
            Path(str(self.database) + "-shm"),
        )

    def create(
        self,
        thread_id: str,
        *,
        owner_user_id: str = "",
        event_type: str | None = None,
        event_data: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO threads VALUES (?, ?, ?)",
                (thread_id, datetime.now(UTC).isoformat(), owner_user_id),
            )
            self._append_event(connection, thread_id, event_type, event_data)
            self._prune_threads(connection, keep_thread_id=thread_id)

    def exists(self, thread_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute("SELECT 1 FROM threads WHERE id=?", (thread_id,)).fetchone()
        return row is not None

    def delete_thread(self, thread_id: str, owner_user_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM threads WHERE id=? AND owner_user_id=?",
                (thread_id, owner_user_id),
            ).fetchone()
            if row is None:
                return False
            connection.execute("DELETE FROM events WHERE thread_id=?", (thread_id,))
            connection.execute("DELETE FROM turns WHERE thread_id=?", (thread_id,))
            connection.execute("DELETE FROM threads WHERE id=?", (thread_id,))
        return True

    def exists_for_user(self, thread_id: str, user_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM threads WHERE id=? AND owner_user_id=?",
                (thread_id, user_id),
            ).fetchone()
        return row is not None

    def list_threads(
        self,
        owner_user_id: str,
        *,
        limit: int = 200,
    ) -> list[dict[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT t.id, t.created_at, "
                "  (SELECT e.data FROM events e "
                "   WHERE e.thread_id = t.id AND e.type = 'turn.started' "
                "   ORDER BY e.sequence ASC LIMIT 1) AS first_event "
                "FROM threads t WHERE t.owner_user_id=? "
                "ORDER BY t.created_at DESC LIMIT ?",
                (owner_user_id, max(1, min(limit, 1_000))),
            ).fetchall()
        result = []
        for row in rows:
            title = ""
            if row[2]:
                try:
                    data = json.loads(row[2])
                    raw = data.get("input", "")
                    title = raw[:60] + ("…" if len(raw) > 60 else "")
                except Exception:
                    pass
            result.append({"id": str(row[0]), "created_at": str(row[1]), "title": title})
        return result

    def append(self, thread_id: str, event_type: str, data: dict[str, Any]) -> int:
        with self._connect() as connection:
            cursor = self._append_event(connection, thread_id, event_type, data)
        assert cursor is not None
        return cursor

    def events(self, thread_id: str, after_id: int = 0, limit: int = 1_000) -> list[RuntimeEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT sequence,type,data,timestamp FROM events "
                "WHERE thread_id=? AND sequence>? ORDER BY sequence LIMIT ?",
                (thread_id, max(0, after_id), max(1, min(limit, 1_001))),
            ).fetchall()
        return [_stored_runtime_event(thread_id, row) for row in rows]

    def reserve_turn(
        self,
        thread_id: str,
        prompt: str,
        idempotency_key: str | None,
        *,
        owner_token: str | None = None,
        owner_pid: int | None = None,
        event_type: str | None = None,
        event_data: dict[str, Any] | None = None,
    ) -> tuple[str, str, bool]:
        normalized_owner = owner_token or self._default_owner_token
        normalized_pid = os.getpid() if owner_pid is None else max(0, int(owner_pid))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._recover_stale_running(connection)
            if idempotency_key is not None:
                existing = connection.execute(
                    "SELECT id,prompt,status FROM turns WHERE thread_id=? AND idempotency_key=?",
                    (thread_id, idempotency_key),
                ).fetchone()
                if existing is not None:
                    if str(existing[1]) != prompt:
                        raise ValueError("Idempotency-Key was already used with different input")
                    return str(existing[0]), str(existing[2]), False
            running = connection.execute(
                "SELECT id FROM turns WHERE thread_id=? AND status='running' LIMIT 1",
                (thread_id,),
            ).fetchone()
            if running is not None:
                raise RuntimeError(f"Thread already has a running turn: {running[0]}")
            turn_id = f"turn_{uuid.uuid4().hex[:12]}"
            now = datetime.now(UTC).isoformat()
            connection.execute(
                "INSERT INTO turns "
                "(id,thread_id,prompt,idempotency_key,status,created_at,updated_at,"
                "owner_token,owner_pid) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    turn_id,
                    thread_id,
                    prompt,
                    idempotency_key,
                    "running",
                    now,
                    now,
                    normalized_owner,
                    normalized_pid,
                ),
            )
            started_event_data = dict(event_data) if event_data is not None else None
            if started_event_data is not None:
                started_event_data["turn_id"] = turn_id
            self._append_event(connection, thread_id, event_type, started_event_data)
            connection.execute(
                "DELETE FROM turns WHERE id IN ("
                "SELECT id FROM turns WHERE thread_id=? AND status!='running' "
                "ORDER BY created_at DESC LIMIT -1 OFFSET ?)",
                (thread_id, self.max_turns_per_thread - 1),
            )
        return turn_id, "running", True

    def completed_messages(
        self,
        thread_id: str,
        *,
        before_turn_id: str | None = None,
        max_bytes: int = MAX_RUNTIME_CONTEXT_BYTES,
        max_turns: int = 50,
    ) -> list[Message]:
        params: list[Any] = [thread_id]
        before_clause = ""
        if before_turn_id is not None:
            before_clause = "AND id<>?"
            params.append(before_turn_id)
        params.append(max(1, min(max_turns, 100)))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT prompt,response FROM turns WHERE thread_id=? AND status='completed' "
                f"AND response IS NOT NULL {before_clause} ORDER BY created_at DESC LIMIT ?",
                params,
            ).fetchall()
        selected: list[tuple[str, str]] = []
        used = 0
        for prompt, response in rows:
            pair_bytes = len(str(prompt).encode("utf-8")) + len(str(response).encode("utf-8"))
            if selected and used + pair_bytes > max_bytes:
                break
            if pair_bytes > max_bytes:
                continue
            selected.append((str(prompt), str(response)))
            used += pair_bytes
        messages: list[Message] = []
        for prompt, response in reversed(selected):
            messages.extend([Message("user", prompt), Message("assistant", response)])
        return messages

    def update_turn_status(
        self,
        turn_id: str,
        status: str,
        *,
        response: str | None = None,
        error: str | None = None,
        owner_token: str | None = None,
        event_type: str | None = None,
        event_data: dict[str, Any] | None = None,
    ) -> bool:
        if status not in {"completed", "failed", "canceled"}:
            raise ValueError(f"Invalid runtime turn status: {status}")
        if response is not None:
            response = bound_utf8(
                response,
                MAX_RUNTIME_CONTEXT_BYTES,
                "\n[response truncated]",
            )
        if error is not None:
            error = safe_redacted_text(error, 2_000, "...[runtime error truncated]")
        with self._connect() as connection:
            owner_predicate = " AND owner_token=?" if owner_token is not None else ""
            owner_arguments = (owner_token,) if owner_token is not None else ()
            cursor = connection.execute(
                "UPDATE turns SET status=?, response=?, error=?, owner_token='', owner_pid=0, "
                f"updated_at=? WHERE id=? AND status='running'{owner_predicate}",
                (
                    status,
                    response,
                    error,
                    datetime.now(UTC).isoformat(),
                    turn_id,
                    *owner_arguments,
                ),
            )
            if cursor.rowcount == 1:
                row = connection.execute(
                    "SELECT thread_id FROM turns WHERE id=?", (turn_id,)
                ).fetchone()
                if row is None:  # pragma: no cover - protected by the successful update
                    raise RuntimeError("Updated runtime turn disappeared")
                self._append_event(
                    connection,
                    str(row[0]),
                    event_type,
                    event_data,
                )
        return cursor.rowcount == 1

    def _append_event(
        self,
        connection: sqlite3.Connection,
        thread_id: str,
        event_type: str | None,
        event_data: dict[str, Any] | None,
    ) -> int | None:
        if (event_type is None) != (event_data is None):
            raise ValueError("Runtime event type and data must be provided together")
        if event_type is None or event_data is None:
            return None
        if _EVENT_TYPE.fullmatch(event_type) is None:
            raise ValueError("Invalid runtime event type")
        encoded = _bounded_event_json(event_data)
        cursor = connection.execute(
            "INSERT INTO events(thread_id,type,data,timestamp) VALUES (?,?,?,?)",
            (
                thread_id,
                event_type,
                encoded,
                datetime.now(UTC).isoformat(),
            ),
        )
        connection.execute(
            "DELETE FROM events WHERE thread_id=? AND sequence < COALESCE(("
            "SELECT sequence FROM events WHERE thread_id=? "
            "ORDER BY sequence DESC LIMIT 1 OFFSET ?), -1)",
            (thread_id, thread_id, self.max_events_per_thread - 1),
        )
        return int(cursor.lastrowid or 0)

    def turn_status(self, thread_id: str, turn_id: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM turns WHERE thread_id=? AND id=?",
                (thread_id, turn_id),
            ).fetchone()
        return str(row[0]) if row is not None else None

    def _recover_stale_running(self, connection: sqlite3.Connection) -> None:
        now = datetime.now(UTC).isoformat()
        rows = connection.execute("SELECT id, owner_pid FROM turns WHERE status='running'")
        for turn_id, owner_pid in rows:
            if _process_is_alive(int(owner_pid or 0)):
                continue
            connection.execute(
                "UPDATE turns SET status='failed', "
                "error='runtime owner exited before completion', owner_token='', owner_pid=0, "
                "updated_at=? WHERE id=? AND status='running'",
                (now, turn_id),
            )

    def _prune_threads(self, connection: sqlite3.Connection, *, keep_thread_id: str) -> None:
        stale = connection.execute(
            "SELECT id FROM threads WHERE id<>? AND id NOT IN ("
            "SELECT DISTINCT thread_id FROM turns WHERE status='running') "
            "ORDER BY created_at DESC LIMIT -1 OFFSET ?",
            (keep_thread_id, self.max_threads - 1),
        ).fetchall()
        stale_ids = [str(row[0]) for row in stale]
        for stale_id in stale_ids:
            connection.execute("DELETE FROM events WHERE thread_id=?", (stale_id,))
            connection.execute("DELETE FROM turns WHERE thread_id=?", (stale_id,))
            connection.execute("DELETE FROM threads WHERE id=?", (stale_id,))


def _bounded_event_json(data: dict[str, Any]) -> str:
    try:
        _validate_runtime_json_tree(data)
        encoded = json.dumps(
            data,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (RecursionError, TypeError, ValueError) as exc:
        encoded = json.dumps({"serialization_error": type(exc).__name__}, separators=(",", ":"))
    raw = encoded.encode("utf-8")
    if len(raw) <= MAX_RUNTIME_EVENT_BYTES:
        return encoded
    preview = raw[: MAX_RUNTIME_EVENT_BYTES // 4].decode("utf-8", errors="ignore")
    return json.dumps(
        {
            "partial": True,
            "original_bytes": len(raw),
            "json_preview": preview,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _stored_runtime_event(thread_id: str, row: tuple[Any, ...]) -> RuntimeEvent:
    sequence = int(row[0])
    event_type = row[1]
    timestamp = row[3]
    try:
        if not isinstance(event_type, str) or _EVENT_TYPE.fullmatch(event_type) is None:
            raise ValueError("invalid event type")
        data = json.loads(
            row[2],
            object_pairs_hook=_runtime_object_without_duplicates,
            parse_constant=_reject_runtime_json_constant,
        )
        if not isinstance(data, dict):
            raise ValueError("event data must be an object")
        _validate_runtime_json_tree(data)
    except (RecursionError, TypeError, UnicodeDecodeError, ValueError):
        return RuntimeEvent(
            sequence,
            thread_id,
            "runtime.event.invalid",
            {"corrupt": True},
            str(timestamp),
        )
    return RuntimeEvent(sequence, thread_id, event_type, data, str(timestamp))


def _runtime_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate runtime event key: {key}")
        result[key] = value
    return result


def _reject_runtime_json_constant(value: str) -> Any:
    raise ValueError(f"invalid runtime event JSON constant: {value}")


def _validate_runtime_json_tree(root: Any) -> None:
    nodes = 0
    stack: list[tuple[Any, int]] = [(root, 1)]
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > MAX_RUNTIME_EVENT_JSON_NODES:
            raise ValueError("runtime event JSON is too complex")
        if depth > MAX_RUNTIME_EVENT_JSON_DEPTH:
            raise ValueError("runtime event JSON is too deeply nested")
        if isinstance(value, dict):
            stack.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            stack.extend((item, depth + 1) for item in value)


def _bearer_credential(value: str | None) -> str:
    if value is None or len(value) > MAX_RUNTIME_API_KEY_CHARS + 16:
        return ""
    matched = _BEARER_AUTHORIZATION.fullmatch(value)
    return matched.group(1) if matched is not None else ""


def _valid_identifier(value: str, pattern: re.Pattern[str]) -> bool:
    return pattern.fullmatch(value) is not None


def _process_is_alive(pid: int) -> bool:
    if pid <= 0 or pid > 2**31 - 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, OverflowError):
        return False
    return True


def _last_event_cursor(value: str) -> int:
    if _EVENT_CURSOR.fullmatch(value) is None:
        raise ValueError("Invalid Last-Event-ID")
    cursor = int(value)
    if cursor > MAX_SQLITE_INTEGER:
        raise ValueError("Invalid Last-Event-ID")
    return cursor


def _finish_detached_runtime_task(task: asyncio.Task[Any]) -> None:
    _DETACHED_RUNTIME_TASKS.discard(task)
    try:
        task.exception()
    except (asyncio.CancelledError, Exception):
        pass


async def _await_runtime_shutdown(task: asyncio.Task[None]) -> None:
    canceled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            canceled = True
    task.result()
    if canceled:
        raise asyncio.CancelledError


def create_app(agent_factory: Any, api_key: str | None = None, database: Path | None = None) -> Any:
    expected_key = api_key or os.getenv("KAIROCLI_RUNTIME_API_KEY", "")
    if not isinstance(expected_key, str) or not expected_key:
        raise ValueError("KAIROCLI_RUNTIME_API_KEY is required")
    if len(expected_key) > MAX_RUNTIME_API_KEY_CHARS or any(
        not 0x21 <= ord(character) <= 0x7E for character in expected_key
    ):
        raise ValueError("KAIROCLI_RUNTIME_API_KEY must contain 1 to 1024 visible ASCII characters")
    runtime_database = database or Path.home() / ".kairocli" / "runtime" / "runtime.db"
    state = RuntimeState(agent_factory, RuntimeThreadStore(runtime_database))

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> Any:
        state.refresh_owner_identity()
        try:
            yield
        finally:

            async def finish_shutdown() -> None:
                for turn_id in tuple(state.active_turn_ids):
                    active_agent = state.active_agents.get(turn_id)
                    if active_agent is not None:
                        try:
                            active_agent.cancel()
                        except Exception:
                            pass
                    try:
                        state.store.update_turn_status(
                            turn_id,
                            "canceled",
                            owner_token=state.owner_token,
                            event_type="turn.canceled",
                            event_data={"turn_id": turn_id},
                        )
                    except Exception:
                        pass
                active_tasks = tuple(state.active_turn_tasks)
                for task in active_tasks:
                    task.cancel()
                if active_tasks:
                    done, pending = await asyncio.wait(
                        active_tasks, timeout=RUNTIME_SHUTDOWN_GRACE_SECONDS
                    )
                    if done:
                        await asyncio.gather(*done, return_exceptions=True)
                    for task in pending:
                        _DETACHED_RUNTIME_TASKS.add(task)
                        task.add_done_callback(_finish_detached_runtime_task)
                state.active_agents.clear()
                state.active_turn_ids.clear()
                state.active_turn_tasks.clear()

            shutdown_task = asyncio.create_task(finish_shutdown(), name="kairo-runtime-shutdown")
            await _await_runtime_shutdown(shutdown_task)

    app = FastAPI(title="Kairo CLI Runtime API", version="1", lifespan=lifespan)
    app.state.runtime = state

    def authorize(authorization: str | None, product_key: str | None) -> None:
        bearer = _bearer_credential(authorization)
        bearer_matches = hmac.compare_digest(bearer, expected_key)
        bounded_product_key = (
            product_key
            if product_key is not None and len(product_key) <= MAX_RUNTIME_API_KEY_CHARS
            else ""
        )
        product_matches = hmac.compare_digest(bounded_product_key, expected_key)
        if not bearer_matches and not product_matches:
            raise HTTPException(status_code=401, detail="Unauthorized")

    @app.post("/v1/threads")
    async def create_thread(
        authorization: str | None = Header(default=None),
        x_kairocli_api_key: str | None = Header(default=None, alias=API_KEY_HEADER),
    ) -> dict[str, str]:
        authorize(authorization, x_kairocli_api_key)
        thread_id = f"thread_{uuid.uuid4().hex[:12]}"
        state.store.create(
            thread_id,
            event_type="thread.created",
            event_data={"thread_id": thread_id},
        )
        return {"id": thread_id}

    async def execute_turn(thread_id: str, prompt: str, turn_id: str) -> None:
        current_task = asyncio.current_task()
        if current_task is not None:
            state.active_turn_tasks.add(current_task)
        agent: Agent | None = None
        delta_buffer = ""
        emitted_delta = False

        def flush_delta(*, final: bool = False) -> None:
            nonlocal delta_buffer
            while len(delta_buffer) >= RUNTIME_DELTA_CHARS or (final and delta_buffer):
                chunk = delta_buffer[:RUNTIME_DELTA_CHARS]
                delta_buffer = delta_buffer[len(chunk) :]
                state.event(
                    thread_id,
                    "message.delta",
                    {"turn_id": turn_id, "delta": chunk},
                )

        try:
            if state.store.turn_status(thread_id, turn_id) != "running":
                return
            agent = agent_factory()
            state.active_agents[turn_id] = agent
            agent.history = state.store.completed_messages(thread_id, before_turn_id=turn_id)

            def emit_delta(delta: str) -> None:
                nonlocal delta_buffer, emitted_delta
                if not delta:
                    return
                emitted_delta = True
                delta_buffer += delta
                flush_delta()

            agent.on_content_delta = emit_delta
            answer = await agent.run(prompt)
            if answer and not emitted_delta:
                emit_delta(answer)
            flush_delta(final=True)
            state.store.update_turn_status(
                turn_id,
                "completed",
                response=answer,
                owner_token=state.owner_token,
                event_type="turn.completed",
                event_data={"turn_id": turn_id},
            )
        except AgentCanceled:
            state.store.update_turn_status(
                turn_id,
                "canceled",
                owner_token=state.owner_token,
                event_type="turn.canceled",
                event_data={"turn_id": turn_id},
            )
        except asyncio.CancelledError:
            state.store.update_turn_status(
                turn_id,
                "canceled",
                owner_token=state.owner_token,
                event_type="turn.canceled",
                event_data={"turn_id": turn_id},
            )
            raise
        except Exception as exc:
            flush_delta(final=True)
            error = safe_redacted_text(exc, 2_000, "...[runtime error truncated]")
            state.store.update_turn_status(
                turn_id,
                "failed",
                error=error,
                owner_token=state.owner_token,
                event_type="turn.failed",
                event_data={"turn_id": turn_id, "error": error},
            )
        finally:
            state.active_agents.pop(turn_id, None)
            state.active_turn_ids.discard(turn_id)
            if current_task is not None:
                state.active_turn_tasks.discard(current_task)
            if agent is not None:
                try:
                    await agent.tools.close()
                except Exception:
                    pass

    @app.post("/v1/threads/{thread_id}/turns", status_code=202)
    async def create_turn(
        thread_id: str,
        payload: dict[str, Any],
        background: BackgroundTasks,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        authorization: str | None = Header(default=None),
        x_kairocli_api_key: str | None = Header(default=None, alias=API_KEY_HEADER),
    ) -> dict[str, str]:
        authorize(authorization, x_kairocli_api_key)
        if not _valid_identifier(thread_id, _THREAD_ID):
            raise HTTPException(status_code=404, detail="Thread not found")
        if not state.store.exists(thread_id):
            raise HTTPException(status_code=404, detail="Thread not found")
        raw_prompt = payload.get("input") if "input" in payload else payload.get("prompt")
        if raw_prompt is None:
            raise HTTPException(status_code=422, detail="input is required")
        if not isinstance(raw_prompt, str):
            raise HTTPException(status_code=422, detail="input must be a string")
        prompt = raw_prompt
        if not prompt:
            raise HTTPException(status_code=422, detail="input is required")
        try:
            prompt = normalize_user_input(prompt)
        except UserInputError as exc:
            raise HTTPException(
                status_code=413,
                detail=safe_redacted_text(exc, 4_000, "...[runtime error truncated]"),
            ) from None
        if not prompt.strip():
            raise HTTPException(status_code=422, detail="input is required")
        if idempotency_key is not None:
            if not _IDEMPOTENCY_KEY.fullmatch(idempotency_key):
                raise HTTPException(status_code=422, detail="Invalid Idempotency-Key")
        state.refresh_owner_identity()
        try:
            turn_id, status, created = state.store.reserve_turn(
                thread_id,
                prompt,
                idempotency_key,
                owner_token=state.owner_token,
                owner_pid=state.owner_pid,
                event_type="turn.started",
                event_data={"input": prompt},
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=409,
                detail=safe_redacted_text(exc, 4_000, "...[runtime error truncated]"),
            ) from None
        except RuntimeError as exc:
            raise HTTPException(
                status_code=409,
                detail=safe_redacted_text(exc, 4_000, "...[runtime error truncated]"),
            ) from None
        if not created:
            return {"id": turn_id, "object": "turn", "status": status}
        # Register immediately after the durable reservation. The Starlette
        # background callback may not have started when lifespan shutdown begins.
        state.active_turn_ids.add(turn_id)
        background.add_task(execute_turn, thread_id, prompt, turn_id)
        return {"id": turn_id, "object": "turn", "status": "running"}

    @app.post("/v1/threads/{thread_id}/turns/{turn_id}/cancel")
    async def cancel_turn(
        thread_id: str,
        turn_id: str,
        authorization: str | None = Header(default=None),
        x_kairocli_api_key: str | None = Header(default=None, alias=API_KEY_HEADER),
    ) -> dict[str, str]:
        authorize(authorization, x_kairocli_api_key)
        if not _valid_identifier(thread_id, _THREAD_ID):
            raise HTTPException(status_code=404, detail="Thread not found")
        if not state.store.exists(thread_id):
            raise HTTPException(status_code=404, detail="Thread not found")
        if not _valid_identifier(turn_id, _TURN_ID):
            raise HTTPException(status_code=404, detail="Turn not found")
        status = state.store.turn_status(thread_id, turn_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Turn not found")
        if status == "running" and state.store.update_turn_status(
            turn_id,
            "canceled",
            event_type="turn.canceled",
            event_data={"turn_id": turn_id},
        ):
            active = state.active_agents.get(turn_id)
            if active is not None:
                active.cancel()
            status = "canceled"
        return {"id": turn_id, "object": "turn", "status": status}

    @app.get("/v1/threads/{thread_id}/events", response_class=PlainTextResponse)
    async def events(
        thread_id: str,
        after: int = Query(default=0, ge=0, le=MAX_SQLITE_INTEGER),
        limit: int = Query(default=100, ge=1, le=1_000),
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
        authorization: str | None = Header(default=None),
        x_kairocli_api_key: str | None = Header(default=None, alias=API_KEY_HEADER),
    ) -> PlainTextResponse:
        authorize(authorization, x_kairocli_api_key)
        if not _valid_identifier(thread_id, _THREAD_ID):
            raise HTTPException(status_code=404, detail="Thread not found")
        if not state.store.exists(thread_id):
            raise HTTPException(status_code=404, detail="Thread not found")
        cursor = after
        if after == 0 and last_event_id:
            try:
                cursor = _last_event_cursor(last_event_id)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid Last-Event-ID") from None
        chunks: list[str] = []
        response_bytes = 0
        next_event_id = cursor
        selected = state.store.events(thread_id, cursor, limit + 1)
        has_more = len(selected) > limit
        for event in selected[:limit]:
            chunk = (
                f"id: {event.id}\nevent: {event.type}\n"
                f"data: {json.dumps(event.data, ensure_ascii=False)}\n\n"
            )
            chunk_bytes = len(chunk.encode("utf-8"))
            if chunks and response_bytes + chunk_bytes > MAX_RUNTIME_EVENT_RESPONSE_BYTES:
                has_more = True
                break
            chunks.append(chunk)
            response_bytes += chunk_bytes
            next_event_id = event.id
        body = "".join(chunks)
        return PlainTextResponse(
            body,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Kairo-CLI-Next-Event-ID": str(next_event_id),
                "X-Kairo-CLI-Has-More": str(has_more).lower(),
            },
        )

    return app
