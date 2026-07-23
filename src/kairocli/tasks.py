from __future__ import annotations

import asyncio
import os
import re
import sqlite3
import uuid
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .paths import reject_symlink_components
from .text_safety import bound_utf8, safe_text
from .trace import safe_redacted_text

MAX_TASK_PROMPT_BYTES = 1024 * 1024
MAX_TASK_OUTPUT_BYTES = 1024 * 1024
MAX_TASK_ERROR_CHARS = 2_000
MAX_TERMINAL_TASKS = 1_000
TASK_SHUTDOWN_GRACE_SECONDS = 0.1
_TASK_ID = re.compile(r"task_[0-9a-f]{12}\Z")
_TERMINAL_STATUSES = {"completed", "failed", "canceled"}
_DETACHED_MANAGER_TASKS: set[asyncio.Task[Any]] = set()
_TASK_COLUMNS = (
    "id, prompt, status, output, error, created_at, updated_at, "
    "started_at, finished_at, duration_ms"
)


@dataclass(slots=True)
class DurableTask:
    id: str
    prompt: str
    status: str
    output: str
    error: str
    created_at: str
    updated_at: str
    started_at: str
    finished_at: str
    duration_ms: int


TaskTerminalCallback = Callable[[DurableTask], Awaitable[None] | None]


class DurableTaskStore:
    STATUSES = {"enqueued", "running", *_TERMINAL_STATUSES}

    def __init__(self, database: Path, *, max_terminal_tasks: int = MAX_TERMINAL_TASKS) -> None:
        self.database = database
        self.max_terminal_tasks = max(0, max_terminal_tasks)
        reject_symlink_components(database, "Task database")
        database.parent.mkdir(parents=True, exist_ok=True)
        reject_symlink_components(database.parent, "Task database")
        if os.name == "posix":
            database.parent.chmod(0o700)
            if not database.exists():
                descriptor = os.open(database, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(descriptor)
        self._initialize()
        self._harden_database_files()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self._check_database_paths()
        connection = sqlite3.connect(self.database, timeout=10)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout=10000")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
            self._harden_database_files()

    def _check_database_paths(self) -> None:
        reject_symlink_components(self.database.parent, "Task database")
        for path in self._database_files():
            if path.is_symlink():
                raise ValueError(f"Task database file cannot be a symlink: {path.name}")

    def _harden_database_files(self) -> None:
        if os.name != "posix":
            return
        for path in self._database_files():
            if path.is_symlink():
                raise ValueError(f"Task database file cannot be a symlink: {path.name}")
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
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY, prompt TEXT NOT NULL, status TEXT NOT NULL,
                output TEXT NOT NULL DEFAULT '', error TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                started_at TEXT NOT NULL DEFAULT '', finished_at TEXT NOT NULL DEFAULT '',
                duration_ms INTEGER NOT NULL DEFAULT 0)"""
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(tasks)").fetchall()
            }
            migrations = {
                "started_at": "TEXT NOT NULL DEFAULT ''",
                "finished_at": "TEXT NOT NULL DEFAULT ''",
                "duration_ms": "INTEGER NOT NULL DEFAULT 0",
                "owner_token": "TEXT NOT NULL DEFAULT ''",
                "owner_pid": "INTEGER NOT NULL DEFAULT 0",
            }
            for column, declaration in migrations.items():
                if column not in columns:
                    connection.execute(f"ALTER TABLE tasks ADD COLUMN {column} {declaration}")
            self._recover_stale_running(connection)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_status_created ON tasks(status, created_at)"
            )
            self._prune_terminal(connection)

    def add(self, prompt: str) -> DurableTask:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Task prompt cannot be empty")
        if len(prompt.encode("utf-8")) > MAX_TASK_PROMPT_BYTES:
            raise ValueError(f"Task prompt exceeds {MAX_TASK_PROMPT_BYTES} bytes")
        now = datetime.now(UTC).isoformat()
        task = DurableTask(
            f"task_{uuid.uuid4().hex[:12]}", prompt, "enqueued", "", "", now, now, "", "", 0
        )
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO tasks "
                "(id, prompt, status, output, error, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (task.id, task.prompt, task.status, task.output, task.error, now, now),
            )
        return task

    def get(self, task_id: str) -> DurableTask | None:
        if not _valid_task_id(task_id):
            return None
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT {_TASK_COLUMNS} FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
        return DurableTask(**dict(row)) if row else None

    def list(self, limit: int = 20) -> list[DurableTask]:
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT {_TASK_COLUMNS} FROM tasks ORDER BY created_at DESC LIMIT ?",
                (max(1, min(limit, 100)),),
            ).fetchall()
        return [DurableTask(**dict(row)) for row in rows]

    def claim(self, owner_token: str = "", owner_pid: int | None = None) -> DurableTask | None:
        normalized_owner = owner_token or f"store_{id(self):x}"
        normalized_pid = os.getpid() if owner_pid is None else max(0, int(owner_pid))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._recover_stale_running(connection)
            row = connection.execute(
                "SELECT id FROM tasks WHERE status='enqueued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            now = datetime.now(UTC).isoformat()
            cursor = connection.execute(
                "UPDATE tasks SET status='running', started_at=?, finished_at='', "
                "duration_ms=0, error='', owner_token=?, owner_pid=?, updated_at=? "
                "WHERE id=? AND status='enqueued'",
                (now, normalized_owner, normalized_pid, now, row["id"]),
            )
            if cursor.rowcount == 0:
                return None
            claimed = connection.execute(
                f"SELECT {_TASK_COLUMNS} FROM tasks WHERE id=?", (row["id"],)
            ).fetchone()
        return DurableTask(**dict(claimed)) if claimed else None

    def update(
        self,
        task_id: str,
        status: str,
        output: str = "",
        error: str = "",
        *,
        owner_token: str | None = None,
    ) -> bool:
        if not _valid_task_id(task_id):
            raise ValueError("Invalid task ID")
        if status not in self.STATUSES:
            raise ValueError(f"Invalid task status: {status}")
        output = bound_utf8(output, MAX_TASK_OUTPUT_BYTES, "\n[task output truncated]")
        error = safe_redacted_text(error, MAX_TASK_ERROR_CHARS, "...[task error truncated]")
        now = datetime.now(UTC)
        allowed_from = {
            "enqueued": ("running",),
            "running": ("enqueued",),
            "completed": ("running",),
            "failed": ("running",),
            "canceled": ("enqueued", "running"),
        }[status]
        placeholders = ",".join("?" for _ in allowed_from)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT started_at FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
            finished_at = now.isoformat() if status in _TERMINAL_STATUSES else ""
            duration_ms = _duration_ms(current["started_at"] if current else "", now, finished_at)
            owner_predicate = " AND owner_token=?" if owner_token is not None else ""
            owner_arguments = (owner_token,) if owner_token is not None else ()
            clear_owner = status in _TERMINAL_STATUSES or status == "enqueued"
            cursor = connection.execute(
                f"UPDATE tasks SET status=?, output=?, error=?, finished_at=?, "
                f"duration_ms=?, owner_token=?, owner_pid=?, updated_at=? "
                f"WHERE id=? AND status IN ({placeholders}){owner_predicate}",
                (
                    status,
                    output,
                    error,
                    finished_at,
                    duration_ms,
                    "" if clear_owner else owner_token or "",
                    0 if clear_owner else os.getpid(),
                    now.isoformat(),
                    task_id,
                    *allowed_from,
                    *owner_arguments,
                ),
            )
            if status in _TERMINAL_STATUSES:
                self._prune_terminal(connection)
        return cursor.rowcount > 0

    def cancel(self, task_id: str) -> bool:
        if not _valid_task_id(task_id):
            return False
        return self.update(task_id, "canceled", error="User canceled")

    def requeue_running(self, owner_token: str | None = None) -> int:
        now = datetime.now(UTC).isoformat()
        owner_predicate = " AND owner_token=?" if owner_token is not None else ""
        arguments: tuple[str, ...] = (now, owner_token) if owner_token is not None else (now,)
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE tasks SET status='enqueued', started_at='', finished_at='', "
                "duration_ms=0, error='', owner_token='', owner_pid=0, updated_at=? "
                f"WHERE status='running'{owner_predicate}",
                arguments,
            )
        return cursor.rowcount

    def _recover_stale_running(self, connection: sqlite3.Connection) -> None:
        now = datetime.now(UTC).isoformat()
        rows = connection.execute("SELECT id, owner_pid FROM tasks WHERE status='running'")
        for row in rows:
            if _process_is_alive(int(row["owner_pid"] or 0)):
                continue
            connection.execute(
                "UPDATE tasks SET status='enqueued', started_at='', finished_at='', "
                "duration_ms=0, error='', owner_token='', owner_pid=0, updated_at=? "
                "WHERE id=? AND status='running'",
                (now, row["id"]),
            )

    def _prune_terminal(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            "DELETE FROM tasks WHERE id IN ("
            "SELECT id FROM tasks WHERE status IN ('completed','failed','canceled') "
            "ORDER BY updated_at DESC, id DESC LIMIT -1 OFFSET ?)",
            (self.max_terminal_tasks,),
        )


class DurableTaskManager:
    def __init__(
        self,
        store: DurableTaskStore,
        agent_factory: Any,
        workers: int = 2,
        terminal_callback: TaskTerminalCallback | None = None,
    ) -> None:
        self.store = store
        self.agent_factory = agent_factory
        self.workers = max(1, workers)
        self._stop = asyncio.Event()
        self._worker_tasks: list[asyncio.Task[None]] = []
        self._notification_tasks: set[asyncio.Task[None]] = set()
        self._close_task: asyncio.Task[None] | None = None
        self._running: dict[str, tuple[str, Any]] = {}
        self._owner_token = f"manager_{uuid.uuid4().hex}"
        self._owner_pid = os.getpid()
        self.terminal_callback = terminal_callback

    def start(self) -> None:
        if self._worker_tasks:
            return
        if self._close_task is not None and not self._close_task.done():
            return
        self._close_task = None
        self._stop = asyncio.Event()
        self._owner_token = f"manager_{uuid.uuid4().hex}"
        self._owner_pid = os.getpid()
        stop = self._stop
        owner_token = self._owner_token
        owner_pid = self._owner_pid
        self._worker_tasks = [
            asyncio.create_task(
                self._worker(index, stop, owner_token, owner_pid),
                name=f"kairo-task-worker-{index}",
            )
            for index in range(self.workers)
        ]

    async def close(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._close_once(), name="kairo-task-manager-shutdown"
            )
        await _await_manager_shutdown(self._close_task)

    async def _close_once(self) -> None:
        stop = self._stop
        owner_token = self._owner_token
        worker_tasks = tuple(self._worker_tasks)
        stop.set()
        for running_owner, agent in tuple(self._running.values()):
            if running_owner == owner_token:
                agent.cancel()
        await _cancel_manager_tasks(worker_tasks)
        self._worker_tasks.clear()
        await _cancel_manager_tasks(tuple(self._notification_tasks))
        self._notification_tasks.clear()
        await asyncio.to_thread(self.store.requeue_running, owner_token)

    def cancel(self, task_id: str) -> bool:
        canceled = self.store.cancel(task_id)
        if canceled:
            running = self._running.get(task_id)
            if running is not None:
                _, agent = running
                agent.cancel()
            self._schedule_terminal_notification(task_id)
        return canceled

    async def _worker(
        self,
        index: int,
        stop: asyncio.Event,
        owner_token: str,
        owner_pid: int,
    ) -> None:
        while not stop.is_set():
            try:
                task = await asyncio.to_thread(self.store.claim, owner_token, owner_pid)
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(1)
                continue
            if task is None:
                await asyncio.sleep(0.25)
                continue
            agent: Any | None = None
            try:
                agent = self.agent_factory()
                self._running[task.id] = (owner_token, agent)
                output = await agent.run(task.prompt)
                if stop.is_set():
                    raise asyncio.CancelledError
                updated = await asyncio.to_thread(
                    self.store.update,
                    task.id,
                    "completed",
                    output,
                    owner_token=owner_token,
                )
                if updated:
                    await self._notify_terminal(task.id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not stop.is_set():
                    try:
                        updated = await asyncio.to_thread(
                            self.store.update,
                            task.id,
                            "failed",
                            "",
                            safe_text(
                                exc,
                                fallback=f"{type(exc).__name__} message unavailable",
                            ),
                            owner_token=owner_token,
                        )
                        if updated:
                            await self._notify_terminal(task.id)
                    except Exception:
                        pass
            finally:
                current = self._running.get(task.id)
                if current is not None and current[0] == owner_token and current[1] is agent:
                    self._running.pop(task.id, None)
                if agent is not None:
                    try:
                        await agent.tools.close()
                    except Exception:
                        pass

    async def _notify_terminal(self, task_id: str) -> None:
        try:
            callback = self.terminal_callback
            if callback is None:
                return
            task = await asyncio.to_thread(self.store.get, task_id)
            if task is None or task.status not in _TERMINAL_STATUSES:
                return
            result = callback(task)
            if result is not None:
                await result
        except Exception:
            pass

    def _schedule_terminal_notification(self, task_id: str) -> None:
        if self.terminal_callback is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._notify_terminal(task_id), name=f"kairo-task-notify-{task_id}")
        self._notification_tasks.add(task)
        task.add_done_callback(self._finish_notification_task)

    def _finish_notification_task(self, task: asyncio.Task[None]) -> None:
        self._notification_tasks.discard(task)
        try:
            task.exception()
        except (asyncio.CancelledError, Exception):
            pass


def handle_task_command(
    payload: str | None,
    tasks: DurableTaskStore,
    task_manager: DurableTaskManager,
) -> str:
    normalized = (payload or "list").strip()
    operation, _, argument = normalized.partition(" ")
    operation = operation.casefold()
    argument = argument.strip()
    if operation == "list":
        limit = int(argument) if argument.isdigit() else 20
        listed = tasks.list(limit)
        if not listed:
            return "No background tasks."
        lines = [f"Recent {len(listed)} background tasks:"]
        lines.extend(
            f"{task.id}  {task.status:10}  {task.duration_ms}ms  {task.prompt[:60]}"
            for task in listed
        )
        return "\n".join(lines)
    if operation == "add" and argument:
        try:
            created = tasks.add(argument)
        except ValueError as exc:
            return "Task rejected: " + safe_redacted_text(
                exc, MAX_TASK_ERROR_CHARS, "...[task error truncated]"
            )
        return f"Background task enqueued: {created.id}\nUse /task log {created.id} to inspect it."
    if operation == "cancel" and argument:
        return (
            f"Cancellation requested: {argument}"
            if task_manager.cancel(argument)
            else f"No cancelable background task: {argument}"
        )
    if operation == "log" and argument:
        found = tasks.get(argument)
        if found is None:
            return f"Background task not found: {argument}"
        lines = [
            f"Background task {found.id}",
            f"Status: {found.status}",
            f"Created: {found.created_at}",
        ]
        if found.started_at:
            lines.append(f"Started: {found.started_at}")
        if found.finished_at:
            lines.append(f"Finished: {found.finished_at} ({found.duration_ms}ms)")
        lines.extend(("", "Task:", found.prompt))
        if found.error:
            lines.extend(("", "Error:", found.error))
        if found.output:
            lines.extend(("", "Result:", found.output))
        return "\n".join(lines)
    return "Usage: /task [list [N]|add TASK|cancel ID|log ID]"


def _valid_task_id(task_id: str) -> bool:
    return bool(_TASK_ID.fullmatch(task_id))


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


async def _cancel_manager_tasks(
    tasks: tuple[asyncio.Task[Any], ...] | list[asyncio.Task[Any]],
) -> None:
    active = tuple(task for task in tasks if not task.done())
    for task in active:
        task.cancel()
    if not active:
        return
    done, pending = await asyncio.wait(active, timeout=TASK_SHUTDOWN_GRACE_SECONDS)
    if done:
        await asyncio.gather(*done, return_exceptions=True)
    for task in pending:
        _DETACHED_MANAGER_TASKS.add(task)
        task.add_done_callback(_finish_detached_manager_task)


async def _await_manager_shutdown(task: asyncio.Task[None]) -> None:
    canceled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            canceled = True
    task.result()
    if canceled:
        raise asyncio.CancelledError


def _finish_detached_manager_task(task: asyncio.Task[Any]) -> None:
    _DETACHED_MANAGER_TASKS.discard(task)
    try:
        task.exception()
    except (asyncio.CancelledError, Exception):
        pass


def _duration_ms(started_at: str, now: datetime, finished_at: str) -> int:
    if not started_at or not finished_at:
        return 0
    try:
        return max(0, int((now - datetime.fromisoformat(started_at)).total_seconds() * 1000))
    except (TypeError, ValueError):
        return 0
