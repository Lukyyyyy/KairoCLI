from __future__ import annotations

import json
import os
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ..paths import reject_symlink_components


@dataclass(frozen=True, slots=True)
class ChannelBinding:
    id: str
    user_id: str
    channel_type: str
    status: str
    enabled: bool
    active_workspace: str
    external_account_id: str
    external_user_id: str
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class WechatCredentials:
    token: str
    base_url: str
    sync_buf: str


class ChannelStore:
    """Private SQLite state shared by Web channel management and workers."""

    def __init__(self, database: Path) -> None:
        self.database = database
        reject_symlink_components(database, "Channel database")
        database.parent.mkdir(parents=True, exist_ok=True)
        reject_symlink_components(database.parent, "Channel database")
        if os.name != "nt":
            database.parent.chmod(0o700)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS channel_bindings (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    channel_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 0,
                    active_workspace TEXT NOT NULL DEFAULT '',
                    external_account_id TEXT NOT NULL DEFAULT '',
                    external_user_id TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    disconnected_at TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
                    UNIQUE(channel_type, external_user_id)
                );
                CREATE INDEX IF NOT EXISTS idx_channel_bindings_user
                    ON channel_bindings(user_id, channel_type);
                CREATE TABLE IF NOT EXISTS wechat_binding_details (
                    binding_id TEXT PRIMARY KEY,
                    token TEXT NOT NULL,
                    base_url TEXT NOT NULL,
                    sync_buf TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY(binding_id) REFERENCES channel_bindings(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS channel_threads (
                    binding_id TEXT NOT NULL,
                    workspace TEXT NOT NULL,
                    thread_id TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(binding_id, workspace),
                    FOREIGN KEY(binding_id) REFERENCES channel_bindings(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS channel_inbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    binding_id TEXT NOT NULL,
                    external_message_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(binding_id, external_message_id),
                    FOREIGN KEY(binding_id) REFERENCES channel_bindings(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_channel_inbox_pending
                    ON channel_inbox(binding_id, status, id);
                """
            )
        self._harden()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self._check_paths()
        connection = sqlite3.connect(self.database, timeout=30)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=30000")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
            self._harden()

    def binding_for_user(self, user_id: str, channel_type: str) -> ChannelBinding | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id,user_id,channel_type,status,enabled,active_workspace,"
                "external_account_id,external_user_id,created_at,updated_at "
                "FROM channel_bindings WHERE user_id=? AND channel_type=? "
                "ORDER BY created_at LIMIT 1",
                (user_id, channel_type),
            ).fetchone()
        return _binding(row) if row is not None else None

    def binding(self, binding_id: str) -> ChannelBinding | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id,user_id,channel_type,status,enabled,active_workspace,"
                "external_account_id,external_user_id,created_at,updated_at "
                "FROM channel_bindings WHERE id=?",
                (binding_id,),
            ).fetchone()
        return _binding(row) if row is not None else None

    def list_bindings(self, *, enabled_only: bool = False) -> list[ChannelBinding]:
        clause = " WHERE enabled=1" if enabled_only else ""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id,user_id,channel_type,status,enabled,active_workspace,"
                "external_account_id,external_user_id,created_at,updated_at "
                f"FROM channel_bindings{clause} ORDER BY created_at"
            ).fetchall()
        return [_binding(row) for row in rows]

    def save_wechat_binding(
        self,
        user_id: str,
        *,
        token: str,
        account_id: str,
        external_user_id: str,
        base_url: str,
        workspace: str,
    ) -> ChannelBinding:
        if not all((token, account_id, external_user_id, base_url, workspace)):
            raise ValueError("WeChat binding is incomplete")
        if (
            len(token) > 16_384
            or len(account_id) > 1_024
            or len(external_user_id) > 1_024
            or len(base_url) > 8_192
            or len(workspace) > 4_096
        ):
            raise ValueError("WeChat binding fields are too long")
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT id FROM channel_bindings WHERE user_id=? AND channel_type='wechat'",
                (user_id,),
            ).fetchone()
            binding_id = str(existing[0]) if existing is not None else f"binding_{uuid.uuid4().hex}"
            try:
                connection.execute(
                    """INSERT INTO channel_bindings
                    (id,user_id,channel_type,status,enabled,active_workspace,
                     external_account_id,external_user_id,created_at,updated_at,disconnected_at)
                    VALUES (?,?,'wechat','connected',0,?,?,?,?,?,NULL)
                    ON CONFLICT(id) DO UPDATE SET
                      status='connected', enabled=0, active_workspace=excluded.active_workspace,
                      external_account_id=excluded.external_account_id,
                      external_user_id=excluded.external_user_id,
                      updated_at=excluded.updated_at, disconnected_at=NULL""",
                    (
                        binding_id,
                        user_id,
                        workspace,
                        account_id,
                        external_user_id,
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError:
                raise ValueError("This WeChat identity is already bound") from None
            connection.execute(
                """INSERT INTO wechat_binding_details(binding_id,token,base_url,sync_buf)
                VALUES (?,?,?,'') ON CONFLICT(binding_id) DO UPDATE SET
                token=excluded.token,base_url=excluded.base_url,sync_buf=''""",
                (binding_id, token, base_url),
            )
        binding = self.binding(binding_id)
        if binding is None:  # pragma: no cover - protected by the transaction
            raise RuntimeError("Failed to save WeChat binding")
        return binding

    def wechat_credentials(self, binding_id: str) -> WechatCredentials | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT token,base_url,sync_buf FROM wechat_binding_details WHERE binding_id=?",
                (binding_id,),
            ).fetchone()
        return WechatCredentials(str(row[0]), str(row[1]), str(row[2])) if row else None

    def set_enabled(self, binding_id: str, enabled: bool) -> bool:
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE channel_bindings SET enabled=?,status=?,updated_at=? "
                "WHERE id=? AND disconnected_at IS NULL",
                (int(enabled), "active" if enabled else "connected", now, binding_id),
            )
        return cursor.rowcount == 1

    def set_workspace(self, binding_id: str, workspace: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE channel_bindings SET active_workspace=?,updated_at=? WHERE id=?",
                (workspace, datetime.now(UTC).isoformat(), binding_id),
            )
        return cursor.rowcount == 1

    def update_sync_buf(self, binding_id: str, sync_buf: str) -> bool:
        if len(sync_buf) > 65_536:
            raise ValueError("WeChat sync buffer exceeds 64 KiB")
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE wechat_binding_details SET sync_buf=? WHERE binding_id=?",
                (sync_buf, binding_id),
            )
        return cursor.rowcount == 1

    def disconnect(self, binding_id: str) -> bool:
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM wechat_binding_details WHERE binding_id=?", (binding_id,)
            )
            cursor = connection.execute(
                "UPDATE channel_bindings SET status='disconnected',enabled=0,"
                "updated_at=?,disconnected_at=? WHERE id=?",
                (now, now, binding_id),
            )
        return cursor.rowcount == 1

    def thread_for_workspace(self, binding_id: str, workspace: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT thread_id FROM channel_threads WHERE binding_id=? AND workspace=?",
                (binding_id, workspace),
            ).fetchone()
        return str(row[0]) if row is not None else None

    def save_thread(self, binding_id: str, workspace: str, thread_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO channel_threads(binding_id,workspace,thread_id,created_at)
                VALUES (?,?,?,?) ON CONFLICT(binding_id,workspace) DO UPDATE SET
                thread_id=excluded.thread_id""",
                (binding_id, workspace, thread_id, datetime.now(UTC).isoformat()),
            )

    def expired_threads(
        self, retention_days: int, *, at: datetime | None = None
    ) -> list[tuple[str, str, str]]:
        if retention_days < 1:
            raise ValueError("Channel history retention must be at least one day")
        current = at or datetime.now(UTC)
        if current.tzinfo is None:
            current = current.replace(tzinfo=UTC)
        cutoff = (current.astimezone(UTC) - timedelta(days=retention_days)).isoformat()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT t.binding_id,t.thread_id,b.user_id FROM channel_threads t "
                "JOIN channel_bindings b ON b.id=t.binding_id "
                "WHERE b.disconnected_at IS NOT NULL AND b.disconnected_at<?",
                (cutoff,),
            ).fetchall()
        return [(str(row[0]), str(row[1]), str(row[2])) for row in rows]

    def delete_thread_mapping(self, binding_id: str, thread_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM channel_threads WHERE binding_id=? AND thread_id=?",
                (binding_id, thread_id),
            )

    def enqueue_messages(
        self, binding_id: str, messages: list[dict[str, Any]], sync_buf: str
    ) -> int:
        now = datetime.now(UTC).isoformat()
        inserted = 0
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for message in messages:
                message_id = str(message.get("message_id", ""))
                if not message_id:
                    continue
                cursor = connection.execute(
                    """INSERT OR IGNORE INTO channel_inbox
                    (binding_id,external_message_id,payload_json,status,attempts,created_at,updated_at)
                    VALUES (?,?,?,'pending',0,?,?)""",
                    (
                        binding_id,
                        message_id,
                        json.dumps(message, ensure_ascii=False, separators=(",", ":")),
                        now,
                        now,
                    ),
                )
                inserted += cursor.rowcount
            connection.execute(
                "UPDATE wechat_binding_details SET sync_buf=? WHERE binding_id=?",
                (sync_buf, binding_id),
            )
        return inserted

    def requeue_running(self, binding_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE channel_inbox SET status='pending',updated_at=? "
                "WHERE binding_id=? AND status='running'",
                (datetime.now(UTC).isoformat(), binding_id),
            )

    def claim_message(self, binding_id: str) -> tuple[int, dict[str, Any]] | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT id,payload_json FROM channel_inbox "
                "WHERE binding_id=? AND status='pending' ORDER BY id LIMIT 1",
                (binding_id,),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE channel_inbox SET status='running',attempts=attempts+1,updated_at=? "
                "WHERE id=?",
                (datetime.now(UTC).isoformat(), int(row[0])),
            )
        payload = json.loads(str(row[1]))
        if not isinstance(payload, dict):
            self.finish_message(int(row[0]), "failed")
            return None
        return int(row[0]), payload

    def finish_message(self, inbox_id: int, status: str) -> None:
        if status not in {"completed", "failed", "canceled"}:
            raise ValueError("Invalid inbox status")
        with self._connect() as connection:
            connection.execute(
                "UPDATE channel_inbox SET status=?,updated_at=? WHERE id=?",
                (status, datetime.now(UTC).isoformat(), inbox_id),
            )

    def _check_paths(self) -> None:
        reject_symlink_components(self.database.parent, "Channel database")
        for path in self._database_files():
            if path.is_symlink():
                raise ValueError(f"Channel database cannot use a symlink: {path.name}")

    def _harden(self) -> None:
        if os.name == "nt":
            return
        for path in self._database_files():
            if path.is_symlink():
                raise ValueError(f"Channel database cannot use a symlink: {path.name}")
            if path.is_file():
                path.chmod(0o600)

    def _database_files(self) -> tuple[Path, Path, Path]:
        return self.database, Path(f"{self.database}-wal"), Path(f"{self.database}-shm")


def _binding(row: sqlite3.Row) -> ChannelBinding:
    return ChannelBinding(
        id=str(row[0]),
        user_id=str(row[1]),
        channel_type=str(row[2]),
        status=str(row[3]),
        enabled=bool(row[4]),
        active_workspace=str(row[5]),
        external_account_id=str(row[6]),
        external_user_id=str(row[7]),
        created_at=str(row[8]),
        updated_at=str(row[9]),
    )
