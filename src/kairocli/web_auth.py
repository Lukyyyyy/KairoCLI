# FastAPI intentionally declares dependency providers in callable defaults.
# ruff: noqa: B008

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer

from .mail import validate_email
from .paths import reject_symlink_components

JWT_ALGORITHM = "HS256"
JWT_EXPIRY_MINUTES = 60 * 8
MAX_PASSWORD_CHARS = 1_024
MIN_PASSWORD_CHARS = 8

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


@dataclass(slots=True)
class WebUser:
    id: str
    email: str
    hashed_password: str
    is_admin: bool
    created_at: str
    auth_version: int = 0
    must_change_password: bool = False


def hash_password(plain: str) -> str:
    import bcrypt

    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    import bcrypt

    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


class WebUserStore:
    def __init__(self, database: Path) -> None:
        self.database = database
        self.legacy_reset = False
        reject_symlink_components(database, "Users database")
        database.parent.mkdir(parents=True, exist_ok=True)
        reject_symlink_components(database.parent, "Users database")
        if os.name != "nt":
            database.parent.chmod(0o700)
        with self._connect() as connection:
            user_columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(users)").fetchall()
            }
            if "username" in user_columns:
                self.legacy_reset = True
                connection.execute("PRAGMA foreign_keys=OFF")
                for table in (
                    "channel_inbox",
                    "channel_threads",
                    "wechat_binding_details",
                    "channel_bindings",
                    "usage_ledger",
                    "user_quotas",
                    "billing_settings",
                    "user_configs",
                    "user_config_presets",
                    "user_workspaces",
                    "users",
                ):
                    connection.execute(f"DROP TABLE IF EXISTS {table}")
                connection.execute("PRAGMA foreign_keys=ON")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                hashed_password TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                auth_version INTEGER NOT NULL DEFAULT 0,
                must_change_password INTEGER NOT NULL DEFAULT 0
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS user_configs (
                user_id TEXT PRIMARY KEY,
                config_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS user_config_presets (
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                config_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (user_id, name)
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS user_workspaces (
                user_id TEXT NOT NULL,
                path TEXT NOT NULL,
                added_at TEXT NOT NULL,
                removed INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (user_id, path)
                )"""
            )
            workspace_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(user_workspaces)").fetchall()
            }
            if "removed" not in workspace_columns:
                connection.execute(
                    "ALTER TABLE user_workspaces ADD COLUMN removed INTEGER NOT NULL DEFAULT 0"
                )
        if os.name != "nt" and database.is_file():
            database.chmod(0o600)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        _check_db_path(self.database)
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
            _harden_db(self.database)

    def create_user(
        self,
        email: str,
        password: str,
        *,
        is_admin: bool = False,
        must_change_password: bool = False,
    ) -> WebUser:
        normalized_email = validate_email(email)
        if len(password) < MIN_PASSWORD_CHARS:
            raise ValueError(f"密码至少需要 {MIN_PASSWORD_CHARS} 位")
        if len(password) > MAX_PASSWORD_CHARS:
            raise ValueError("密码过长")
        user_id = f"user_{uuid.uuid4().hex}"
        hashed = hash_password(password)
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            try:
                connection.execute(
                    """INSERT INTO users
                    (id, email, hashed_password, is_admin, created_at, must_change_password)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        user_id,
                        normalized_email,
                        hashed,
                        int(is_admin),
                        now,
                        int(must_change_password),
                    ),
                )
            except sqlite3.IntegrityError:
                raise ValueError("该邮箱已被注册") from None
        return WebUser(
            id=user_id,
            email=normalized_email,
            hashed_password=hashed,
            is_admin=is_admin,
            created_at=now,
            must_change_password=must_change_password,
        )

    def get_by_email(self, email: str) -> WebUser | None:
        normalized_email = validate_email(email)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id,email,hashed_password,is_admin,created_at,auth_version,"
                "must_change_password "
                "FROM users WHERE email=?",
                (normalized_email,),
            ).fetchone()
        return _row_to_user(row) if row is not None else None

    def get_by_id(self, user_id: str) -> WebUser | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id,email,hashed_password,is_admin,created_at,auth_version,"
                "must_change_password "
                "FROM users WHERE id=?",
                (user_id,),
            ).fetchone()
        return _row_to_user(row) if row is not None else None

    def list_users(self) -> list[WebUser]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id,email,hashed_password,is_admin,created_at,auth_version,"
                "must_change_password "
                "FROM users ORDER BY created_at"
            ).fetchall()
        return [_row_to_user(row) for row in rows]

    def update_password(
        self, user_id: str, new_password: str, *, must_change_password: bool = False
    ) -> bool:
        if len(new_password) < MIN_PASSWORD_CHARS:
            raise ValueError(f"密码至少需要 {MIN_PASSWORD_CHARS} 位")
        hashed = hash_password(new_password)
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE users SET hashed_password=?,auth_version=auth_version+1,"
                "must_change_password=? WHERE id=?",
                (hashed, int(must_change_password), user_id),
            )
        return cursor.rowcount == 1

    def get_config(self, user_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT config_json FROM user_configs WHERE user_id=?", (user_id,)
            ).fetchone()
        if row is None:
            return {}
        try:
            value = json.loads(str(row[0]))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def save_config(self, user_id: str, config: dict[str, Any]) -> None:
        now = datetime.now(UTC).isoformat()
        payload = json.dumps(config, ensure_ascii=False, separators=(",", ":"))
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO user_configs (user_id, config_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    config_json=excluded.config_json,
                    updated_at=excluded.updated_at""",
                (user_id, payload, now),
            )

    def list_config_presets(self, user_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT name, config_json, created_at
                FROM user_config_presets WHERE user_id=?
                ORDER BY created_at DESC""",
                (user_id,),
            ).fetchall()
        presets: list[dict[str, Any]] = []
        for name, config_json, created_at in rows:
            try:
                config = json.loads(str(config_json))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(config, dict):
                presets.append({"name": str(name), "created_at": str(created_at), **config})
        return presets

    def save_config_preset(self, user_id: str, name: str, config: dict[str, Any]) -> dict[str, Any]:
        created_at = datetime.now(UTC).isoformat()
        payload = json.dumps(config, ensure_ascii=False, separators=(",", ":"))
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO user_config_presets
                (user_id, name, config_json, created_at) VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, name) DO UPDATE SET
                    config_json=excluded.config_json,
                    created_at=excluded.created_at""",
                (user_id, name, payload, created_at),
            )
        return {"name": name, "created_at": created_at, **config}

    def get_config_preset(self, user_id: str, name: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT config_json FROM user_config_presets
                WHERE user_id=? AND name=?""",
                (user_id, name),
            ).fetchone()
        if row is None:
            return None
        try:
            value = json.loads(str(row[0]))
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def delete_config_preset(self, user_id: str, name: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM user_config_presets WHERE user_id=? AND name=?",
                (user_id, name),
            )
        return cursor.rowcount == 1

    def list_workspaces(self, user_id: str) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT path FROM user_workspaces WHERE user_id=? AND removed=0 ORDER BY added_at",
                (user_id,),
            ).fetchall()
        return [str(row[0]) for row in rows]

    def removed_workspaces(self, user_id: str) -> set[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT path FROM user_workspaces WHERE user_id=? AND removed=1",
                (user_id,),
            ).fetchall()
        return {str(row[0]) for row in rows}

    def add_workspace(self, user_id: str, path: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO user_workspaces (user_id, path, added_at, removed)
                VALUES (?, ?, ?, 0)
                ON CONFLICT(user_id, path) DO UPDATE SET
                    added_at=excluded.added_at,
                    removed=0""",
                (user_id, path, datetime.now(UTC).isoformat()),
            )

    def remove_workspace(self, user_id: str, path: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO user_workspaces (user_id, path, added_at, removed)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(user_id, path) DO UPDATE SET
                    added_at=excluded.added_at,
                    removed=1""",
                (user_id, path, datetime.now(UTC).isoformat()),
            )

    def delete_user(self, user_id: str) -> bool:
        with self._connect() as connection:
            connection.execute("DELETE FROM user_configs WHERE user_id=?", (user_id,))
            connection.execute("DELETE FROM user_config_presets WHERE user_id=?", (user_id,))
            connection.execute("DELETE FROM user_workspaces WHERE user_id=?", (user_id,))
            cursor = connection.execute("DELETE FROM users WHERE id=?", (user_id,))
        return cursor.rowcount == 1

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM users").fetchone()
        return int(row[0]) if row else 0


def _row_to_user(row: tuple[Any, ...]) -> WebUser:
    return WebUser(
        id=str(row[0]),
        email=str(row[1]),
        hashed_password=str(row[2]),
        is_admin=bool(row[3]),
        created_at=str(row[4]),
        auth_version=int(row[5]),
        must_change_password=bool(row[6]),
    )


def _check_db_path(path: Path) -> None:
    reject_symlink_components(path.parent, "Users database")
    if path.is_symlink():
        raise ValueError(f"Users database file cannot be a symlink: {path.name}")


def _harden_db(path: Path) -> None:
    if os.name == "nt":
        return
    for p in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        if p.is_symlink():
            raise ValueError(f"Users database file cannot be a symlink: {p.name}")
        if p.is_file():
            p.chmod(0o600)


class JwtSecretStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        reject_symlink_components(path, "JWT secret")
        path.parent.mkdir(parents=True, exist_ok=True)
        reject_symlink_components(path.parent, "JWT secret")
        if os.name != "nt":
            path.parent.chmod(0o700)

    def load_or_generate(self) -> bytes:
        if self.path.exists():
            if self.path.is_symlink():
                raise ValueError("JWT secret file cannot be a symlink")
            data = self.path.read_bytes()
            if len(data) != 32:
                raise ValueError("JWT secret file is corrupt (expected 32 bytes)")
            return data
        secret = secrets.token_bytes(32)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(str(self.path), flags, 0o600)
        try:
            os.write(fd, secret)
        finally:
            os.close(fd)
        return secret


def create_access_token(
    user_id: str,
    is_admin: bool,
    auth_version: int,
    secret: bytes,
    expiry_minutes: int = JWT_EXPIRY_MINUTES,
) -> str:
    import jwt

    payload = {
        "sub": user_id,
        "is_admin": is_admin,
        "ver": auth_version,
        "exp": datetime.now(UTC) + timedelta(minutes=expiry_minutes),
    }
    return jwt.encode(payload, secret.hex(), algorithm=JWT_ALGORITHM)


def decode_access_token(token: str, secret: bytes) -> dict[str, Any]:
    import jwt

    try:
        return jwt.decode(token, secret.hex(), algorithms=[JWT_ALGORITHM])
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登录凭证无效，请重新登录",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None


def make_get_current_user(user_store: WebUserStore, jwt_secret: bytes) -> Callable[..., Any]:
    async def get_current_user(
        request: Request,
        token: str | None = Depends(_oauth2_scheme),
    ) -> WebUser:
        token = token or request.cookies.get("kairo_session")
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="登录凭证无效，请重新登录",
            )
        payload = decode_access_token(token, jwt_secret)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录凭证无效")
        user = user_store.get_by_id(str(user_id))
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户不存在")
        if payload.get("ver") != user.auth_version:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="登录凭证已失效，请重新登录",
            )
        if user.must_change_password and request.url.path not in {
            "/auth/me",
            "/auth/me/password",
            "/auth/logout",
        }:
            raise HTTPException(status_code=403, detail="请先修改临时密码")
        return user

    return get_current_user


def make_require_admin(get_current_user: Callable[..., Any]) -> Callable[..., Any]:
    async def require_admin(user: WebUser = Depends(get_current_user)) -> WebUser:
        if not user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="需要管理员权限"
            )
        return user

    return require_admin


class LoginRateLimiter:
    """Sliding window: 5 attempts per IP per 60 seconds."""

    def __init__(
        self,
        max_attempts: int = 5,
        window_seconds: float = 60.0,
        message: str = "登录尝试过于频繁，请稍后再试",
    ) -> None:
        self._max = max_attempts
        self._window = window_seconds
        self._message = message
        self._log: dict[str, deque[float]] = {}

    def check_and_record(self, ip: str) -> None:
        now = datetime.now(UTC).timestamp()
        times = self._log.setdefault(ip, deque())
        cutoff = now - self._window
        while times and times[0] < cutoff:
            times.popleft()
        if len(times) >= self._max:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=self._message,
            )
        times.append(now)
