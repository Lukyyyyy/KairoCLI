from __future__ import annotations

import os
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .paths import reject_symlink_components

CNY_UNITS = 100_000_000
DEFAULT_BALANCE_UNITS = CNY_UNITS


@dataclass(frozen=True, slots=True)
class UserQuota:
    user_id: str
    balance_units: int
    monthly_units: int

    @property
    def balance_cny(self) -> str:
        return _format_cny(self.balance_units)

    @property
    def monthly_cny(self) -> str:
        return _format_cny(self.monthly_units)


class BillingStore:
    def __init__(self, database: Path) -> None:
        self.database = database
        reject_symlink_components(database, "Billing database")
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS user_quotas (
                    user_id TEXT PRIMARY KEY,
                    balance_units INTEGER NOT NULL,
                    monthly_units INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS usage_ledger (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    cached_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    cost_units INTEGER NOT NULL,
                    rate_input TEXT NOT NULL,
                    rate_cached TEXT NOT NULL,
                    rate_output TEXT NOT NULL,
                    charged_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_usage_ledger_user_time
                    ON usage_ledger(user_id, charged_at DESC);
                CREATE TABLE IF NOT EXISTS billing_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
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

    def quota(self, user_id: str) -> UserQuota:
        self.apply_monthly_reset()
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO user_quotas
                (user_id,balance_units,monthly_units,updated_at) VALUES (?,?,?,?)""",
                (user_id, DEFAULT_BALANCE_UNITS, DEFAULT_BALANCE_UNITS, now),
            )
            row = connection.execute(
                "SELECT user_id,balance_units,monthly_units FROM user_quotas WHERE user_id=?",
                (user_id,),
            ).fetchone()
        if row is None:
            raise ValueError("用户不存在")
        return UserQuota(str(row[0]), int(row[1]), int(row[2]))

    def monthly_reset_enabled(self) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM billing_settings WHERE key='monthly_reset_enabled'"
            ).fetchone()
        return row is not None and str(row[0]) == "1"

    def set_monthly_reset_enabled(self, enabled: bool) -> None:
        period = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO billing_settings(key,value) VALUES "
                "('monthly_reset_enabled',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                ("1" if enabled else "0",),
            )
            if enabled:
                connection.execute(
                    "INSERT INTO billing_settings(key,value) VALUES "
                    "('monthly_reset_period',?) ON CONFLICT(key) "
                    "DO UPDATE SET value=excluded.value",
                    (period,),
                )

    def apply_monthly_reset(self, at: datetime | None = None) -> bool:
        current = at or datetime.now(ZoneInfo("Asia/Shanghai"))
        if current.tzinfo is None:
            current = current.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
        period = current.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            enabled = connection.execute(
                "SELECT value FROM billing_settings WHERE key='monthly_reset_enabled'"
            ).fetchone()
            previous = connection.execute(
                "SELECT value FROM billing_settings WHERE key='monthly_reset_period'"
            ).fetchone()
            if (
                enabled is None
                or str(enabled[0]) != "1"
                or (previous is not None and str(previous[0]) >= period)
            ):
                return False
            connection.execute(
                "UPDATE user_quotas SET balance_units=monthly_units,updated_at=?",
                (current.astimezone(UTC).isoformat(),),
            )
            connection.execute(
                "INSERT INTO billing_settings(key,value) VALUES "
                "('monthly_reset_period',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (period,),
            )
        return True

    def set_quota(self, user_id: str, balance_units: int, monthly_units: int) -> UserQuota:
        if not -(2**63) < balance_units < 2**63 or not 0 <= monthly_units < 2**63:
            raise ValueError("配额数值超出范围")
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO user_quotas(user_id,balance_units,monthly_units,updated_at)
                VALUES (?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET
                balance_units=excluded.balance_units,
                monthly_units=excluded.monthly_units,
                updated_at=excluded.updated_at""",
                (user_id, balance_units, monthly_units, datetime.now(UTC).isoformat()),
            )
        return self.quota(user_id)

    def charge(
        self,
        user_id: str,
        *,
        provider: str,
        model: str,
        input_tokens: int,
        cached_tokens: int,
        output_tokens: int,
        cost_units: int,
        rates: tuple[str, str, str],
        charged_at: datetime,
        call_id: str | None = None,
    ) -> UserQuota:
        if min(input_tokens, cached_tokens, output_tokens, cost_units) < 0:
            raise ValueError("用量不能为负数")
        ledger_id = call_id or f"llm_{uuid.uuid4().hex}"
        now = charged_at.astimezone(UTC).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT OR IGNORE INTO user_quotas
                (user_id,balance_units,monthly_units,updated_at) VALUES (?,?,?,?)""",
                (user_id, DEFAULT_BALANCE_UNITS, DEFAULT_BALANCE_UNITS, now),
            )
            cursor = connection.execute(
                """INSERT OR IGNORE INTO usage_ledger
                (id,user_id,provider,model,input_tokens,cached_tokens,output_tokens,
                 cost_units,rate_input,rate_cached,rate_output,charged_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    ledger_id,
                    user_id,
                    provider,
                    model,
                    input_tokens,
                    cached_tokens,
                    output_tokens,
                    cost_units,
                    *rates,
                    now,
                ),
            )
            if cursor.rowcount:
                connection.execute(
                    "UPDATE user_quotas SET balance_units=balance_units-?,updated_at=? "
                    "WHERE user_id=?",
                    (cost_units, now, user_id),
                )
            row = connection.execute(
                "SELECT user_id,balance_units,monthly_units FROM user_quotas WHERE user_id=?",
                (user_id,),
            ).fetchone()
        if row is None:  # pragma: no cover - protected by the transaction
            raise RuntimeError("Quota disappeared")
        return UserQuota(str(row[0]), int(row[1]), int(row[2]))

    def list_usage(self, user_id: str, limit: int = 100) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT id,provider,model,input_tokens,cached_tokens,output_tokens,
                cost_units,charged_at FROM usage_ledger WHERE user_id=?
                ORDER BY charged_at DESC LIMIT ?""",
                (user_id, max(1, min(limit, 500))),
            ).fetchall()
        return [
            {
                "id": str(row[0]),
                "provider": str(row[1]),
                "model": str(row[2]),
                "input_tokens": int(row[3]),
                "cached_tokens": int(row[4]),
                "output_tokens": int(row[5]),
                "cost_cny": _format_cny(int(row[6])),
                "charged_at": str(row[7]),
            }
            for row in rows
        ]

    def _check_paths(self) -> None:
        reject_symlink_components(self.database.parent, "Billing database")
        for path in self._database_files():
            if path.is_symlink():
                raise ValueError(f"Billing database cannot use a symlink: {path.name}")

    def _harden(self) -> None:
        if os.name == "nt":
            return
        for path in self._database_files():
            if path.is_symlink():
                raise ValueError(f"Billing database cannot use a symlink: {path.name}")
            if path.is_file():
                path.chmod(0o600)

    def _database_files(self) -> tuple[Path, Path, Path]:
        return self.database, Path(f"{self.database}-wal"), Path(f"{self.database}-shm")


def cny_to_units(value: str) -> int:
    from decimal import Decimal, InvalidOperation

    try:
        amount = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("金额无效") from exc
    units = amount * CNY_UNITS
    if not amount.is_finite() or units != units.to_integral_value():
        raise ValueError("金额最多支持 8 位小数")
    return int(units)


def _format_cny(units: int) -> str:
    sign = "-" if units < 0 else ""
    whole, fraction = divmod(abs(units), CNY_UNITS)
    decimals = f"{fraction:08d}".rstrip("0").ljust(2, "0")
    return f"{sign}{whole}.{decimals}"
