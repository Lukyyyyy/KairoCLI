from __future__ import annotations

import asyncio
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..cancellation import AgentCanceled, wait_with_cancellation
from .process import _terminate_process_tree

SHELL_ID_PATTERN = re.compile(r"shell_[0-9a-f]{12}$")
MAX_SHELL_SESSIONS = 4
MAX_SHELL_OUTPUT_BYTES = 50_000
MAX_SHELL_COMMAND_BYTES = 100_000
SHELL_IDLE_SECONDS = 30 * 60


@dataclass(slots=True)
class _ShellSession:
    id: str
    process: asyncio.subprocess.Process
    cwd: Path
    created_at: float
    last_used: float
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending: bytearray = field(default_factory=bytearray)


@dataclass(slots=True)
class _Collector:
    limit: int = MAX_SHELL_OUTPUT_BYTES
    total: int = 0
    head: bytearray = field(default_factory=bytearray)
    tail: bytearray = field(default_factory=bytearray)

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.total += len(chunk)
        head_limit = self.limit * 2 // 3
        tail_limit = self.limit - head_limit
        if len(self.head) < head_limit:
            take = min(head_limit - len(self.head), len(chunk))
            self.head.extend(chunk[:take])
            chunk = chunk[take:]
        if chunk and tail_limit:
            self.tail.extend(chunk)
            if len(self.tail) > tail_limit:
                del self.tail[: len(self.tail) - tail_limit]

    def result(self) -> tuple[str, int, bool]:
        truncated = self.total > self.limit
        if truncated:
            value = (
                bytes(self.head)
                + b"\n...[output truncated; middle omitted]...\n"
                + bytes(self.tail)
            )
        else:
            value = bytes(self.head + self.tail)
        return value.decode(errors="replace"), self.total, truncated


class ShellSessionManager:
    """Bounded stateful shells for commands that need persistent cwd or environment."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self._sessions: dict[str, _ShellSession] = {}
        self._registry_lock = asyncio.Lock()
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None

    async def start(self, cwd: Path | None = None) -> dict[str, Any]:
        self._ensure_open()
        await self._prune_idle()
        target = (cwd or self.workspace).resolve()
        try:
            relative_cwd = target.relative_to(self.workspace)
        except ValueError as exc:
            raise ValueError("Shell cwd must stay within the workspace") from exc
        if not target.is_dir():
            raise ValueError("Shell cwd must be an existing directory")
        async with self._registry_lock:
            self._ensure_open()
            if len(self._sessions) >= MAX_SHELL_SESSIONS:
                raise ValueError(
                    f"At most {MAX_SHELL_SESSIONS} persistent shell sessions may be active"
                )
            options: dict[str, Any] = {}
            if os.name == "posix":
                executable = "/bin/sh"
                arguments: tuple[str, ...] = ()
                options["start_new_session"] = True
            elif os.name == "nt":  # pragma: no cover - exercised on Windows
                executable = os.environ.get("COMSPEC", "cmd.exe")
                arguments = ("/Q", "/D")
                options["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            else:  # pragma: no cover - unsupported Python platform
                raise RuntimeError("Persistent shell sessions are unsupported on this platform")
            process = await asyncio.create_subprocess_exec(
                executable,
                *arguments,
                cwd=target,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                **options,
            )
            if self._closing:
                await _terminate_process_tree(process)
                raise RuntimeError("Shell session manager is closed")
            now = time.monotonic()
            identifier = f"shell_{uuid.uuid4().hex[:12]}"
            self._sessions[identifier] = _ShellSession(identifier, process, target, now, now)
        return {
            "session_id": identifier,
            "cwd": str(relative_cwd or "."),
            "shell": executable,
        }

    async def execute(
        self,
        session_id: str,
        command: str,
        timeout_seconds: float,
        cancel_event: asyncio.Event | None,
    ) -> dict[str, Any]:
        self._ensure_open()
        _validate_shell_id(session_id)
        if not command.strip():
            raise ValueError("Shell command cannot be empty")
        if len(command.encode()) > MAX_SHELL_COMMAND_BYTES:
            raise ValueError("Shell command exceeds the 100 KiB limit")
        await self._prune_idle()
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError("Shell session does not exist or has expired")
        async with session.lock:
            if session.process.returncode is not None:
                self._sessions.pop(session_id, None)
                raise ValueError("Shell session has already exited")
            nonce = uuid.uuid4().hex
            marker = f"__KAIROCLI_DONE_{nonce}__:".encode()
            payload = self._wrap_command(command, marker.decode())
            assert session.process.stdin is not None
            try:
                session.process.stdin.write(payload.encode())
                await session.process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                await self._discard(session_id, session)
                raise ValueError("Shell session exited before accepting the command") from exc
            try:
                collector = _Collector()
                result = await wait_with_cancellation(
                    asyncio.wait_for(
                        self._read_until_marker(session, marker, collector),
                        timeout_seconds,
                    ),
                    cancel_event,
                )
            except TimeoutError:
                await self._discard(session_id, session)
                await self._drain_closed_output(session, collector)
                output, total, truncated = collector.result()
                return {
                    "session_id": session_id,
                    "exit_code": None,
                    "timed_out": True,
                    "session_closed": True,
                    "stdout": output,
                    "stdout_bytes": total,
                    "stdout_truncated": truncated,
                    "error": f"Command exceeded {timeout_seconds:g}s; shell session was terminated",
                }
            except (AgentCanceled, asyncio.CancelledError):
                await self._discard(session_id, session)
                await self._drain_closed_output(session, collector)
                raise
            output, exit_code, closed, total, truncated = result
            session.last_used = time.monotonic()
            if closed:
                self._sessions.pop(session_id, None)
            return {
                "session_id": session_id,
                "exit_code": exit_code,
                "timed_out": False,
                "session_closed": closed,
                "stdout": output,
                "stdout_bytes": total,
                "stdout_truncated": truncated,
            }

    async def list(self) -> list[dict[str, Any]]:
        if self._closing:
            return []
        await self._prune_idle()
        now = time.monotonic()
        return [
            {
                "session_id": item.id,
                "cwd": str(item.cwd.relative_to(self.workspace) or "."),
                "idle_seconds": int(now - item.last_used),
                "running": item.process.returncode is None,
            }
            for item in self._sessions.values()
        ]

    async def stop(self, session_id: str) -> bool:
        if self._closing:
            return False
        _validate_shell_id(session_id)
        session = self._sessions.get(session_id)
        if session is None:
            return False
        await self._discard(session_id, session)
        return True

    async def close(self) -> None:
        self._closing = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._close_once(), name="kairo-shell-manager-shutdown"
            )
        await _await_shell_shutdown(self._close_task)

    async def _close_once(self) -> None:
        async with self._registry_lock:
            sessions = tuple(self._sessions.items())
            self._sessions.clear()
        await asyncio.gather(
            *(_terminate_process_tree(session.process) for _, session in sessions),
            return_exceptions=True,
        )

    def _ensure_open(self) -> None:
        if self._closing:
            raise RuntimeError("Shell session manager is closed")

    async def _read_until_marker(
        self, session: _ShellSession, marker: bytes, collector: _Collector
    ) -> tuple[str, int | None, bool, int, bool]:
        assert session.process.stdout is not None
        buffer = bytes(session.pending)
        session.pending.clear()
        try:
            while True:
                marker_index = buffer.find(marker)
                if marker_index >= 0:
                    newline = buffer.find(b"\n", marker_index + len(marker))
                    if newline < 0:
                        chunk = await session.process.stdout.read(4_096)
                        if not chunk:
                            collector.feed(buffer)
                            output, total, truncated = collector.result()
                            return output, None, True, total, truncated
                        buffer += chunk
                        continue
                    collector.feed(buffer[:marker_index])
                    raw_status = buffer[marker_index + len(marker) : newline].strip(b"\r ")
                    try:
                        exit_code = int(raw_status)
                    except ValueError:
                        exit_code = None
                    session.pending.extend(buffer[newline + 1 :])
                    output, total, truncated = collector.result()
                    return output, exit_code, False, total, truncated
                if len(buffer) > len(marker):
                    safe = len(buffer) - len(marker)
                    collector.feed(buffer[:safe])
                    buffer = buffer[safe:]
                chunk = await session.process.stdout.read(4_096)
                if not chunk:
                    collector.feed(buffer)
                    output, total, truncated = collector.result()
                    return output, session.process.returncode, True, total, truncated
                buffer += chunk
        except asyncio.CancelledError:
            session.pending.extend(buffer)
            raise

    @staticmethod
    async def _drain_closed_output(session: _ShellSession, collector: _Collector) -> None:
        collector.feed(bytes(session.pending))
        session.pending.clear()
        if session.process.stdout is None:
            return

        async def drain() -> None:
            assert session.process.stdout is not None
            while chunk := await session.process.stdout.read(4_096):
                collector.feed(chunk)

        try:
            await asyncio.wait_for(drain(), 1.0)
        except TimeoutError:
            return

    @staticmethod
    def _wrap_command(command: str, marker: str) -> str:
        if os.name == "nt":  # pragma: no cover - exercised on Windows
            return f"{command}\r\n@echo {marker}%errorlevel%\r\n"
        return (
            f"{command}\n"
            "__kairocli_status=$?\n"
            "case $- in *x*) __kairocli_restore_x=1; set +x ;; "
            "*) __kairocli_restore_x=0 ;; esac\n"
            "case $- in *v*) __kairocli_restore_v=1; set +v ;; "
            "*) __kairocli_restore_v=0 ;; esac\n"
            f"printf '\\n{marker}%s\\n' \"$__kairocli_status\"; "
            '[ "$__kairocli_restore_v" -eq 1 ] && set -v; '
            '[ "$__kairocli_restore_x" -eq 1 ] && set -x\n'
        )

    async def _prune_idle(self) -> None:
        now = time.monotonic()
        for identifier, session in tuple(self._sessions.items()):
            if session.lock.locked():
                continue
            if (
                session.process.returncode is not None
                or now - session.last_used > SHELL_IDLE_SECONDS
            ):
                await self._discard(identifier, session)

    async def _discard(self, identifier: str, session: _ShellSession) -> None:
        self._sessions.pop(identifier, None)
        await _terminate_process_tree(session.process)


def _validate_shell_id(session_id: str) -> None:
    if not SHELL_ID_PATTERN.fullmatch(session_id):
        raise ValueError("Invalid shell session ID")


async def _await_shell_shutdown(task: asyncio.Task[None]) -> None:
    canceled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            canceled = True
    task.result()
    if canceled:
        raise asyncio.CancelledError
