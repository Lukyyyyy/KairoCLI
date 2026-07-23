from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import shlex
import shutil
import signal
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .text_safety import safe_text

MAX_LSP_DIAGNOSTICS_PER_FILE = 500
MAX_LSP_DIAGNOSTIC_MESSAGE_CHARS = 4_000
MAX_LSP_CODE_ACTIONS = 50
MAX_LSP_CODE_ACTION_TITLE_CHARS = 1_000
MAX_LSP_CODE_ACTION_DIAGNOSTICS = 50
MAX_LSP_WORKSPACE_FILES = 500
MAX_LSP_WORKSPACE_DIAGNOSTICS = 1_000
MAX_LSP_WORKSPACE_REPORTS = 500
MAX_LSP_PARSER_FILE_BYTES = 1024 * 1024
MAX_LSP_SCAN_ENTRIES = 20_000
MAX_LSP_MESSAGE_BYTES = 10 * 1024 * 1024
MAX_LSP_HEADER_BYTES = 16 * 1024
LSP_CANCEL_GRACE_SECONDS = 0.2
MAX_LSP_JSON_DEPTH = 32
MAX_LSP_JSON_NODES = 200_000
_PARSER_EXTENSIONS = {
    ".py",
    ".json",
    ".toml",
    ".java",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".c",
    ".h",
    ".cc",
    ".cpp",
    ".cxx",
    ".hh",
    ".hpp",
}
_LSP_SCAN_EXCLUDES = {
    ".git",
    ".kairocli",
    ".venv",
    "venv",
    "node_modules",
    "target",
    "dist",
    "build",
    "__pycache__",
}


@dataclass(frozen=True, slots=True)
class Diagnostic:
    path: str
    line: int
    column: int
    severity: str
    message: str
    source: str = "lsp"


@dataclass(frozen=True, slots=True)
class LspServerConfig:
    command: str
    args: tuple[str, ...]
    extensions: tuple[str, ...]
    language_id: str
    initialization_options: dict[str, Any] | None = None


def format_diagnostics(diagnostics: list[Diagnostic], max_items: int | None = None) -> str:
    if max_items is None:
        try:
            max_items = max(1, int(os.getenv("KAIROCLI_LSP_MAX_DIAGNOSTICS", "20")))
        except ValueError:
            max_items = 20
    selected = diagnostics[:max_items]
    lines = [
        f"{item.path}:{item.line}:{item.column}: {item.severity}: {item.message} [{item.source}]"
        for item in selected
    ]
    omitted = len(diagnostics) - len(selected)
    if omitted:
        lines.append(f"... {omitted} additional diagnostics omitted")
    return "\n".join(lines)


class LspClient:
    def __init__(
        self,
        config: LspServerConfig,
        workspace: Path,
        diagnostic_timeout: float = 2,
    ) -> None:
        self.config = config
        self.workspace = workspace.resolve()
        self.diagnostic_timeout = max(0.1, diagnostic_timeout)
        self.process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._events: dict[str, asyncio.Event] = {}
        self._diagnostics: dict[str, list[Diagnostic]] = {}
        self._versions: dict[str, int] = {}
        self._document_locks: dict[str, asyncio.Lock] = {}
        self._request_id = 0
        self._write_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        self._active_operations: dict[asyncio.Task[Any], int] = {}
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._ensure_open()
        async with self._start_lock:
            self._ensure_open()
            if (
                self.process is not None
                and self.process.returncode is None
                and self._reader_task is not None
                and not self._reader_task.done()
            ):
                return
            if self.process is not None:
                await self._abort()
            self.process = await asyncio.create_subprocess_exec(
                self.config.command,
                *self.config.args,
                cwd=self.workspace,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name == "posix",
            )
            if self._closing:
                await self._abort()
                raise RuntimeError("LSP client is closed")
            self._reader_task = asyncio.create_task(self._reader_loop())
            self._stderr_task = asyncio.create_task(self._drain_stderr())
            try:
                await asyncio.wait_for(
                    self._request(
                        "initialize",
                        {
                            "processId": os.getpid(),
                            "rootUri": self.workspace.as_uri(),
                            "capabilities": {
                                "textDocument": {
                                    "publishDiagnostics": {
                                        "relatedInformation": True,
                                        "versionSupport": True,
                                    },
                                    "codeAction": {
                                        "codeActionLiteralSupport": {
                                            "codeActionKind": {"valueSet": ["quickfix", "refactor"]}
                                        }
                                    },
                                    "diagnostic": {"relatedDocumentSupport": False},
                                },
                                "workspace": {"diagnostics": {"refreshSupport": False}},
                            },
                            "initializationOptions": self.config.initialization_options or {},
                        },
                    ),
                    10,
                )
                await self._notify("initialized", {})
            except BaseException:
                await self._abort()
                raise

    async def diagnose(self, path: Path, content: str) -> list[Diagnostic]:
        current_task = self._enter_operation()
        try:
            await self.start()
            uri = await asyncio.to_thread(_path_uri, path)
            lock = self._document_locks.setdefault(uri, asyncio.Lock())
            async with lock:
                self._ensure_open()
                return await self._diagnose_document(uri, content)
        finally:
            self._leave_operation(current_task)

    async def _diagnose_document(self, uri: str, content: str) -> list[Diagnostic]:
        event = self._events.setdefault(uri, asyncio.Event())
        event.clear()
        self._diagnostics[uri] = []
        version = self._versions.get(uri, 0) + 1
        self._versions[uri] = version
        if version == 1:
            await self._notify(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": uri,
                        "languageId": self.config.language_id,
                        "version": version,
                        "text": content,
                    }
                },
            )
        else:
            await self._notify(
                "textDocument/didChange",
                {
                    "textDocument": {"uri": uri, "version": version},
                    "contentChanges": [{"text": content}],
                },
            )
        try:
            await asyncio.wait_for(event.wait(), self.diagnostic_timeout)
        except TimeoutError:
            return []
        return self._diagnostics.get(uri, [])

    async def diagnose_with_code_actions(
        self,
        path: Path,
        content: str,
        *,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> tuple[list[Diagnostic], list[dict[str, Any]]]:
        current_task = self._enter_operation()
        try:
            await self.start()
            uri = await asyncio.to_thread(_path_uri, path)
            lock = self._document_locks.setdefault(uri, asyncio.Lock())
            async with lock:
                self._ensure_open()
                diagnostics = await self._diagnose_document(uri, content)
                last_line = max(1, content.count("\n") + 1)
                start = min(max(start_line, 1), last_line)
                end = min(max(end_line or last_line, start), last_line)
                try:
                    result = await asyncio.wait_for(
                        self._request(
                            "textDocument/codeAction",
                            {
                                "textDocument": {"uri": uri},
                                "range": {
                                    "start": {"line": start - 1, "character": 0},
                                    "end": {
                                        "line": end - 1,
                                        "character": 2_147_483_647,
                                    },
                                },
                                "context": {
                                    "diagnostics": [
                                        _diagnostic_context(item)
                                        for item in diagnostics[:MAX_LSP_CODE_ACTION_DIAGNOSTICS]
                                    ],
                                    "triggerKind": 1,
                                },
                            },
                        ),
                        self.diagnostic_timeout,
                    )
                except (RuntimeError, TimeoutError):
                    return diagnostics, []
                return diagnostics, _normalize_code_actions(result)
        finally:
            self._leave_operation(current_task)

    async def workspace_diagnostics(self) -> list[Diagnostic]:
        current_task = self._enter_operation()
        try:
            await self.start()
            try:
                result = await asyncio.wait_for(
                    self._request(
                        "workspace/diagnostic",
                        {"identifier": "kairocli", "previousResultIds": []},
                    ),
                    max(2.0, self.diagnostic_timeout),
                )
            except (RuntimeError, TimeoutError):
                return []
            if not isinstance(result, dict) or not isinstance(result.get("items"), list):
                return []
            diagnostics: list[Diagnostic] = []
            for report in result["items"][:MAX_LSP_WORKSPACE_REPORTS]:
                if len(diagnostics) >= MAX_LSP_WORKSPACE_DIAGNOSTICS:
                    break
                if not isinstance(report, dict):
                    continue
                uri = report.get("uri")
                if not isinstance(uri, str) or not _uri_is_within_workspace(uri, self.workspace):
                    continue
                raw_items = report.get("items")
                if not isinstance(raw_items, list):
                    continue
                display = _display_path(uri, self.workspace)
                remaining = MAX_LSP_WORKSPACE_DIAGNOSTICS - len(diagnostics)
                diagnostics.extend(
                    _from_lsp_diagnostic(display, raw)
                    for raw in raw_items[:remaining]
                    if isinstance(raw, dict)
                )
            return diagnostics
        finally:
            self._leave_operation(current_task)

    async def close(self) -> None:
        self._closing = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._close_once(), name="kairo-lsp-client-shutdown"
            )
        await _await_lsp_shutdown(self._close_task)

    async def _close_once(self) -> None:
        active = tuple(self._active_operations)
        for task in active:
            task.cancel()
        if active:
            await asyncio.wait(active, timeout=LSP_CANCEL_GRACE_SECONDS)
        async with self._start_lock:
            await self._close_process()

    async def _close_process(self) -> None:
        process = self.process
        if process is None:
            self._reset_document_state()
            return
        try:
            if process.returncode is None:
                await asyncio.wait_for(self._request("shutdown", None, allow_closing=True), 2)
                await self._notify("exit", None, allow_closing=True)
                await asyncio.wait_for(process.wait(), 2)
        except BaseException:
            try:
                await _terminate_lsp_process(process)
            except Exception:
                pass
        finally:
            tasks = tuple(
                task for task in (self._reader_task, self._stderr_task) if task is not None
            )
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self.process = None
            self._reader_task = None
            self._stderr_task = None
            self._reset_document_state()

    def _ensure_open(self) -> None:
        if self._closing:
            raise RuntimeError("LSP client is closed")

    def _enter_operation(self) -> asyncio.Task[Any] | None:
        self._ensure_open()
        task = asyncio.current_task()
        if task is not None:
            self._active_operations[task] = self._active_operations.get(task, 0) + 1
        return task

    def _leave_operation(self, task: asyncio.Task[Any] | None) -> None:
        if task is None:
            return
        depth = self._active_operations.get(task, 0)
        if depth <= 1:
            self._active_operations.pop(task, None)
        else:
            self._active_operations[task] = depth - 1

    async def _abort(self) -> None:
        process = self.process
        if process is not None:
            await _terminate_lsp_process(process)
        for task in (self._reader_task, self._stderr_task):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (self._reader_task, self._stderr_task) if task is not None),
            return_exceptions=True,
        )
        self.process = None
        self._reader_task = None
        self._stderr_task = None
        self._reset_document_state()

    def _reset_document_state(self) -> None:
        self._versions.clear()
        self._diagnostics.clear()
        self._events.clear()
        self._document_locks.clear()

    async def _request(self, method: str, params: Any, *, allow_closing: bool = False) -> Any:
        self._request_id += 1
        request_id = self._request_id
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[request_id] = future
        try:
            await self._send(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
                allow_closing=allow_closing,
            )
            return await future
        finally:
            self._pending.pop(request_id, None)

    async def _notify(self, method: str, params: Any, *, allow_closing: bool = False) -> None:
        await self._send(
            {"jsonrpc": "2.0", "method": method, "params": params},
            allow_closing=allow_closing,
        )

    async def _send(self, payload: dict[str, Any], *, allow_closing: bool = False) -> None:
        if self._closing and not allow_closing:
            raise RuntimeError("LSP client is closed")
        process = self.process
        if process is None or process.stdin is None or process.returncode is not None:
            raise RuntimeError("LSP server is not running")
        try:
            body = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        except (OverflowError, RecursionError, TypeError, ValueError) as exc:
            raise RuntimeError("LSP payload must be valid finite JSON") from exc
        if len(body) > MAX_LSP_MESSAGE_BYTES:
            raise RuntimeError("LSP payload exceeds the 10 MiB limit")
        framed = f"Content-Length: {len(body)}\r\n\r\n".encode() + body
        async with self._write_lock:
            process.stdin.write(framed)
            await process.stdin.drain()

    async def _reader_loop(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        try:
            while True:
                header = await process.stdout.readuntil(b"\r\n\r\n")
                if len(header) > MAX_LSP_HEADER_BYTES:
                    raise RuntimeError("LSP header exceeds the 16 KiB limit")
                length = _content_length(header)
                if length < 0 or length > MAX_LSP_MESSAGE_BYTES:
                    raise RuntimeError("Invalid LSP Content-Length")
                body = await process.stdout.readexactly(length)
                try:
                    payload = _decode_lsp_json(body)
                except (OverflowError, RecursionError, UnicodeError, ValueError):
                    continue
                if isinstance(payload, dict):
                    self._dispatch(payload)
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            for future in tuple(self._pending.values()):
                if not future.done():
                    future.set_exception(exc)
        finally:
            for future in tuple(self._pending.values()):
                if not future.done():
                    future.set_exception(RuntimeError("LSP server closed"))

    def _dispatch(self, payload: dict[str, Any]) -> None:
        if payload.get("jsonrpc") != "2.0":
            return
        request_id = payload.get("id")
        if (
            isinstance(request_id, int)
            and not isinstance(request_id, bool)
            and (future := self._pending.get(request_id))
        ):
            if future.done():
                return
            has_result = "result" in payload
            has_error = "error" in payload
            if has_result == has_error:
                future.set_exception(
                    RuntimeError("LSP response must contain exactly one of result or error")
                )
            elif has_error:
                error = payload["error"]
                if (
                    not isinstance(error, dict)
                    or type(error.get("code")) is not int
                    or not isinstance(error.get("message"), str)
                ):
                    future.set_exception(RuntimeError("LSP error response is invalid"))
                else:
                    future.set_exception(RuntimeError(safe_text(error)[:4_000]))
            else:
                future.set_result(payload["result"])
            return
        if payload.get("method") != "textDocument/publishDiagnostics":
            return
        params = payload.get("params")
        if not isinstance(params, dict):
            return
        uri = str(params.get("uri", ""))
        if uri not in self._versions or not _uri_is_within_workspace(uri, self.workspace):
            return
        published_version = params.get("version")
        current_version = self._versions.get(uri)
        if published_version is not None and (
            isinstance(published_version, bool) or not isinstance(published_version, int)
        ):
            return
        if (
            isinstance(published_version, int)
            and current_version is not None
            and published_version != current_version
        ):
            return
        raw_items = params.get("diagnostics")
        items = raw_items if isinstance(raw_items, list) else []
        display = _display_path(uri, self.workspace)
        self._diagnostics[uri] = [
            _from_lsp_diagnostic(display, raw)
            for raw in items[:MAX_LSP_DIAGNOSTICS_PER_FILE]
            if isinstance(raw, dict)
        ]
        self._events.setdefault(uri, asyncio.Event()).set()

    async def _drain_stderr(self) -> None:
        process = self.process
        if process is None or process.stderr is None:
            return
        while await process.stderr.read(65_536):
            pass


class LspManager:
    def __init__(
        self,
        workspace: Path,
        configs: list[LspServerConfig] | None = None,
        diagnostic_timeout: float = 2,
    ) -> None:
        self.workspace = workspace.resolve()
        self.diagnostics: dict[str, list[Diagnostic]] = {}
        self.enabled = os.getenv("KAIROCLI_LSP_ENABLED", "true").casefold() not in {
            "0",
            "false",
            "no",
            "off",
        }
        selected = configs if configs is not None else _discover_configs()
        self.clients = [
            LspClient(config, self.workspace, diagnostic_timeout) for config in selected
        ]

    def diagnose_file(self, file: Path) -> list[Diagnostic]:
        path, display = self._safe_file(file)
        if path is None or not self.enabled:
            return []
        content = _read_parser_source(path)
        if content is None:
            self.diagnostics.pop(str(path), None)
            return []
        result = _parser_diagnostics(display, path, content)
        self.diagnostics[str(path)] = result
        return result

    async def diagnose_file_async(self, file: Path) -> list[Diagnostic]:
        path, display = self._safe_file(file)
        if path is None or not self.enabled:
            return []
        content = await asyncio.to_thread(_read_parser_source, path)
        if content is None:
            self.diagnostics.pop(str(path), None)
            return []
        result = _parser_diagnostics(display, path, content)
        client = next(
            (
                candidate
                for candidate in self.clients
                if path.suffix.casefold() in candidate.config.extensions
            ),
            None,
        )
        if client is not None:
            try:
                lsp_items = await client.diagnose(path, content)
            except (OSError, RuntimeError, TimeoutError):
                lsp_items = []
            seen = {(item.line, item.column, item.message, item.source) for item in result}
            result.extend(
                item
                for item in lsp_items
                if (item.line, item.column, item.message, item.source) not in seen
            )
        self.diagnostics[str(path)] = result
        return result

    async def inspect_file_async(
        self,
        file: Path,
        *,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        path, display = self._safe_file(file)
        if path is None or not self.enabled:
            return {"path": display, "diagnostics": [], "code_actions": [], "server": None}
        content = await asyncio.to_thread(_read_parser_source, path)
        if content is None:
            self.diagnostics.pop(str(path), None)
            return {"path": display, "diagnostics": [], "code_actions": [], "server": None}
        result = _parser_diagnostics(display, path, content)
        actions: list[dict[str, Any]] = []
        client = next(
            (
                candidate
                for candidate in self.clients
                if path.suffix.casefold() in candidate.config.extensions
            ),
            None,
        )
        if client is not None:
            try:
                lsp_items, actions = await client.diagnose_with_code_actions(
                    path,
                    content,
                    start_line=start_line,
                    end_line=end_line,
                )
            except (OSError, RuntimeError, TimeoutError):
                lsp_items = []
            seen = {(item.line, item.column, item.message, item.source) for item in result}
            result.extend(
                item
                for item in lsp_items
                if (item.line, item.column, item.message, item.source) not in seen
            )
        self.diagnostics[str(path)] = result
        return {
            "path": display,
            "diagnostics": [
                {
                    "line": item.line,
                    "column": item.column,
                    "severity": item.severity,
                    "message": item.message,
                    "source": item.source,
                }
                for item in result
            ],
            "code_actions": actions,
            "server": client.config.command if client is not None else None,
        }

    async def workspace_diagnostics_async(
        self,
        root: Path,
        *,
        max_files: int = MAX_LSP_WORKSPACE_FILES,
        max_diagnostics: int = MAX_LSP_WORKSPACE_DIAGNOSTICS,
    ) -> dict[str, Any]:
        resolved = await asyncio.to_thread(root.resolve)
        try:
            prefix = resolved.relative_to(self.workspace).as_posix()
        except ValueError as exc:
            raise ValueError("LSP diagnostic root must stay within the workspace") from exc
        if not await asyncio.to_thread(resolved.is_dir):
            raise ValueError("LSP diagnostic root must be a directory")
        if prefix == ".":
            prefix = ""
        file_limit = min(max(max_files, 1), MAX_LSP_WORKSPACE_FILES)
        diagnostic_limit = min(max(max_diagnostics, 1), MAX_LSP_WORKSPACE_DIAGNOSTICS)
        parser_items, scanned_files, scan_truncated = await asyncio.to_thread(
            _scan_parser_diagnostics,
            resolved,
            self.workspace,
            file_limit,
            diagnostic_limit,
        )
        server_results = await asyncio.gather(
            *(client.workspace_diagnostics() for client in self.clients),
            return_exceptions=True,
        )
        combined = list(parser_items)
        for server_result in server_results:
            if isinstance(server_result, BaseException):
                continue
            combined.extend(
                item for item in server_result if _display_is_within_prefix(item.path, prefix)
            )
        rank = {"error": 0, "warning": 1, "information": 2, "hint": 3}
        unique: dict[tuple[str, int, int, str, str], Diagnostic] = {}
        for item in combined:
            key = (item.path, item.line, item.column, item.severity, item.message)
            unique.setdefault(key, item)
        ordered = sorted(
            unique.values(),
            key=lambda item: (
                rank.get(item.severity, 4),
                item.path,
                item.line,
                item.column,
            ),
        )
        truncated = scan_truncated or len(ordered) > diagnostic_limit
        selected = ordered[:diagnostic_limit]
        return {
            "root": prefix or ".",
            "scanned_files": scanned_files,
            "servers_queried": len(self.clients),
            "diagnostics": [
                {
                    "path": item.path,
                    "line": item.line,
                    "column": item.column,
                    "severity": item.severity,
                    "message": item.message,
                    "source": item.source,
                }
                for item in selected
            ],
            "partial": truncated,
            "diagnostics_before_limit": len(ordered),
        }

    def flush(self) -> list[Diagnostic]:
        rank = {"error": 0, "warning": 1, "information": 2, "hint": 3}
        result = [item for diagnostics in self.diagnostics.values() for item in diagnostics]
        self.diagnostics.clear()
        return sorted(
            result,
            key=lambda item: (
                rank.get(item.severity, 4),
                item.path,
                item.line,
                item.column,
            ),
        )

    def publish(self, uri: str, raw_diagnostics: list[dict[str, object]]) -> None:
        if not _uri_is_within_workspace(uri, self.workspace):
            return
        display = _display_path(uri, self.workspace)
        self.diagnostics[uri] = [
            _from_lsp_diagnostic(display, raw)
            for raw in raw_diagnostics[:MAX_LSP_DIAGNOSTICS_PER_FILE]
        ]

    async def close(self) -> None:
        clients = tuple(self.clients)
        self.clients.clear()
        await asyncio.gather(*(client.close() for client in clients), return_exceptions=True)

    def _safe_file(self, file: Path) -> tuple[Path | None, str]:
        path = file.resolve()
        try:
            display = path.relative_to(self.workspace).as_posix()
        except ValueError:
            return None, ""
        if not path.is_file():
            self.diagnostics.pop(str(path), None)
            return None, display
        return path, display


def _parser_diagnostics(display: str, path: Path, content: str) -> list[Diagnostic]:
    try:
        suffix = path.suffix.casefold()
        if suffix == ".py":
            ast.parse(content, filename=display)
        elif suffix == ".json":
            json.loads(content)
        elif suffix == ".toml":
            tomllib.loads(content)
        elif suffix in _PARSER_EXTENSIONS:
            issue = _delimiter_issue(content)
            if issue is not None:
                line, column, message = issue
                return [Diagnostic(display, line, column, "error", message, "parser")]
    except (SyntaxError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        line = int(getattr(exc, "lineno", 1) or 1)
        column = int(getattr(exc, "offset", getattr(exc, "colno", 0)) or 0)
        return [
            Diagnostic(
                display,
                line,
                column,
                "error",
                safe_text(exc).splitlines()[0],
                "parser",
            )
        ]
    return []


def _read_parser_source(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_LSP_PARSER_FILE_BYTES:
            return None
        with path.open("rb") as stream:
            raw = stream.read(MAX_LSP_PARSER_FILE_BYTES + 1)
    except OSError:
        return None
    if len(raw) > MAX_LSP_PARSER_FILE_BYTES or b"\0" in raw:
        return None
    return raw.decode("utf-8", errors="replace")


def _scan_parser_diagnostics(
    root: Path,
    workspace: Path,
    max_files: int,
    max_diagnostics: int,
) -> tuple[list[Diagnostic], int, bool]:
    diagnostics: list[Diagnostic] = []
    scanned = 0
    entries_seen = 0
    pending = [root]
    while pending:
        current_path = pending.pop()
        children: list[os.DirEntry[str]] = []
        try:
            with os.scandir(current_path) as iterator:
                for entry in iterator:
                    entries_seen += 1
                    if entries_seen > MAX_LSP_SCAN_ENTRIES:
                        return diagnostics, scanned, True
                    children.append(entry)
        except OSError:
            continue
        directories: list[Path] = []
        for entry in sorted(children, key=lambda item: item.name):
            path = Path(entry.path)
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if entry.name not in _LSP_SCAN_EXCLUDES:
                        directories.append(path)
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
            except OSError:
                continue
            if path.suffix.casefold() not in _PARSER_EXTENSIONS or path.is_symlink():
                continue
            if scanned >= max_files:
                return diagnostics, scanned, True
            try:
                resolved = path.resolve()
                display = resolved.relative_to(workspace).as_posix()
            except (OSError, ValueError):
                continue
            scanned += 1
            content = _read_parser_source(resolved)
            if content is None:
                continue
            diagnostics.extend(_parser_diagnostics(display, resolved, content))
            if len(diagnostics) >= max_diagnostics:
                return diagnostics[:max_diagnostics], scanned, True
        pending.extend(reversed(directories))
    return diagnostics, scanned, False


def _display_is_within_prefix(display: str, prefix: str) -> bool:
    return not prefix or display == prefix or display.startswith(prefix + "/")


def _delimiter_issue(content: str) -> tuple[int, int, str] | None:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[tuple[str, int, int]] = []
    for line_number, line in enumerate(content.splitlines(), 1):
        cleaned = re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', "", line)
        for column, character in enumerate(cleaned, 1):
            if character in "([{":
                stack.append((character, line_number, column))
            elif character in pairs:
                if not stack or stack[-1][0] != pairs[character]:
                    return line_number, column, f"Unmatched closing delimiter {character}"
                stack.pop()
    if stack:
        character, open_line, column = stack[-1]
        return open_line, column, f"Unclosed delimiter {character}"
    return None


def _from_lsp_diagnostic(display: str, raw: dict[str, object]) -> Diagnostic:
    raw_range = raw.get("range")
    range_data = raw_range if isinstance(raw_range, dict) else {}
    raw_start = range_data.get("start")
    start = raw_start if isinstance(raw_start, dict) else {}
    raw_severity = raw.get("severity", 2)
    severity_value = (
        raw_severity if isinstance(raw_severity, int) and not isinstance(raw_severity, bool) else 2
    )
    severity = {
        1: "error",
        2: "warning",
        3: "information",
        4: "hint",
    }.get(severity_value, str(severity_value).casefold())
    return Diagnostic(
        display,
        _nonnegative_int(start.get("line")) + 1,
        _nonnegative_int(start.get("character")) + 1,
        severity,
        str(raw.get("message", ""))[:MAX_LSP_DIAGNOSTIC_MESSAGE_CHARS],
        str(raw.get("source", "lsp"))[:200],
    )


def _diagnostic_context(item: Diagnostic) -> dict[str, Any]:
    line = max(0, item.line - 1)
    character = max(0, item.column - 1)
    severity = {"error": 1, "warning": 2, "information": 3, "hint": 4}.get(item.severity, 2)
    return {
        "range": {
            "start": {"line": line, "character": character},
            "end": {"line": line, "character": character},
        },
        "severity": severity,
        "message": item.message[:1_000],
        "source": item.source[:200],
    }


def _normalize_code_actions(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    actions: list[dict[str, Any]] = []
    for item in raw:
        if len(actions) >= MAX_LSP_CODE_ACTIONS:
            break
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        if not isinstance(title, str) or not title.strip():
            continue
        raw_command = item.get("command")
        command = ""
        if isinstance(raw_command, str):
            command = raw_command[:200]
        elif isinstance(raw_command, dict) and isinstance(raw_command.get("command"), str):
            command = str(raw_command["command"])[:200]
        disabled = item.get("disabled")
        disabled_reason = ""
        if isinstance(disabled, dict) and isinstance(disabled.get("reason"), str):
            disabled_reason = str(disabled["reason"])[:1_000]
        kind = item.get("kind")
        actions.append(
            {
                "title": title.strip()[:MAX_LSP_CODE_ACTION_TITLE_CHARS],
                "kind": str(kind)[:200] if isinstance(kind, str) else "",
                "preferred": item.get("isPreferred") is True,
                "disabled_reason": disabled_reason,
                "has_edit": isinstance(item.get("edit"), dict),
                "command": command,
            }
        )
    return actions


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, min(value, 2_147_483_647))


def _display_path(uri: str, workspace: Path) -> str:
    parsed = urlparse(uri)
    raw_path = unquote(parsed.path) if parsed.scheme == "file" else uri
    path = Path(raw_path).resolve()
    try:
        return path.relative_to(workspace).as_posix()
    except ValueError:
        return path.as_posix()


def _uri_is_within_workspace(uri: str, workspace: Path) -> bool:
    parsed = urlparse(uri)
    if parsed.scheme not in {"", "file"}:
        return False
    raw_path = unquote(parsed.path) if parsed.scheme == "file" else uri
    try:
        Path(raw_path).resolve().relative_to(workspace)
    except (OSError, ValueError):
        return False
    return True


def _path_uri(path: Path) -> str:
    return path.resolve().as_uri()


def _content_length(header: bytes) -> int:
    try:
        lines = header.decode("ascii").split("\r\n")
    except UnicodeDecodeError:
        return -1
    values: list[str] = []
    for line in lines:
        name, separator, value = line.partition(":")
        if separator and name.casefold() == "content-length":
            values.append(value.strip())
    if len(values) != 1 or not re.fullmatch(r"[0-9]{1,8}", values[0]):
        return -1
    return int(values[0])


def _decode_lsp_json(value: bytes) -> Any:
    payload = json.loads(
        value,
        object_pairs_hook=_lsp_object_without_duplicates,
        parse_constant=_reject_lsp_json_constant,
    )
    _validate_lsp_json_shape(payload)
    return payload


def _lsp_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate LSP JSON key: {key}")
        result[key] = value
    return result


def _reject_lsp_json_constant(value: str) -> Any:
    raise ValueError(f"Non-standard JSON constant is not allowed: {value}")


def _validate_lsp_json_shape(value: Any) -> None:
    stack: list[tuple[Any, int]] = [(value, 0)]
    visited = 0
    while stack:
        current, depth = stack.pop()
        visited += 1
        if visited > MAX_LSP_JSON_NODES:
            raise ValueError("LSP JSON exceeds the node limit")
        if depth > MAX_LSP_JSON_DEPTH:
            raise ValueError("LSP JSON exceeds the nesting limit")
        children: Iterable[Any]
        if isinstance(current, dict):
            children = current.values()
        elif isinstance(current, list):
            children = current
        else:
            continue
        child_count = len(current)
        if visited + len(stack) + child_count > MAX_LSP_JSON_NODES:
            raise ValueError("LSP JSON exceeds the node limit")
        stack.extend((child, depth + 1) for child in children)


async def _terminate_lsp_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), 1)
        return
    except TimeoutError:
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        return
    await process.wait()


async def _await_lsp_shutdown(task: asyncio.Task[None]) -> None:
    """Finish the shared shutdown even when one caller is cancelled."""
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    task.result()
    if cancelled:
        raise asyncio.CancelledError


def _discover_configs() -> list[LspServerConfig]:
    result: list[LspServerConfig] = []

    def configured(env_name: str, extensions: tuple[str, ...], language: str) -> bool:
        raw = os.getenv(env_name, "").strip()
        if not raw:
            return False
        if len(raw) > 8_192:
            raise ValueError(f"{env_name} exceeds 8192 characters")
        try:
            parts = shlex.split(raw)
        except ValueError as exc:
            raise ValueError(f"{env_name} is not valid shell-style syntax") from exc
        if not parts:
            return False
        if len(parts) > 64 or any(len(part) > 2_048 for part in parts):
            raise ValueError(f"{env_name} contains too many or oversized arguments")
        result.append(LspServerConfig(parts[0], tuple(parts[1:]), extensions, language))
        return True

    python_configured = configured("KAIROCLI_LSP_PYTHON_COMMAND", (".py",), "python")
    typescript_configured = configured(
        "KAIROCLI_LSP_TYPESCRIPT_COMMAND",
        (".js", ".jsx", ".ts", ".tsx"),
        "typescript",
    )
    go_configured = configured("KAIROCLI_LSP_GO_COMMAND", (".go",), "go")
    rust_configured = configured("KAIROCLI_LSP_RUST_COMMAND", (".rs",), "rust")
    java_configured = configured("KAIROCLI_LSP_JAVA_COMMAND", (".java",), "java")
    clang_configured = configured(
        "KAIROCLI_LSP_CLANG_COMMAND",
        (".c", ".h", ".cc", ".cpp", ".cxx", ".hh", ".hpp"),
        "cpp",
    )
    if os.getenv("KAIROCLI_LSP_AUTO", "true").casefold() in {"0", "false", "no", "off"}:
        return result
    if not python_configured:
        command = shutil.which("basedpyright-langserver") or shutil.which("pyright-langserver")
        if command:
            result.append(LspServerConfig(command, ("--stdio",), (".py",), "python"))
    if not typescript_configured:
        command = shutil.which("typescript-language-server")
        if command:
            result.append(
                LspServerConfig(
                    command,
                    ("--stdio",),
                    (".js", ".jsx", ".ts", ".tsx"),
                    "typescript",
                )
            )
    for is_configured, command_name, args, extensions, language in (
        (go_configured, "gopls", (), (".go",), "go"),
        (rust_configured, "rust-analyzer", (), (".rs",), "rust"),
        (java_configured, "jdtls", (), (".java",), "java"),
        (
            clang_configured,
            "clangd",
            (),
            (".c", ".h", ".cc", ".cpp", ".cxx", ".hh", ".hpp"),
            "cpp",
        ),
    ):
        if is_configured:
            continue
        command = shutil.which(command_name)
        if command:
            result.append(LspServerConfig(command, args, extensions, language))
    return result
