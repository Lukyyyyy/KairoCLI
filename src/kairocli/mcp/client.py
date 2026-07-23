from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import re
import signal
import subprocess
from collections import deque
from collections.abc import AsyncIterator
from functools import partial
from typing import Any

from ..models import ToolOutput
from ..text_safety import safe_text
from ..tools import ToolDefinition, ToolRegistry
from ..trace import redact_sensitive_text
from .config import McpServerConfig
from .constants import (
    _MCP_ACTIONABLE_NOTIFICATIONS,
    _MCP_PROTOCOL_VERSION,
    _MCP_RESOURCE_TOOL_NAMES,
    _MCP_STREAM_CHUNK_BYTES,
    _MCP_TOOL_NAME,
    MAX_MCP_LIST_BYTES,
    MAX_MCP_LIST_ITEMS,
    MAX_MCP_MESSAGE_BYTES,
    MAX_MCP_NOTIFICATION_TASKS,
    MAX_MCP_REQUEST_SECONDS,
    MAX_MCP_RESOURCE_CACHE_BYTES,
    MAX_MCP_RESOURCE_CACHE_ITEMS,
    MAX_MCP_RESOURCE_INFLIGHT,
    MAX_MCP_RESOURCE_URI_CHARS,
    MAX_MCP_SERVER_REQUEST_TASKS,
    MAX_MCP_SESSION_ID_CHARS,
    MAX_MCP_TOOL_DESCRIPTION_CHARS,
    MCP_PROTOCOL_VERSION,
)
from .protocol import (
    McpProtocolError,
    _decode_mcp_protocol_json,
    _serialize_mcp_message,
    format_tool_output,
    parse_sse_messages,
    sanitize_schema,
)
from .protocol import (
    format_tool_result as format_tool_result,
)
from .safety import (
    MAX_MCP_RESOURCE_ERROR_CHARS,
)
from .safety import (
    bounded_mcp_error as _bounded_mcp_error,
)

log = logging.getLogger(__name__)


class McpClient:
    def __init__(self, name: str, config: McpServerConfig) -> None:
        self.name = name
        self.config = config
        self.process: asyncio.subprocess.Process | None = None
        self.http_client: Any = None
        self.session_id: str | None = None
        self._next_id = 1
        self._write_lock = asyncio.Lock()
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._notification_tasks: set[asyncio.Task[None]] = set()
        self._server_request_tasks: set[asyncio.Task[None]] = set()
        self._notification_lock = asyncio.Lock()
        self._registry: ToolRegistry | None = None
        self._registered_tool_names: set[str] = set()
        self._resource_content_cache: dict[str, tuple[str, str]] = {}
        self._resource_cache_bytes = 0
        self._resource_reads: dict[str, asyncio.Task[tuple[str, str]]] = {}
        self._resource_epoch = 0
        self._resource_versions: dict[str, int] = {}
        self._resource_subscriptions: set[str] = set()
        self.tools: list[dict[str, Any]] = []
        self.resources: list[dict[str, Any]] = []
        self.prompts: list[dict[str, Any]] = []
        self.capabilities: dict[str, Any] = {}
        self.protocol_version: str | None = None
        self.stderr_log: deque[str] = deque(maxlen=200)
        self._stderr_task: asyncio.Task[None] | None = None

    async def start(self, initialize_timeout: float = 15) -> None:
        if self.config.command:
            env = os.environ | self.config.env
            process_options: dict[str, Any] = {}
            if os.name == "posix":
                process_options["start_new_session"] = True
            elif os.name == "nt":
                process_options["creationflags"] = getattr(
                    subprocess, "CREATE_NEW_PROCESS_GROUP", 0
                )
            self.process = await asyncio.create_subprocess_exec(
                self.config.command,
                *self.config.args,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=MAX_MCP_MESSAGE_BYTES + 1,
                **process_options,
            )
            self._stderr_task = asyncio.create_task(self._capture_stderr())
            self._reader_task = asyncio.create_task(self._read_stdout())
        elif self.config.url:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("Install Kairo CLI dependencies for HTTP MCP") from exc
            self.http_client = httpx.AsyncClient(timeout=initialize_timeout)
        else:
            raise ValueError("MCP server requires command or url")
        try:
            initialized = await asyncio.wait_for(
                self.request(
                    "initialize",
                    {
                        "protocolVersion": MCP_PROTOCOL_VERSION,
                        "capabilities": {},
                        "clientInfo": {"name": "Kairo CLI", "version": "0.1.0"},
                    },
                ),
                initialize_timeout,
            )
            negotiated_version = initialized.get("protocolVersion")
            if not isinstance(negotiated_version, str) or not _MCP_PROTOCOL_VERSION.fullmatch(
                negotiated_version
            ):
                raise McpProtocolError(
                    -32_602, "MCP initialize returned an invalid protocol version"
                )
            self.protocol_version = negotiated_version
            capabilities = initialized.get("capabilities") or {}
            self.capabilities = dict(capabilities) if isinstance(capabilities, dict) else {}
            await self.notify("notifications/initialized", {})
            self.tools = await self._list_paginated("tools/list", "tools")
            if "resources" in self.capabilities:
                self.resources = await self._list_paginated("resources/list", "resources")
            if "prompts" in self.capabilities:
                self.prompts = await self._list_paginated("prompts/list", "prompts")
        except BaseException:
            await self.close()
            raise

    async def request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        request_timeout: float = MAX_MCP_REQUEST_SECONDS,
    ) -> dict[str, Any]:
        if not 0 < request_timeout <= MAX_MCP_REQUEST_SECONDS:
            raise ValueError(
                f"MCP request timeout must be between 0 and {MAX_MCP_REQUEST_SECONDS:g} seconds"
            )
        request_id = self._take_id()
        if self.http_client is not None:
            return await asyncio.wait_for(
                self._http_request(method, params, request_id=request_id), request_timeout
            )
        if not self.process or not self.process.stdin or not self.process.stdout:
            raise RuntimeError("MCP server is not running")
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        message = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        try:
            await self._send_stdio(message)
            return await asyncio.wait_for(future, request_timeout)
        finally:
            self._pending.pop(request_id, None)

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        if self.http_client is not None:
            await self._http_request(method, params, request_id=None)
            return
        await self._send_stdio({"jsonrpc": "2.0", "method": method, "params": params})

    async def close(self) -> None:
        canceled = False
        self._invalidate_resource_cache()
        self._resource_epoch += 1
        self._resource_versions.clear()
        self._resource_subscriptions.clear()
        http_client = self.http_client
        session_id = self.session_id
        process = self.process
        reader_task = self._reader_task
        stderr_task = self._stderr_task
        notification_tasks = tuple(self._notification_tasks)
        server_request_tasks = tuple(self._server_request_tasks)
        resource_read_tasks = tuple(self._resource_reads.values())
        self.http_client = None
        self.session_id = None
        self.process = None
        self._reader_task = None
        self._stderr_task = None
        self._notification_tasks.clear()
        self._server_request_tasks.clear()
        self._resource_reads.clear()
        for task in (
            reader_task,
            stderr_task,
            *notification_tasks,
            *server_request_tasks,
            *resource_read_tasks,
        ):
            if task is not None and not task.done():
                task.cancel()
        pending = tuple(self._pending.values())
        self._pending.clear()
        for future in pending:
            if not future.done():
                future.set_exception(RuntimeError("MCP client closed"))
        if http_client is not None:
            if session_id and self.config.url:
                try:
                    await asyncio.wait_for(
                        http_client.delete(
                            self.config.url,
                            headers={
                                **self.config.headers,
                                "Mcp-Session-Id": session_id,
                                "MCP-Protocol-Version": self._http_protocol_version(),
                            },
                        ),
                        timeout=5,
                    )
                except asyncio.CancelledError:
                    canceled = True
                except Exception as exc:
                    self._log("HTTP MCP session cleanup failed: " + _bounded_mcp_error(exc))
            try:
                await http_client.aclose()
            except asyncio.CancelledError:
                canceled = True
            except Exception as exc:
                self._log("HTTP MCP client close failed: " + _bounded_mcp_error(exc))
        if process is not None:
            try:
                await _terminate_mcp_process(process)
            except asyncio.CancelledError:
                canceled = True
            except Exception as exc:
                self._log("MCP process cleanup failed: " + _bounded_mcp_error(exc))
        background = tuple(
            task
            for task in (
                reader_task,
                stderr_task,
                *notification_tasks,
                *server_request_tasks,
                *resource_read_tasks,
            )
            if task is not None
        )
        if background:
            await asyncio.gather(*background, return_exceptions=True)
        if canceled:
            raise asyncio.CancelledError

    def _take_id(self) -> int:
        request_id = self._next_id
        self._next_id += 1
        return request_id

    def _log(self, value: Any) -> None:
        normalized = safe_text(value).replace("\r", "\\r").replace("\n", "\\n")
        self.stderr_log.append(_bounded_mcp_error(normalized).replace("\n", " "))

    async def _http_request(
        self, method: str, params: dict[str, Any], request_id: int | None
    ) -> dict[str, Any]:
        if self.http_client is None or not self.config.url:
            raise RuntimeError("HTTP MCP server is not running")
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params}
        if request_id is not None:
            payload["id"] = request_id
        if len(_serialize_mcp_message(payload)) > MAX_MCP_MESSAGE_BYTES:
            raise McpProtocolError(-32_600, "MCP request exceeds the 2 MiB limit")
        headers = {
            **self.config.headers,
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": self._http_protocol_version(),
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        async with self.http_client.stream(
            "POST", self.config.url, json=payload, headers=headers
        ) as response:
            response.raise_for_status()
            self._capture_http_session(response.headers.get("Mcp-Session-Id"))
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > MAX_MCP_MESSAGE_BYTES:
                    raise McpProtocolError(-32_600, "MCP response exceeds the 2 MiB limit")
            content_type = response.headers.get("content-type", "")
        if not body:
            if request_id is not None:
                raise McpProtocolError(-32_603, "MCP request returned an empty response")
            return {}
        try:
            text_body = bytes(body).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise McpProtocolError(-32_700, "Invalid MCP JSON response") from exc
        if "text/event-stream" in content_type:
            messages = parse_sse_messages(text_body)
        else:
            try:
                raw = _decode_mcp_protocol_json(text_body)
            except (OverflowError, RecursionError, UnicodeError, ValueError) as exc:
                raise McpProtocolError(-32_700, "Invalid MCP JSON response") from exc
            messages = [raw]
        for message in messages:
            if isinstance(message, dict) and message.get("id") is None and message.get("method"):
                notification_params = message.get("params")
                self._schedule_notification(
                    str(message["method"]),
                    notification_params if isinstance(notification_params, dict) else {},
                )
        if request_id is None:
            return {}
        raw = next(
            (
                item
                for item in messages
                if isinstance(item, dict)
                and type(item.get("id")) is int
                and item.get("id") == request_id
            ),
            None,
        )
        if raw is None:
            raise McpProtocolError(-32_603, "MCP response ID does not match request")
        return self._decode_response(raw, request_id)

    def _decode_response(self, message: dict[str, Any], request_id: int) -> dict[str, Any]:
        if message.get("jsonrpc") != "2.0":
            raise McpProtocolError(-32_603, "MCP response has an invalid JSON-RPC version")
        if type(message.get("id")) is not int or message.get("id") != request_id:
            raise McpProtocolError(-32_603, "MCP response ID does not match request")
        has_error = "error" in message
        has_result = "result" in message
        if has_error == has_result:
            raise McpProtocolError(
                -32_603, "MCP response must contain exactly one of result or error"
            )
        if has_error:
            error = message["error"]
            if not isinstance(error, dict):
                raise McpProtocolError(-32_603, "MCP error response must be an object")
            code = error.get("code", -32_000)
            if type(code) is not int:
                code = -32_000
            raise McpProtocolError(
                code,
                str(error.get("message", error)),
                error.get("data"),
            )
        result = message["result"]
        if result is None:
            return {}
        if not isinstance(result, dict):
            raise McpProtocolError(-32_603, "MCP result must be a JSON object")
        return dict(result)

    def _http_protocol_version(self) -> str:
        return self.protocol_version or MCP_PROTOCOL_VERSION

    def _capture_http_session(self, value: Any) -> None:
        if value is None or value == "":
            return
        if (
            not isinstance(value, str)
            or len(value) > MAX_MCP_SESSION_ID_CHARS
            or any(not 0x21 <= ord(character) <= 0x7E for character in value)
        ):
            raise McpProtocolError(-32_603, "Invalid MCP session ID response header")
        if self.session_id is not None and value != self.session_id:
            raise McpProtocolError(-32_603, "MCP session ID changed unexpectedly")
        self.session_id = value

    def register_tools(self, registry: ToolRegistry) -> None:
        self._validate_tool_list(self.tools)
        self.unregister_tools(registry)
        self._registry = registry
        for raw in self.tools:
            raw_name = str(raw.get("name", ""))
            public_name = f"mcp__{self.name}__{raw_name}"

            async def handler(args: dict[str, Any], tool_name: str = raw_name) -> ToolOutput:
                result = await self.request("tools/call", {"name": tool_name, "arguments": args})
                return format_tool_output(result)

            registry.register(
                ToolDefinition(
                    public_name,
                    _bounded_tool_description(raw.get("description")),
                    sanitize_schema(raw.get("inputSchema")),
                    handler,
                )
            )
            self._registered_tool_names.add(public_name)
        if self._supports_resource_tools():

            async def list_resources(arguments: dict[str, Any]) -> dict[str, Any]:
                return {"resources": self.resources}

            async def read_resource(arguments: dict[str, Any]) -> str:
                uri = str(arguments["uri"])
                content, mime_type = await self.read_resource(uri)
                escaped_content = html.escape(content[:200_000], quote=False)
                opening = (
                    f'<resource uri="{html.escape(uri, quote=True)}" '
                    f'mimeType="{html.escape(mime_type, quote=True)}"'
                )
                closing = "\n</resource>"
                complete = opening + ">\n" + escaped_content + closing
                if len(content) <= 200_000 and len(complete) <= 200_000:
                    return complete
                prefix = opening + ' partial="true">\n'
                suffix = "\n[resource truncated by Kairo CLI context limit]"
                available = max(0, 200_000 - len(prefix) - len(suffix) - len(closing))
                return prefix + escaped_content[:available] + suffix + closing

            registry.register(
                ToolDefinition(
                    f"mcp__{self.name}__list_resources",
                    "List resources exposed by this MCP server.",
                    {"type": "object", "properties": {}},
                    list_resources,
                )
            )
            self._registered_tool_names.add(f"mcp__{self.name}__list_resources")
            registry.register(
                ToolDefinition(
                    f"mcp__{self.name}__read_resource",
                    "Read a resource exposed by this MCP server.",
                    {
                        "type": "object",
                        "properties": {
                            "uri": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": MAX_MCP_RESOURCE_URI_CHARS,
                            }
                        },
                        "required": ["uri"],
                    },
                    read_resource,
                )
            )
            self._registered_tool_names.add(f"mcp__{self.name}__read_resource")

    def unregister_tools(self, registry: ToolRegistry | None = None) -> None:
        target = registry or self._registry
        if target is not None:
            for name in tuple(self._registered_tool_names):
                target.unregister(name)
        self._registered_tool_names.clear()
        self._registry = None

    def _supports_resource_tools(self) -> bool:
        return "resources" in self.capabilities or bool(self.resources)

    def _validate_tool_list(self, tools: list[dict[str, Any]]) -> None:
        seen: set[str] = set()
        for raw in tools:
            raw_name = raw.get("name")
            if not isinstance(raw_name, str) or not _MCP_TOOL_NAME.fullmatch(raw_name):
                raise McpProtocolError(-32_602, "MCP server returned an invalid tool name")
            if raw_name in seen:
                raise McpProtocolError(
                    -32_602,
                    f"MCP server {self.name} returned duplicate tool name: {raw_name}",
                )
            if self._supports_resource_tools() and raw_name in _MCP_RESOURCE_TOOL_NAMES:
                raise McpProtocolError(
                    -32_602,
                    f"MCP tool conflicts with reserved resource helper: {raw_name}",
                )
            seen.add(raw_name)

    async def _capture_stderr(self) -> None:
        if not self.process or not self.process.stderr:
            return
        async for line, exceeded in _bounded_stream_lines(
            self.process.stderr, MAX_MCP_RESOURCE_ERROR_CHARS
        ):
            value = line.decode(errors="replace").rstrip()
            if exceeded:
                marker = "[MCP stderr line truncated]"
                available = max(0, MAX_MCP_RESOURCE_ERROR_CHARS - len(marker) - 1)
                value = f"{value[:available]} {marker}"
            self._log(value)

    async def _send_stdio(self, message: dict[str, Any]) -> None:
        encoded = _serialize_mcp_message(message) + b"\n"
        if len(encoded) > MAX_MCP_MESSAGE_BYTES:
            raise McpProtocolError(-32_600, "MCP request exceeds the 2 MiB limit")
        if not self.process or not self.process.stdin:
            raise RuntimeError("MCP server is not running")
        async with self._write_lock:
            self.process.stdin.write(encoded)
            await self.process.stdin.drain()

    async def _read_stdout(self) -> None:
        if not self.process or not self.process.stdout:
            return
        disconnect_error: BaseException = RuntimeError("MCP server disconnected")
        try:
            async for line, exceeded in _bounded_stream_lines(
                self.process.stdout, MAX_MCP_MESSAGE_BYTES
            ):
                if exceeded:
                    self._log("Invalid JSON-RPC message: exceeds the 2 MiB limit")
                    continue
                try:
                    message = _decode_mcp_protocol_json(line)
                except (
                    OverflowError,
                    RecursionError,
                    UnicodeError,
                    ValueError,
                ) as exc:
                    self._log("Invalid JSON-RPC message: " + _bounded_mcp_error(exc))
                    continue
                if not isinstance(message, dict):
                    self._log("Invalid JSON-RPC message: expected object")
                    continue
                if message.get("jsonrpc") != "2.0":
                    self._log("Invalid JSON-RPC message: expected version 2.0")
                    request_id = message.get("id")
                    if type(request_id) is int:
                        future = self._pending.get(request_id)
                        if future is not None and not future.done():
                            future.set_exception(
                                McpProtocolError(
                                    -32_603,
                                    "MCP response has an invalid JSON-RPC version",
                                )
                            )
                    continue
                request_id = message.get("id")
                method = message.get("method")
                if method is not None:
                    if not isinstance(method, str) or not method:
                        self._log("Invalid JSON-RPC message: invalid method")
                        continue
                    if request_id is None:
                        params = message.get("params")
                        self._schedule_notification(
                            method, params if isinstance(params, dict) else {}
                        )
                    else:
                        self._schedule_server_request(request_id, method)
                    continue
                if type(request_id) is not int:
                    self._log("Invalid JSON-RPC response ID")
                    continue
                future = self._pending.get(request_id)
                if future is None or future.done():
                    self._log(f"Unknown JSON-RPC response ID: {request_id}")
                    continue
                try:
                    future.set_result(self._decode_response(message, request_id))
                except McpProtocolError as exc:
                    future.set_exception(exc)
        except asyncio.CancelledError as exc:
            disconnect_error = exc
            raise
        except Exception as exc:
            disconnect_error = exc
            self._log("MCP stdout reader failed: " + _bounded_mcp_error(exc))
        finally:
            for future in tuple(self._pending.values()):
                if not future.done():
                    future.set_exception(disconnect_error)

    async def _respond_to_server_request(self, request_id: Any, method: str) -> None:
        if not (
            type(request_id) is int or (isinstance(request_id, str) and 0 < len(request_id) <= 200)
        ):
            self._log("Invalid JSON-RPC server request ID")
            return
        if method == "ping":
            response: dict[str, Any] = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {},
            }
        else:
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32_601,
                    "message": f"Client does not support server request: {method}",
                },
            }
        await self._send_stdio(response)

    def _schedule_server_request(self, request_id: Any, method: str) -> None:
        if len(self._server_request_tasks) >= MAX_MCP_SERVER_REQUEST_TASKS:
            self._log("MCP server request dropped: pending task limit reached")
            return
        task = asyncio.create_task(self._respond_to_server_request(request_id, method))
        self._server_request_tasks.add(task)
        task.add_done_callback(self._finish_server_request_task)

    def _finish_server_request_task(self, task: asyncio.Task[None]) -> None:
        self._server_request_tasks.discard(task)
        try:
            error = task.exception()
        except asyncio.CancelledError:
            return
        if error is not None:
            self._log("MCP server request response failed: " + _bounded_mcp_error(error))

    def _schedule_notification(self, method: str, params: dict[str, Any]) -> None:
        self._log(f"notification: {method}")
        if method not in _MCP_ACTIONABLE_NOTIFICATIONS:
            return
        if len(self._notification_tasks) >= MAX_MCP_NOTIFICATION_TASKS:
            if method.startswith("notifications/resources/"):
                self._mark_resource_changed()
            self._log("MCP notification dropped: pending task limit reached")
            return
        task = asyncio.create_task(self._handle_notification(method, params))
        self._notification_tasks.add(task)
        task.add_done_callback(self._finish_notification_task)

    def _finish_notification_task(self, task: asyncio.Task[None]) -> None:
        self._notification_tasks.discard(task)
        try:
            task.exception()
        except (asyncio.CancelledError, Exception):
            pass

    async def _handle_notification(self, method: str, params: dict[str, Any]) -> None:
        async with self._notification_lock:
            try:
                if method == "notifications/tools/list_changed":
                    refreshed = await self._list_paginated("tools/list", "tools")
                    self._validate_tool_list(refreshed)
                    self.tools = refreshed
                    if self._registry is not None:
                        self.register_tools(self._registry)
                elif method == "notifications/resources/list_changed":
                    self._mark_resource_changed()
                    self.resources = await self._list_paginated("resources/list", "resources")
                    if self._registry is not None:
                        self.register_tools(self._registry)
                elif method == "notifications/resources/updated":
                    uri = params.get("uri")
                    if isinstance(uri, str) and uri:
                        self._mark_resource_changed(uri)
                elif method == "notifications/prompts/list_changed":
                    self.prompts = await self._list_paginated("prompts/list", "prompts")
            except Exception as exc:
                self._log(f"notification refresh failed ({method}): " + _bounded_mcp_error(exc))

    async def read_resource(self, uri: str) -> tuple[str, str]:
        if (
            not isinstance(uri, str)
            or not uri
            or len(uri) > MAX_MCP_RESOURCE_URI_CHARS
            or any(ord(character) < 0x20 or ord(character) == 0x7F for character in uri)
        ):
            raise ValueError("MCP resource URI is invalid or exceeds 8,192 characters")
        cached = self._resource_content_cache.pop(uri, None)
        if cached is not None:
            self._resource_content_cache[uri] = cached
            return cached
        existing = self._resource_reads.get(uri)
        if existing is not None:
            if not existing.done():
                return await asyncio.shield(existing)
            self._resource_reads.pop(uri, None)
        if len(self._resource_reads) >= MAX_MCP_RESOURCE_INFLIGHT:
            raise RuntimeError(
                f"MCP resource reads exceed the {MAX_MCP_RESOURCE_INFLIGHT} in-flight limit"
            )
        task = asyncio.create_task(self._read_resource_consistent(uri))
        self._resource_reads[uri] = task
        task.add_done_callback(partial(self._finish_resource_read, uri))
        return await asyncio.shield(task)

    async def _read_resource_consistent(self, uri: str) -> tuple[str, str]:
        for attempt in range(2):
            epoch = self._resource_epoch
            version = self._resource_versions.get(uri, 0)
            value = await self._fetch_resource(uri)
            if epoch == self._resource_epoch and version == self._resource_versions.get(uri, 0):
                self._cache_resource(uri, value)
                return value
            if attempt == 1:
                break
        raise RuntimeError("MCP resource changed repeatedly while it was being read")

    async def _fetch_resource(self, uri: str) -> tuple[str, str]:
        result = await self.request("resources/read", {"uri": uri})
        contents = result.get("contents", [])
        text_parts: list[str] = []
        mime_type = "text/plain"
        if isinstance(contents, list):
            for item in contents:
                if not isinstance(item, dict):
                    continue
                item_mime = item.get("mimeType")
                if isinstance(item_mime, str) and item_mime:
                    mime_type = item_mime
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
                elif item.get("blob"):
                    text_parts.append("[Binary MCP resource content omitted]")
        value = ("\n\n".join(text_parts), mime_type)
        await self._subscribe_resource(uri)
        return value

    def _finish_resource_read(
        self,
        uri: str,
        task: asyncio.Task[tuple[str, str]],
    ) -> None:
        if self._resource_reads.get(uri) is task:
            self._resource_reads.pop(uri, None)
            self._resource_versions.pop(uri, None)
        try:
            task.exception()
        except (asyncio.CancelledError, Exception):
            pass

    async def _subscribe_resource(self, uri: str) -> None:
        capability = self.capabilities.get("resources")
        if (
            not isinstance(capability, dict)
            or capability.get("subscribe") is not True
            or uri in self._resource_subscriptions
        ):
            return
        try:
            await self.request("resources/subscribe", {"uri": uri})
        except Exception as exc:
            self._log(f"resource subscription failed ({uri}): {_bounded_mcp_error(exc)}")
            return
        self._resource_subscriptions.add(uri)

    def _cache_resource(self, uri: str, value: tuple[str, str]) -> None:
        self._invalidate_resource_cache(uri)
        size = _utf8_size(uri) + _utf8_size(value[0]) + _utf8_size(value[1])
        if size > MAX_MCP_RESOURCE_CACHE_BYTES:
            return
        while self._resource_content_cache and (
            len(self._resource_content_cache) >= MAX_MCP_RESOURCE_CACHE_ITEMS
            or self._resource_cache_bytes + size > MAX_MCP_RESOURCE_CACHE_BYTES
        ):
            oldest = next(iter(self._resource_content_cache))
            self._invalidate_resource_cache(oldest)
        self._resource_content_cache[uri] = value
        self._resource_cache_bytes += size

    def _invalidate_resource_cache(self, uri: str | None = None) -> None:
        if uri is None:
            self._resource_content_cache.clear()
            self._resource_cache_bytes = 0
            return
        previous = self._resource_content_cache.pop(uri, None)
        if previous is not None:
            self._resource_cache_bytes = max(
                0,
                self._resource_cache_bytes
                - _utf8_size(uri)
                - _utf8_size(previous[0])
                - _utf8_size(previous[1]),
            )

    def _mark_resource_changed(self, uri: str | None = None) -> None:
        if uri is None:
            self._resource_epoch += 1
            self._resource_versions.clear()
            self._invalidate_resource_cache()
            return
        if uri in self._resource_reads:
            self._resource_versions[uri] = self._resource_versions.get(uri, 0) + 1
        else:
            self._resource_versions.pop(uri, None)
        self._invalidate_resource_cache(uri)

    async def _list_paginated(self, method: str, key: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        aggregate_bytes = 0
        cursor: str | None = None
        seen: set[str] = set()
        for _ in range(100):
            params = {"cursor": cursor} if cursor else {}
            page = await self.request(method, params)
            raw_items = page.get(key, [])
            if isinstance(raw_items, list):
                for item in raw_items:
                    if not isinstance(item, dict):
                        continue
                    aggregate_bytes += len(
                        json.dumps(
                            item,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    )
                    if len(items) >= MAX_MCP_LIST_ITEMS:
                        raise McpProtocolError(
                            -32_600,
                            f"MCP {key} list exceeds {MAX_MCP_LIST_ITEMS} items",
                        )
                    if aggregate_bytes > MAX_MCP_LIST_BYTES:
                        raise McpProtocolError(
                            -32_600,
                            f"MCP {key} list exceeds the aggregate byte limit",
                        )
                    items.append(item)
            next_cursor = page.get("nextCursor")
            if not isinstance(next_cursor, str) or not next_cursor or next_cursor in seen:
                break
            seen.add(next_cursor)
            cursor = next_cursor
        return items


async def _bounded_stream_lines(
    stream: asyncio.StreamReader, maximum: int
) -> AsyncIterator[tuple[bytes, bool]]:
    retained = bytearray()
    exceeded = False
    while chunk := await stream.read(_MCP_STREAM_CHUNK_BYTES):
        start = 0
        while start < len(chunk):
            newline = chunk.find(b"\n", start)
            end = len(chunk) if newline < 0 else newline
            piece = chunk[start:end]
            remaining = max(0, maximum - len(retained))
            if remaining:
                retained.extend(piece[:remaining])
            if len(piece) > remaining:
                exceeded = True
            if newline < 0:
                break
            yield bytes(retained), exceeded
            retained.clear()
            exceeded = False
            start = newline + 1
    if retained or exceeded:
        yield bytes(retained), exceeded


def _bounded_tool_description(value: Any) -> str:
    description = value if isinstance(value, str) and value.strip() else "MCP tool"
    if len(description) > MAX_MCP_TOOL_DESCRIPTION_CHARS:
        return description[:MAX_MCP_TOOL_DESCRIPTION_CHARS] + "..."
    return description


def _utf8_size(value: str) -> int:
    return len(value.encode("utf-8", errors="replace"))


def _escape_resource_metadata(value: str, max_chars: int) -> tuple[str, bool]:
    normalized = re.sub(r"[\r\n\t]+", " ", redact_sensitive_text(value))
    pieces: list[str] = []
    used = 0
    for character in normalized:
        codepoint = ord(character)
        if not (
            codepoint == 0x20
            or 0x21 <= codepoint <= 0xD7FF
            or 0xE000 <= codepoint <= 0xFFFD
            or 0x10000 <= codepoint <= 0x10FFFF
        ):
            character = "�"
        escaped = html.escape(character, quote=False)
        if used + len(escaped) > max_chars:
            return "".join(pieces), True
        pieces.append(escaped)
        used += len(escaped)
    return "".join(pieces), False


async def _terminate_mcp_process(process: asyncio.subprocess.Process) -> None:
    if os.name == "nt":
        try:
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/PID",
                str(process.pid),
                "/T",
                "/F",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(killer.wait(), 3)
        except (OSError, TimeoutError):
            if process.returncode is None:
                process.kill()
    elif os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        for _ in range(10):
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return
            await asyncio.sleep(0.05)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        if process.returncode is None:
            await process.wait()
        return
    if process.returncode is not None:
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
