from __future__ import annotations

import asyncio
import html
import logging
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import replace

from ..paths import KairoPaths
from ..tools import ToolRegistry
from .client import McpClient, _escape_resource_metadata
from .config import (
    McpServerConfig,
    _load_mcp_state,
    _mcp_text_list,
    _update_mcp_state,
    load_mcp_config,
)
from .protocol import (
    format_tool_result as format_tool_result,
)
from .resources import parse_resource_mentions
from .safety import (
    bounded_mcp_error as _bounded_mcp_error,
)

log = logging.getLogger(__name__)


class McpServerManager:
    def __init__(self, paths: KairoPaths, registry: ToolRegistry) -> None:
        self.paths = paths
        self.registry = registry
        self.clients: dict[str, McpClient] = {}
        self.errors: dict[str, str] = {}
        self.configs: dict[str, McpServerConfig] = {}
        self._lifecycle_lock = asyncio.Lock()
        self._started = False

    async def start_all(self) -> None:
        log.info("mcp_start_all")
        async with self._lifecycle_lock:
            if self._started:
                return
            self.configs = load_mcp_config(self.paths)
            self.errors.clear()
            overrides = _load_mcp_state(self.paths)
            for name, enabled in overrides.items():
                if name in self.configs:
                    self.configs[name].enabled = enabled

            async def start_one(name: str, config: McpServerConfig) -> None:
                if config.error:
                    self.errors[name] = config.error
                    return
                if not config.enabled:
                    return
                client = McpClient(name, config)
                try:
                    await client.start()
                    client.register_tools(self.registry)
                    self.clients[name] = client
                except asyncio.CancelledError:
                    self._unregister_client(name, client)
                    await asyncio.gather(client.close(), return_exceptions=True)
                    self._unregister_client(name, client)
                    raise
                except Exception as exc:
                    self.errors[name] = _bounded_mcp_error(exc)
                    self._unregister_client(name, client)
                    await asyncio.gather(client.close(), return_exceptions=True)
                    self._unregister_client(name, client)

            try:
                await asyncio.gather(
                    *(start_one(name, config) for name, config in self.configs.items())
                )
            except BaseException:
                clients = list(self.clients.items())
                self.clients.clear()
                for name, client in clients:
                    self._unregister_client(name, client)
                await asyncio.gather(
                    *(client.close() for _, client in clients),
                    return_exceptions=True,
                )
                for name, client in clients:
                    self._unregister_client(name, client)
                raise
            self._started = True
            log.info(
                "mcp_start_complete configured=%d ready=%d errors=%d",
                len(self.configs),
                len(self.clients),
                len(self.errors),
            )

    async def restart(self, name: str) -> None:
        log.info("mcp_restart name=%s", name)
        async with self._lifecycle_lock:
            async with self.registry.mcp_server_transition(name):
                config = self.configs.get(name)
                if config is None:
                    raise KeyError(name)
                if config.error:
                    raise ValueError(f"Invalid MCP server configuration: {config.error}")
                candidate = replace(config, enabled=True, error=None)
                self._persist_enabled(name, True)
                await self._restart_locked(name, config, candidate)

    async def restart_with_args(
        self,
        name: str,
        args: list[str],
        on_success: Callable[[], None] | None = None,
    ) -> None:
        log.info("mcp_restart_with_args name=%s arg_count=%d", name, len(args))
        validated_args = _mcp_text_list(args, "args", 100, 16_384)
        async with self._lifecycle_lock:
            async with self.registry.mcp_server_transition(name):
                config = self.configs.get(name)
                if config is None:
                    raise KeyError(name)
                if config.error:
                    raise ValueError(f"Invalid MCP server configuration: {config.error}")
                candidate = replace(
                    config,
                    args=validated_args,
                    enabled=True,
                    error=None,
                )
                self._persist_enabled(name, True)
                await self._restart_locked(
                    name,
                    config,
                    candidate,
                    on_success=on_success,
                )

    async def _restart_locked(
        self,
        name: str,
        previous_config: McpServerConfig,
        candidate_config: McpServerConfig,
        *,
        on_success: Callable[[], None] | None = None,
    ) -> None:
        current = self.clients.pop(name, None)
        self._unregister_client(name, current)
        candidate: McpClient | None = None
        try:
            if current is not None:
                await current.close()
                self._unregister_client(name, current)
            candidate = McpClient(name, candidate_config)
            await candidate.start()
            candidate.register_tools(self.registry)
            if on_success is not None:
                on_success()
        except BaseException as exc:
            if candidate is not None:
                self._unregister_client(name, candidate)
                await asyncio.gather(candidate.close(), return_exceptions=True)
                self._unregister_client(name, candidate)
            rollback_error = ""
            rollback_succeeded = False
            if current is not None:
                restored = McpClient(name, previous_config)
                try:
                    await restored.start()
                    restored.register_tools(self.registry)
                    self.clients[name] = restored
                    rollback_succeeded = True
                except BaseException as rollback_exc:
                    self._unregister_client(name, restored)
                    await asyncio.gather(restored.close(), return_exceptions=True)
                    self._unregister_client(name, restored)
                    rollback_error = "; rollback failed: " + _bounded_mcp_error(rollback_exc)
            detail = _bounded_mcp_error(exc) + rollback_error
            if rollback_succeeded:
                self.errors.pop(name, None)
            else:
                self.errors[name] = detail
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise RuntimeError(detail) from exc
        assert candidate is not None
        self.configs[name] = candidate_config
        self.clients[name] = candidate
        self.errors.pop(name, None)

    async def disable(self, name: str) -> None:
        async with self._lifecycle_lock:
            async with self.registry.mcp_server_transition(name):
                config = self.configs.get(name)
                if config is None:
                    raise KeyError(name)
                self._persist_enabled(name, False)
                client = self.clients.pop(name, None)
                self._unregister_client(name, client)
                if client:
                    await client.close()
                    self._unregister_client(name, client)
                config.enabled = False
                self.errors.pop(name, None)

    async def enable(self, name: str) -> None:
        if name not in self.configs:
            raise KeyError(name)
        await self.restart(name)

    def _persist_enabled(self, name: str, enabled: bool) -> None:
        _update_mcp_state(self.paths, name, enabled)

    def _unregister_client(self, name: str, client: McpClient | None) -> None:
        if client:
            unregister = getattr(client, "unregister_tools", None)
            if callable(unregister):
                unregister(self.registry)
            else:
                for schema in client.tools:
                    self.registry.unregister(f"mcp__{name}__{schema.get('name', '')}")
        self.registry.unregister(f"mcp__{name}__list_resources")
        self.registry.unregister(f"mcp__{name}__read_resource")

    def status(self) -> dict[str, str]:
        return {
            name: "RUNNING"
            if name in self.clients
            else f"ERROR: {self.errors.get(name, 'disabled')}"
            for name in self.configs
        }

    async def close(self) -> None:
        log.info("mcp_close client_count=%d", len(self.clients))
        async with self._lifecycle_lock:
            clients = list(self.clients.items())
            async with AsyncExitStack() as transitions:
                for name, _ in sorted(clients):
                    await transitions.enter_async_context(self.registry.mcp_server_transition(name))
                self.clients.clear()
                self._started = False
                for name, client in clients:
                    self._unregister_client(name, client)
                await asyncio.gather(
                    *(client.close() for _, client in clients),
                    return_exceptions=True,
                )
                for name, client in clients:
                    self._unregister_client(name, client)

    async def expand_resource_mentions(
        self,
        value: str,
        max_chars: int = 200_000,
        *,
        max_mentions: int = 20,
    ) -> str:
        tokens = parse_resource_mentions(value)
        if not tokens or max_chars <= 0 or max_mentions <= 0:
            return value
        parts: list[str] = []
        cursor = 0
        remaining = max_chars
        expanded = 0
        seen: set[tuple[str, str]] = set()
        for token in tokens:
            parts.append(value[cursor : token.start])
            cursor = token.end
            if expanded >= max_mentions:
                parts.append(token.raw)
                continue
            escaped_server = html.escape(token.server, quote=True)
            escaped_uri = html.escape(token.uri, quote=True)
            key = (token.server, token.uri)
            if key in seen:
                replacement = (
                    f'<resource server="{escaped_server}" uri="{escaped_uri}" duplicate="true" />'
                )
                if len(replacement) <= remaining:
                    parts.append(replacement)
                    remaining -= len(replacement)
                else:
                    parts.append(token.raw)
                continue
            try:
                if token.server not in self.clients and token.server not in self.configs:
                    raise KeyError(token.server)
                async with self.registry.mcp_server_call(token.server):
                    client = self.clients[token.server]
                    content, mime_type = await client.read_resource(token.uri)
                escaped_mime = html.escape(mime_type, quote=True)
                opening = (
                    f'<resource server="{escaped_server}" uri="{escaped_uri}" '
                    f'mimeType="{escaped_mime}"'
                )
                closing = "\n</resource>"
                escaped_content = html.escape(content, quote=False)
                complete = opening + ">\n" + escaped_content + closing
                if len(complete) <= remaining:
                    replacement = complete
                else:
                    suffix = "\n[resource truncated by Kairo CLI context budget]"
                    prefix = opening + ' partial="true">\n'
                    available = remaining - len(prefix) - len(suffix) - len(closing)
                    if available <= 0:
                        replacement = (
                            token.raw + '\n<resource_error partial="true">'
                            "context budget exhausted</resource_error>"
                        )
                    else:
                        replacement = prefix + escaped_content[:available] + suffix + closing
            except Exception as exc:
                safe_error = html.escape(_bounded_mcp_error(exc), quote=False)
                replacement = (
                    token.raw + f'\n<resource_error server="{escaped_server}" '
                    f'uri="{escaped_uri}">'
                    f"{safe_error}</resource_error>"
                )
            expanded += 1
            seen.add(key)
            if len(replacement) <= remaining:
                parts.append(replacement)
                remaining -= len(replacement)
            else:
                parts.append(token.raw)
        parts.append(value[cursor:])
        return "".join(parts)

    def resource_mentions(self) -> list[tuple[str, str, str]]:
        mentions: list[tuple[str, str, str]] = []
        for server, client in sorted(self.clients.items()):
            for resource in client.resources:
                uri = resource.get("uri")
                if not isinstance(uri, str):
                    continue
                description = str(
                    resource.get("description") or resource.get("mimeType") or "MCP resource"
                )
                mentions.append((f"@{server}:{uri}", str(resource.get("name") or uri), description))
        return mentions

    def resource_index(self, max_chars: int = 20_000) -> str:
        if max_chars <= 0:
            return ""
        lines: list[str] = []
        used = 0
        for server, client in sorted(self.clients.items()):
            for resource in client.resources:
                uri = resource.get("uri")
                if not isinstance(uri, str):
                    continue
                name = str(resource.get("name") or uri)
                description = str(resource.get("description") or "")
                mime_type = str(resource.get("mimeType") or "")
                raw_line = (f"- {server}: {uri} | {name} | {mime_type} | {description}").rstrip()
                separator = 1 if lines else 0
                remaining = max_chars - used - separator
                if remaining <= 0:
                    return "\n".join(lines)
                escaped, truncated = _escape_resource_metadata(raw_line, remaining)
                if not escaped:
                    return "\n".join(lines)
                lines.append(escaped)
                used += separator + len(escaped)
                if truncated:
                    return "\n".join(lines)
        return "\n".join(lines)
