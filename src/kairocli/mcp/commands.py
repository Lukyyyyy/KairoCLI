from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from rich.text import Text

from ..rendering.terminal import sanitize_terminal_text
from .constants import (
    _MCP_SERVER_NAME,
)
from .manager import McpServerManager
from .protocol import (
    format_tool_result as format_tool_result,
)
from .safety import (
    bounded_mcp_error as _bounded_mcp_error,
)

log = logging.getLogger(__name__)

_MCP_RESOURCE_INDEX = re.compile(r"\n*<mcp_resource_index>.*?</mcp_resource_index>", re.DOTALL)


def _format_uptime(started_at: float) -> str:
    seconds = max(0, int(time.monotonic() - started_at))
    if seconds >= 86_400:
        return f"{seconds // 86_400}d"
    if seconds >= 3_600:
        return f"{seconds // 3_600}h"
    if seconds >= 60:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def _format_mcp_servers(manager: McpServerManager, statuses: dict[str, str]) -> Text:
    names = sorted(manager.configs)
    name_width = max(map(len, names))
    tool_width = max(
        len(str(len(getattr(manager.clients.get(name), "tools", ())))) for name in names
    )
    output = Text("🔌 MCP Servers", style="bold")
    for name in names:
        client = manager.clients.get(name)
        status = statuses[name]
        state, style = (
            ("ready", "green")
            if client is not None
            else ("disabled", "yellow")
            if not manager.configs[name].enabled
            else ("error", "red")
        )
        transport = "http" if manager.configs[name].url else "stdio"
        tools = len(getattr(client, "tools", ()))
        output.append(f"\n  {name:<{name_width}}  ")
        output.append(f"● {state:<8}", style=style)
        output.append(f"  {transport:<5}  {tools:>{tool_width}} tools")
        started_at = getattr(client, "started_at", None)
        if isinstance(started_at, (int, float)):
            output.append(f"  uptime {_format_uptime(started_at)}")
        pid = getattr(getattr(client, "process", None), "pid", None)
        if isinstance(pid, int):
            output.append(f"  pid {pid}")
        if state == "error":
            detail = sanitize_terminal_text(status.removeprefix("ERROR: ")).replace("\n", " ")
            output.append("  " + detail, style="dim red")
    return output


def refresh_agent_resource_index(agent: Any, manager: McpServerManager) -> None:
    base = _MCP_RESOURCE_INDEX.sub("", str(agent.base_system_prompt)).rstrip()
    if agent.context_profile.mcp_resource_index_enabled:
        index = manager.resource_index()
        if index:
            base += (
                "\n\n<mcp_resource_index>\n"
                "Available MCP resource metadata follows. Read content only when relevant.\n"
                + index
                + "\n</mcp_resource_index>"
            )
    agent.base_system_prompt = base
    agent.system_prompt = base


async def handle_mcp_command(
    payload: str,
    manager: McpServerManager,
    agent: Any | None = None,
    approval_policy: Any | None = None,
) -> str | Text:
    operation, _, raw_name = payload.strip().partition(" ")
    name = raw_name.strip()
    if not operation or operation == "list":
        statuses = manager.status()
        if not statuses:
            return "No MCP servers configured."
        return _format_mcp_servers(manager, statuses)
    if operation not in {"restart", "logs", "disable", "enable", "resources", "prompts"}:
        return "Usage: /mcp [list|restart|logs|disable|enable|resources|prompts] <name>"
    if not name:
        return f"Usage: /mcp {operation} <name>"
    if not _MCP_SERVER_NAME.fullmatch(name):
        return "MCP error: invalid server name"
    try:
        if operation == "restart":
            await manager.restart(name)
            message = f"MCP server restarted: {name}"
        elif operation == "disable":
            await manager.disable(name)
            message = f"MCP server disabled: {name}"
        elif operation == "enable":
            await manager.enable(name)
            message = f"MCP server enabled: {name}"
        else:
            async with manager.registry.mcp_server_call(name):
                client = manager.clients.get(name)
                if operation == "logs":
                    return "\n".join(client.stderr_log) if client else "MCP server is not running."
                values = (
                    client.resources
                    if client is not None and operation == "resources"
                    else client.prompts
                    if client is not None
                    else []
                )
                return json.dumps(values, ensure_ascii=False, indent=2)
        if approval_policy is not None and operation in {
            "restart",
            "disable",
            "enable",
        }:
            approval_policy.clear_mcp_server_approvals(name)
        if agent is not None:
            refresh_agent_resource_index(agent, manager)
        return message
    except (KeyError, RuntimeError, ValueError, OSError) as exc:
        return "MCP error: " + _bounded_mcp_error(exc)
