from __future__ import annotations

import json
import logging
import re
from typing import Any

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
) -> str:
    operation, _, raw_name = payload.strip().partition(" ")
    name = raw_name.strip()
    if not operation or operation == "list":
        statuses = manager.status()
        if not statuses:
            return "No MCP servers configured."
        return "\n".join(f"{server}: {statuses[server]}" for server in sorted(statuses))
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
