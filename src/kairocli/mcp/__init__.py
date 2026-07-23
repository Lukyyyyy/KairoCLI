"""Model Context Protocol support for Kairo CLI."""

from .client import McpClient
from .commands import handle_mcp_command, refresh_agent_resource_index
from .config import (
    McpConfigBootstrapResult,
    McpServerConfig,
    ensure_default_mcp_config,
    load_mcp_config,
)
from .manager import McpServerManager
from .protocol import (
    McpProtocolError,
    format_tool_output,
    format_tool_result,
    parse_sse_messages,
    sanitize_schema,
)
from .resources import ResourceMention, parse_resource_mentions

__all__ = [
    "McpClient",
    "McpConfigBootstrapResult",
    "McpProtocolError",
    "McpServerConfig",
    "McpServerManager",
    "ResourceMention",
    "ensure_default_mcp_config",
    "format_tool_output",
    "format_tool_result",
    "handle_mcp_command",
    "load_mcp_config",
    "parse_resource_mentions",
    "parse_sse_messages",
    "refresh_agent_resource_index",
    "sanitize_schema",
]
