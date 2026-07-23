from __future__ import annotations

import re
import threading

MAX_MCP_CONFIG_BYTES = 1024 * 1024
MAX_MCP_STATE_BYTES = 128 * 1024
MAX_MCP_SERVERS = 100
MAX_MCP_JSON_DEPTH = 16
MAX_MCP_JSON_NODES = 10_000
_MCP_STATE_THREAD_LOCK = threading.Lock()
MAX_MCP_MESSAGE_BYTES = 2 * 1024 * 1024
MAX_MCP_LIST_ITEMS = 1_000
MAX_MCP_LIST_BYTES = 4 * 1024 * 1024
MAX_MCP_TOOL_DESCRIPTION_CHARS = 1_000
MAX_MCP_RESOURCE_CACHE_ITEMS = 128
MAX_MCP_RESOURCE_CACHE_BYTES = 16 * 1024 * 1024
MAX_MCP_RESOURCE_INFLIGHT = 128
MAX_MCP_RESOURCE_URI_CHARS = 8_192
MAX_MCP_REQUEST_SECONDS = 60.0
MCP_PROTOCOL_VERSION = "2025-06-18"
MAX_MCP_SESSION_ID_CHARS = 1_024
MAX_MCP_NOTIFICATION_TASKS = 100
MAX_MCP_SERVER_REQUEST_TASKS = 100
_MCP_STREAM_CHUNK_BYTES = 64 * 1024
_MCP_SERVER_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}$")
_MCP_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")
_MCP_HEADER_NAME = re.compile(r"[A-Za-z0-9!#$%&'*+.^_`|~-]{1,128}$")
_MCP_TOOL_NAME = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_MCP_PROTOCOL_VERSION = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")
_MCP_RESOURCE_TOOL_NAMES = {"list_resources", "read_resource"}
_MCP_ACTIONABLE_NOTIFICATIONS = {
    "notifications/tools/list_changed",
    "notifications/resources/list_changed",
    "notifications/resources/updated",
    "notifications/prompts/list_changed",
}
_DEFAULT_CHROME_DEVTOOLS_MCP = {
    "mcpServers": {
        "chrome-devtools": {
            "command": "npx",
            "args": ["-y", "chrome-devtools-mcp@latest", "--isolated=true"],
        }
    }
}
