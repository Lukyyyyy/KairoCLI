from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CommandType(StrEnum):
    NONE = "none"
    UNKNOWN = "unknown"
    HELP = "help"
    EXIT = "exit"
    CANCEL = "cancel"
    CLEAR = "clear"
    COMPACT = "compact"
    HISTORY = "history"
    HISTORY_CLEAR = "history_clear"
    INIT = "init"
    MODEL = "model"
    PLAN = "plan"
    TEAM = "team"
    HITL = "hitl"
    MEMORY = "memory"
    SAVE = "save"
    INDEX = "index"
    SEARCH = "search"
    GRAPH = "graph"
    CONTEXT = "context"
    POLICY = "policy"
    AUDIT = "audit"
    SNAPSHOT = "snapshot"
    RESTORE = "restore"
    BROWSER = "browser"
    WECHAT = "wechat"
    TASK = "task"
    SKILL = "skill"
    CONFIG = "config"
    SESSION = "session"
    TODO = "todo"
    SHELL = "shell"
    TRACE = "trace"
    EXPORT = "export"
    MCP = "mcp"


@dataclass(frozen=True, slots=True)
class ParsedCommand:
    type: CommandType
    payload: str | None = None


ALIASES = {"quit": "exit", "mem": "memory", "ctx": "context"}
KNOWN = {
    item.value: item for item in CommandType if item not in {CommandType.NONE, CommandType.UNKNOWN}
}

SLASH_COMMAND_DESCRIPTIONS: dict[str, str] = {
    "/help": "show the command reference",
    "/exit": "quit Kairo CLI",
    "/cancel": "cancel the active turn",
    "/clear": "clear the current conversation",
    "/compact": "compact the conversation context",
    "/history": "manage saved input history",
    "/init": "create or refresh project KAIRO.md",
    "/model": "inspect or select the active model",
    "/plan": "run a task in Plan-and-Execute mode",
    "/team": "run a task with a team of agents",
    "/hitl": "inspect or configure approval behavior",
    "/memory": "search and manage durable memory",
    "/save": "save a fact to durable memory",
    "/index": "index the workspace for code search",
    "/search": "search indexed workspace code",
    "/graph": "explore the workspace code graph",
    "/context": "inspect conversation context usage",
    "/policy": "inspect the active safety policy",
    "/audit": "inspect recent audit events",
    "/snapshot": "inspect side-history snapshots",
    "/restore": "restore a side-history snapshot",
    "/browser": "manage the browser session",
    "/wechat": "manage WeChat · setup/start/status/stop",
    "/task": "manage durable background tasks",
    "/skill": "install and manage available skills",
    "/config": "inspect or update provider configuration",
    "/session": "manage saved conversations",
    "/todo": "manage the current todo list",
    "/shell": "manage persistent shell sessions",
    "/trace": "inspect model diagnostics and tracing",
    "/export": "export the current session",
    "/mcp": "manage MCP servers and resources",
}

SLASH_SUBCOMMAND_DESCRIPTIONS: dict[str, str] = {
    "/browser connect": "connect to a Chrome browser",
    "/browser disconnect": "disconnect the shared browser",
    "/browser status": "show browser connection status",
    "/browser tabs": "list visible browser tabs",
    "/config provider": "select or configure a provider",
    "/history clear": "clear saved input history",
    "/hitl on": "enable approval prompts",
    "/hitl off": "disable approval prompts",
    "/mcp list": "list configured MCP servers",
    "/mcp disable": "disable an MCP server",
    "/mcp enable": "enable an MCP server",
    "/mcp logs": "show MCP server logs",
    "/mcp prompts": "list MCP prompts",
    "/mcp resources": "list MCP resources",
    "/mcp restart": "restart an MCP server",
    "/memory clear": "clear durable memory",
    "/memory delete": "delete a memory entry",
    "/memory list": "list durable memories",
    "/memory search": "search durable memory",
    "/session delete": "delete sessions by ID or remove all empty sessions",
    "/session list": "list saved conversations; add --all to include empty sessions",
    "/session new": "start a new conversation",
    "/session resume": "resume a saved conversation",
    "/session save": "save the current conversation",
    "/session status": "show current conversation status",
    "/shell exec": "run a command in a shell session",
    "/shell list": "list persistent shell sessions",
    "/shell start": "start a persistent shell session",
    "/shell stop": "stop a persistent shell session",
    "/skill install": "install a Skill",
    "/skill list": "list available Skills",
    "/skill off": "disable a Skill",
    "/skill on": "enable a Skill",
    "/skill reload": "reload discovered Skills",
    "/skill show": "show Skill details",
    "/snapshot clean": "clear snapshot history",
    "/snapshot list": "list side-history snapshots",
    "/snapshot status": "show snapshot service status",
    "/task add": "enqueue a background task",
    "/task cancel": "cancel a background task",
    "/task list": "list background tasks",
    "/task log": "show background task details",
    "/todo add": "add a todo",
    "/todo clear": "clear todos",
    "/todo clear completed": "clear completed todos",
    "/todo clear all": "clear all todos",
    "/todo done": "mark a todo completed",
    "/todo list": "list current todos",
    "/todo remove": "remove a todo",
    "/todo reopen": "reopen a todo",
    "/todo start": "mark a todo in progress",
    "/trace off": "disable private model tracing",
    "/trace on": "enable private model tracing",
    "/trace reasoning": "configure private reasoning traces",
    "/trace reasoning off": "exclude reasoning from private traces",
    "/trace reasoning on": "include reasoning in private traces",
    "/trace status": "show private trace status",
    "/wechat setup": "connect a WeChat account",
    "/wechat start": "start receiving WeChat messages",
    "/wechat status": "show connection and channel status",
    "/wechat stop": "stop receiving WeChat messages",
}


def parse_command(value: str | None) -> ParsedCommand:
    if value is None or not value.strip():
        return ParsedCommand(CommandType.NONE)
    stripped = value.strip()
    plain_command = stripped.lower() in {
        "exit",
        "quit",
        "cancel",
        "clear",
        "help",
        "?",
    }
    if not stripped.startswith("/") and not plain_command:
        return ParsedCommand(CommandType.NONE)
    content = stripped[1:] if stripped.startswith("/") else stripped
    if content == "?":
        content = "help"
    if content.lower() == "history clear":
        return ParsedCommand(CommandType.HISTORY_CLEAR)
    head, _, payload = content.partition(" ")
    head = ALIASES.get(head.lower(), head.lower())
    command = KNOWN.get(head)
    if command is None:
        return ParsedCommand(CommandType.UNKNOWN, stripped)
    return ParsedCommand(command, payload.strip() or None)


SLASH_HELP = """Commands:
  /help                         show this command reference
  /plan TASK | /team TASK      run Plan-and-Execute or team mode
  /model | /config             inspect or update provider configuration
  /clear | /compact | /context manage conversation context
  /memory | /save FACT         search or save durable memory
  /index | /search | /graph    index and explore workspace code
  /skill install SOURCE        install a local, curated, or Git-hosted Skill
  /mcp | /skill | /browser     manage integrations, Skills, and browser sessions
  /policy | /hitl | /audit     inspect safety, approvals and audit events
  /snapshot | /restore N       inspect or restore side-history snapshots
  /task | /todo | /session     manage durable work, todos and conversations
  /shell | /trace              manage persistent shells and model diagnostics
  /init | /history clear       initialize project memory or clear input history
  /export | /wechat            export this session or inspect WeChat binding
  /cancel | /exit              cancel the active turn or quit Kairo CLI
"""
