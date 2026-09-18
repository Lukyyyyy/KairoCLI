import json
from collections import OrderedDict

from ..json_boundary import decode_strict_json
from ..models import ToolCall, ToolOutput
from ..tools.tool_result import is_failed_tool_text
from ..trace import redact_sensitive_text
from .terminal import sanitize_terminal_text

_LABELS = {
    "read_file": "📖 Read file",
    "write_file": "✏️ Write file",
    "apply_patch": "📝 Apply patch",
    "list_dir": "📂 List directory",
    "glob_files": "🔎 Find files",
    "grep_code": "🔍 Search text",
    "search_code": "🔍 Search code",
    "query_code_graph": "🕸 Query code graph",
    "execute_command": "⚡ Shell",
    "shell_exec": "⚡ Persistent shell",
    "web_search": "🌐 Web search",
    "web_fetch": "📰 Web fetch",
    "save_memory": "💾 Save memory",
    "search_memory": "🔍 Search memory",
    "lsp_inspect": "🩺 Inspect diagnostics",
    "lsp_workspace_diagnostics": "🩺 Workspace diagnostics",
}
_KEYS = {
    "read_file": "path",
    "write_file": "path",
    "list_dir": "path",
    "glob_files": "pattern",
    "grep_code": "query",
    "search_code": "query",
    "query_code_graph": "symbol",
    "execute_command": "command",
    "shell_exec": "command",
    "web_search": "query",
    "web_fetch": "url",
    "save_memory": "fact",
    "search_memory": "query",
    "lsp_inspect": "path",
    "lsp_workspace_diagnostics": "path",
}


def format_tool_calls(calls: list[ToolCall], *, compact: bool = True) -> str:
    if not calls:
        return ""
    grouped: OrderedDict[str, list[ToolCall]] = OrderedDict()
    for call in calls[:100]:
        grouped.setdefault(call.name, []).append(call)
    if compact and len(grouped) > 1:
        return f"⏵ {len(grouped)} tool groups / {sum(map(len, grouped.values()))} calls"
    lines: list[str] = []
    for name, group in grouped.items():
        label = _tool_label(name)
        if compact and len(group) == 1:
            detail = _key_argument(group[0])
            lines.append(f"⏵ {label}{f'({detail})' if detail else ''}")
            continue
        lines.append(f"  {label} × {len(group)}")
        for call in group:
            detail = _key_argument(call)
            if detail:
                lines.append(f"    └ {detail}")
    return "\n".join(lines)


def format_tool_results(calls: list[ToolCall], results: list[ToolOutput]) -> str:
    paired = list(zip(calls, results, strict=False))[:100]
    if not paired:
        return ""
    failed = sum(_failed(output) for _, output in paired)
    timed_out = sum(output.timed_out for _, output in paired)
    truncated = sum(output.truncated for _, output in paired)
    images = sum(len(output.image_urls) for _, output in paired)
    elapsed_ms = max((output.elapsed_ms for _, output in paired), default=0)
    parts = [f"{'⚠' if failed else '✓'} {len(paired)} tool(s)"]
    if failed:
        parts.append(f"{failed} failed")
    if timed_out:
        parts.append(f"{timed_out} timed out")
    if truncated:
        parts.append(f"{truncated} truncated")
    if images:
        parts.append(f"{images} image(s)")
    parts.append(_duration(elapsed_ms))
    summary = " · ".join(parts)
    blocked = [
        f"{_tool_label(call.name)}: {reason}"
        for call, output in paired
        if (reason := _policy_denial_reason(output)) is not None
    ]
    if blocked:
        summary += "\n⛔ Blocked by safety policy · " + "; ".join(blocked[:3])
    return summary


def _policy_denial_reason(output: ToolOutput) -> str | None:
    try:
        payload = decode_strict_json(
            output.text,
            max_bytes=800_000,
            max_depth=32,
            max_nodes=100_000,
        )
    except (RecursionError, TypeError, UnicodeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("policy_denied") is not True:
        return None
    return _safe_argument_text(payload.get("error", "Operation denied"))[:400]


def _tool_label(name: str) -> str:
    if name.startswith("mcp__"):
        parts = name.split("__", 2)
        return "🔌 MCP " + (f"{parts[1]}.{parts[2]}" if len(parts) == 3 else name)
    return _LABELS.get(name, f"🔧 {name}")


def _key_argument(call: ToolCall) -> str:
    key = _KEYS.get(call.name)
    if key is not None:
        value = call.arguments.get(key, "")
    else:
        value = call.arguments
    rendered = _safe_argument_text(value)
    if call.name == "web_fetch":
        rendered = rendered.removeprefix("https://").removeprefix("http://").rstrip("/")
    return rendered if len(rendered) <= 80 else rendered[:77] + "..."


def _safe_argument_text(value: object) -> str:
    if isinstance(value, str):
        raw = value
    else:
        try:
            raw = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            )
        except (RecursionError, TypeError, ValueError):
            raw = "<invalid arguments>"
    unicode_safe = raw.encode("utf-8", errors="replace").decode("utf-8")
    return sanitize_terminal_text(redact_sensitive_text(unicode_safe)).replace("\n", " ")


def _failed(output: ToolOutput) -> bool:
    if output.timed_out:
        return True
    return is_failed_tool_text(output.text)


def _duration(elapsed_ms: int) -> str:
    if elapsed_ms < 1_000:
        return f"{max(elapsed_ms, 0)} ms"
    return f"{elapsed_ms / 1_000:.1f} s"
