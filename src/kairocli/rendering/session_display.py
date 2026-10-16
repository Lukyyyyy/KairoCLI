from __future__ import annotations

import unicodedata
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from itertools import groupby

from ..models import Message, ToolCall, ToolOutput
from ..sessions import SessionMeta
from ..trace import redact_sensitive_text
from .terminal import sanitize_terminal_text
from .tool_display import format_tool_calls, format_tool_results

DEFAULT_SESSION_LIST_LIMIT = 20


def iter_session_history(
    messages: Sequence[Message], *, compact_tools: bool = True
) -> Iterator[tuple[str, str]]:
    """Replay conversation and tool summaries without exposing reasoning or result bodies."""
    pending: dict[str, ToolCall] = {}
    for role, group in groupby(messages, key=lambda message: message.role):
        if role == "tool":
            pairs = [
                (call, ToolOutput(text=_session_message_text(message)))
                for message in group
                if (call := pending.pop(message.tool_call_id or "", None)) is not None
            ]
            if pairs:
                # shortcut: only result text is stored; persist ToolOutput for timing and diffs.
                yield (
                    "tool_result",
                    format_tool_results(
                        [call for call, _ in pairs],
                        [output for _, output in pairs],
                        include_duration=False,
                    ),
                )
            continue
        if role not in {"user", "assistant"}:
            continue
        if role == "user":
            pending.clear()
        for message in group:
            text = sanitize_terminal_text(redact_sensitive_text(_session_message_text(message)))
            if text.strip():
                yield role, text
            if role == "assistant" and message.tool_calls:
                pending.update((call.id, call) for call in message.tool_calls)
                yield "tool_call", format_tool_calls(message.tool_calls, compact=compact_tools)


def _session_message_text(message: Message) -> str:
    content = message.content
    return (
        content
        if isinstance(content, str)
        else "\n".join(
            part["text"]
            for part in content
            if part.get("type") == "text" and isinstance(part.get("text"), str)
        )
    )


def format_session_list(
    sessions: Sequence[SessionMeta],
    current_id: str,
    *,
    show_all: bool = False,
    width: int = 100,
    now: datetime | None = None,
) -> str:
    """Format saved sessions without relying on terminal auto-wrapping."""

    if not sessions:
        return "No saved sessions."

    visible_candidates = [
        item for item in sessions if show_all or item.message_count > 0 or item.id == current_id
    ]
    visible = visible_candidates[:DEFAULT_SESSION_LIST_LIMIT]
    hidden_empty = sum(item.message_count == 0 and item.id != current_id for item in sessions)

    if not visible:
        lines = ["No non-empty saved sessions."]
    else:
        lines = [f"Sessions ({len(visible)})", ""]
        for index, item in enumerate(visible):
            marker = "●" if item.id == current_id else "○"
            title = item.title.strip() or "New session"
            title_budget = max(12, width - 2)
            lines.append(f"{marker} {_truncate_display(title, title_budget)}")

            count = f"{item.message_count} " + (
                "message" if item.message_count == 1 else "messages"
            )
            timestamp = _friendly_session_time(item.updated_at, now=now)
            status = "Current · " if item.id == current_id else ""
            metadata = f"  {status}{count} · {timestamp} · {item.id}"
            if _display_width(metadata) <= max(width, 40):
                lines.append(metadata)
            else:
                lines.append(f"  {status}{count} · {timestamp}")
                lines.append(f"  {item.id}")
            if index != len(visible) - 1:
                lines.append("")

    if not show_all and hidden_empty:
        if lines and lines[-1]:
            lines.append("")
        noun = "session" if hidden_empty == 1 else "sessions"
        lines.append(f"{hidden_empty} empty {noun} hidden · /session list --all")
    return "\n".join(lines)


def _friendly_session_time(value: str, *, now: datetime | None = None) -> str:
    current = now or datetime.now().astimezone()
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    local = parsed.astimezone(current.tzinfo)
    if local.date() == current.date():
        return f"Today {local:%H:%M}"
    if local.date() == (current - timedelta(days=1)).date():
        return f"Yesterday {local:%H:%M}"
    if local.year == current.year:
        return local.strftime("%b %d %H:%M")
    return local.strftime("%Y-%m-%d %H:%M")


def _truncate_display(value: str, maximum: int) -> str:
    if _display_width(value) <= maximum:
        return value
    budget = max(maximum - 1, 0)
    output: list[str] = []
    used = 0
    for character in value:
        character_width = _character_width(character)
        if used + character_width > budget:
            break
        output.append(character)
        used += character_width
    return "".join(output).rstrip() + "…"


def _display_width(value: str) -> int:
    return sum(_character_width(character) for character in value)


def _character_width(character: str) -> int:
    if unicodedata.combining(character):
        return 0
    if unicodedata.category(character).startswith("C"):
        return 0
    return 2 if unicodedata.east_asian_width(character) in {"F", "W"} else 1
