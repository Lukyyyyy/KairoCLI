from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import LlmResponse, Message, ToolCall, ToolOutput
from .paths import KairoPaths, reject_symlink_components
from .text_safety import bound_utf8, safe_text

MAX_TRACE_FILES = 50
MAX_CONTENT_CHARS = 200_000
MAX_TOOL_SCHEMAS_CHARS = 1_000_000
MAX_TRACE_REASONING_CHARS = 100_000
MAX_TRACE_ERROR_CHARS = 10_000

_DIV = "=" * 80
_SEP = "-" * 40


class LlmTraceLogger:
    """Opt-in full-session diagnostics that record the complete conversation for debugging.

    Each agent.run() call produces one human-readable .log file under the traces directory.
    Enabled via KAIROCLI_TRACE_ENABLED=true; add KAIROCLI_TRACE_REASONING=true to also
    capture extended reasoning. Sensitive credentials are always redacted from content.
    """

    def __init__(
        self,
        directory: Path,
        *,
        enabled: bool = False,
        include_reasoning: bool = False,
    ) -> None:
        self.directory = directory
        self.enabled = enabled
        self.include_reasoning = include_reasoning
        self._session_files: dict[str, Path] = {}

    @classmethod
    def from_environment(cls, paths: KairoPaths) -> LlmTraceLogger:
        return cls(
            paths.user_dir / "traces",
            enabled=_env_bool("KAIROCLI_TRACE_ENABLED", False),
            include_reasoning=_env_bool("KAIROCLI_TRACE_REASONING", False),
        )

    def open_session(self, *, scope: str, provider: str, model: str) -> str | None:
        """Create a trace file for a new agent run. Returns session_id, or None if disabled."""
        if not self.enabled:
            return None
        session_id = uuid.uuid4().hex[:12]
        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
        path = self.directory / f"sess-{ts}-{session_id}.log"
        header = (
            f"{_DIV}\n"
            f"KAIRO TRACE  {session_id}  {datetime.now(UTC).isoformat()}\n"
            f"scope={scope or 'unknown'}  provider={provider}  model={model}\n"
            f"{_DIV}\n\n"
        )
        self._session_files[session_id] = path
        self._write(path, header)
        return session_id

    async def record_llm_request(
        self,
        *,
        session_id: str | None,
        turn: int,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> None:
        """Record the complete logical request immediately before an LLM call."""
        if not self.enabled or session_id is None:
            return
        await asyncio.to_thread(
            self._record_llm_request_sync, session_id, turn, messages, tools
        )

    def _record_llm_request_sync(
        self,
        session_id: str,
        turn: int,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> None:
        path = self._session_files.get(session_id)
        if path is None:
            return
        lines = [
            f"[REQUEST #{turn}]  {datetime.now(UTC).isoformat()}  "
            f"msgs={len(messages)}  tools={len(tools)}",
            _SEP,
        ]
        for index, message in enumerate(messages, start=1):
            metadata = f"  tool_call_id={message.tool_call_id}" if message.tool_call_id else ""
            lines += [
                f"[MESSAGE #{index} {message.role.upper()}]{metadata}",
                _indent(_format_content(message.content)),
            ]
            if (
                self.include_reasoning
                and message.role == "assistant"
                and message.reasoning_content
                and message.reasoning_content.strip()
            ):
                reasoning = _clip(
                    redact_sensitive_text(message.reasoning_content),
                    MAX_TRACE_REASONING_CHARS,
                )
                lines += ["  [HISTORICAL REASONING]", _indent(reasoning, "    ")]
            for call in message.tool_calls:
                arguments = _clip(
                    redact_sensitive_text(_safe_json(call.arguments)), MAX_CONTENT_CHARS
                )
                lines += [
                    f"  [TOOL CALL]  {call.name}  id={call.id}",
                    _indent(arguments, "    "),
                ]
            lines.append("")
        schemas = _clip(
            redact_sensitive_text(_safe_json(tools)), MAX_TOOL_SCHEMAS_CHARS
        )
        lines += ["[TOOL SCHEMAS]", _indent(schemas), _SEP, ""]
        self._append(path, "\n".join(lines) + "\n")

    async def record_llm_turn(
        self,
        *,
        session_id: str | None,
        turn: int,
        response: LlmResponse,
        duration_ms: int,
        msg_count: int,
        tool_count: int,
    ) -> None:
        """Record a completed LLM call: stats, assistant text, and any tool calls requested."""
        if not self.enabled or session_id is None:
            return
        await asyncio.to_thread(
            self._record_llm_turn_sync,
            session_id, turn, response, duration_ms, msg_count, tool_count,
        )

    def _record_llm_turn_sync(
        self,
        session_id: str, turn: int, response: LlmResponse,
        duration_ms: int, msg_count: int, tool_count: int,
    ) -> None:
        path = self._session_files.get(session_id)
        if path is None:
            return
        lines: list[str] = [
            f"[CALL #{turn}]  {datetime.now(UTC).isoformat()}  "
            f"duration={duration_ms / 1000:.2f}s  msgs={msg_count}  tools={tool_count}  "
            f"in={response.usage.input_tokens} out={response.usage.output_tokens} "
            f"cached={response.usage.cache_tokens}",
            "",
        ]
        if (
            self.include_reasoning
            and response.reasoning_content
            and response.reasoning_content.strip()
        ):
            reasoning = _clip(
                redact_sensitive_text(response.reasoning_content), MAX_TRACE_REASONING_CHARS
            )
            lines += ["  [REASONING]", *[f"  | {line}" for line in reasoning.splitlines()], ""]
        if response.content and response.content.strip():
            text = _clip(redact_sensitive_text(response.content), MAX_CONTENT_CHARS)
            lines += ["  [ASSISTANT]", *[f"  {line}" for line in text.splitlines()], ""]
        for call in response.tool_calls:
            args_str = _clip(
                redact_sensitive_text(_safe_json(call.arguments)),
                10_000,
            )
            lines += [
                f"  [TOOL CALL]  {call.name}  id={call.id}",
                *[f"    {line}" for line in args_str.splitlines()],
                "",
            ]
        self._append(path, "\n".join(lines) + "\n")

    async def record_llm_error(
        self,
        *,
        session_id: str | None,
        turn: int,
        error: BaseException,
        duration_ms: int,
        msg_count: int,
        tool_count: int,
    ) -> None:
        """Record a failed LLM call."""
        if not self.enabled or session_id is None:
            return
        await asyncio.to_thread(
            self._record_llm_error_sync,
            session_id, turn, error, duration_ms, msg_count, tool_count,
        )

    def _record_llm_error_sync(
        self,
        session_id: str, turn: int, error: BaseException,
        duration_ms: int, msg_count: int, tool_count: int,
    ) -> None:
        path = self._session_files.get(session_id)
        if path is None:
            return
        err_msg = safe_redacted_text(error, MAX_TRACE_ERROR_CHARS, "...[truncated]")
        content = (
            f"[CALL #{turn}]  {datetime.now(UTC).isoformat()}  "
            f"duration={duration_ms / 1000:.2f}s  msgs={msg_count}  tools={tool_count}\n\n"
            f"  [ERROR]  {type(error).__name__}\n"
            + "\n".join(f"  {line}" for line in err_msg.splitlines())
            + "\n\n"
        )
        self._append(path, content)

    async def record_tool_results(
        self,
        *,
        session_id: str | None,
        calls: list[ToolCall],
        results: list[ToolOutput],
    ) -> None:
        """Record the outputs from tool execution."""
        if not self.enabled or session_id is None:
            return
        await asyncio.to_thread(self._record_tool_results_sync, session_id, calls, results)

    def _record_tool_results_sync(
        self, session_id: str, calls: list[ToolCall], results: list[ToolOutput]
    ) -> None:
        path = self._session_files.get(session_id)
        if path is None:
            return
        lines: list[str] = []
        for call, result in zip(calls, results, strict=True):
            parts = [f"id={call.id}"]
            if result.elapsed_ms:
                parts.append(f"elapsed={result.elapsed_ms}ms")
            if result.timed_out:
                parts.append("TIMED_OUT")
            if result.truncated:
                parts.append(f"truncated(original={result.original_chars})")
            text = _clip(redact_sensitive_text(result.text), MAX_CONTENT_CHARS)
            lines += [
                f"[TOOL RESULT]  {call.name}  {' '.join(parts)}",
                *[f"  {line}" for line in text.splitlines()],
                "",
            ]
            if result.has_images:
                lines += [f"  [{len(result.image_urls)} image(s) omitted]", ""]
        self._append(path, "\n".join(lines) + "\n")

    async def close_session(
        self,
        *,
        session_id: str | None,
        duration_ms: int,
        total_calls: int,
        total_input: int,
        total_output: int,
        total_cached: int,
        error: BaseException | None = None,
    ) -> None:
        """Write the session footer and prune old files."""
        if session_id is None:
            return
        await asyncio.to_thread(
            self._close_session_sync,
            session_id, duration_ms, total_calls,
            total_input, total_output, total_cached, error,
        )

    def _close_session_sync(
        self,
        session_id: str, duration_ms: int,
        total_calls: int, total_input: int, total_output: int, total_cached: int,
        error: BaseException | None,
    ) -> None:
        path = self._session_files.pop(session_id, None)
        if path is None:
            return
        status = f"ABORTED  {type(error).__name__}" if error is not None else "DONE"
        footer = (
            f"\n{_DIV}\n"
            f"{status}  duration={duration_ms / 1000:.2f}s  calls={total_calls}  "
            f"in={total_input} out={total_output} cached={total_cached}\n"
            f"{_DIV}\n"
        )
        self._append(path, footer)
        self._prune()

    def _write(self, path: Path, content: str) -> None:
        try:
            reject_symlink_components(path, "LLM trace")
            self.directory.mkdir(parents=True, exist_ok=True)
            if os.name == "posix":
                self.directory.chmod(0o700)
            flags = os.O_CREAT | os.O_WRONLY | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                if os.name == "posix":
                    os.fchmod(stream.fileno(), 0o600)
                stream.write(content.encode("utf-8", errors="replace"))
                stream.flush()
                os.fsync(stream.fileno())
        except (OSError, RecursionError, TypeError, UnicodeError, ValueError):
            pass

    def _append(self, path: Path, content: str) -> None:
        try:
            reject_symlink_components(path, "LLM trace")
            flags = os.O_CREAT | os.O_APPEND | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags, 0o600)
            with os.fdopen(descriptor, "ab") as stream:
                if os.name == "posix":
                    os.fchmod(stream.fileno(), 0o600)
                stream.write(content.encode("utf-8", errors="replace"))
                stream.flush()
                os.fsync(stream.fileno())
        except (OSError, RecursionError, TypeError, UnicodeError, ValueError):
            pass

    def _prune(self) -> None:
        try:
            candidates: list[tuple[float, Path]] = []
            for path in self.directory.glob("sess-*.log"):
                try:
                    reject_symlink_components(path, "LLM trace")
                    candidates.append((path.stat().st_mtime, path))
                except (OSError, ValueError):
                    continue
            for _, stale in sorted(candidates, reverse=True)[MAX_TRACE_FILES:]:
                try:
                    stale.unlink()
                except OSError:
                    pass
        except OSError:
            pass


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[truncated at {max_chars} chars]"


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(f"{prefix}{line}" if line else line for line in text.splitlines())


def _format_content(value: Any) -> str:
    if isinstance(value, str):
        return _clip(redact_sensitive_text(safe_text(value)), MAX_CONTENT_CHARS)
    if isinstance(value, list):
        parts = []
        for part in value:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text":
                parts.append(redact_sensitive_text(safe_text(part.get("text", ""))))
            elif ptype == "image_url":
                parts.append("[image]")
            else:
                parts.append(f"[{ptype or 'unknown'}]")
        return _clip("\n".join(parts), MAX_CONTENT_CHARS)
    return _clip(redact_sensitive_text(safe_text(value)), MAX_CONTENT_CHARS)


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    except (RecursionError, TypeError, ValueError):
        return safe_text(value)


_SENSITIVE_KEY = (
    r"[A-Za-z0-9_-]*(?:api[_-]?key|access[_-]?token|token|secret|password|authorization)"
)
_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----[\s\S]*?"
    r"(?:-----END (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----|\Z)",
    re.IGNORECASE,
)
_QUOTED_OR_BARE_SECRET = (
    r'(?:(?:"(?:\\.|[^"\\\r\n])*")|'
    r"(?:'(?:\\.|[^'\\\r\n])*')|[^\s,;}&#]+)"
)
_RAW_CREDENTIAL = re.compile(
    r"(?<![A-Za-z0-9_-])(?:"
    r"sk-(?:proj-)?[A-Za-z0-9_-]{16,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|"
    r"gh[pousr]_[A-Za-z0-9]{20,}|"
    r"xox[baprs]-[A-Za-z0-9-]{10,}|"
    r"(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}|"
    r"(?:hf|gsk|npm)_[A-Za-z0-9]{20,}|"
    r"pypi-[A-Za-z0-9_-]{30,}|"
    r"ya29\.[A-Za-z0-9_-]{20,}|"
    r"AIza[A-Za-z0-9_-]{25,}|"
    r"(?:AKIA|ASIA)[A-Z0-9]{16}"
    r")(?![A-Za-z0-9_-])"
)
_JWT_CREDENTIAL = re.compile(
    r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
    r"(?:\.[A-Za-z0-9_-]{8,})?(?![A-Za-z0-9_-])"
)


def redact_sensitive_text(value: str) -> str:
    redacted = _PRIVATE_KEY_BLOCK.sub("<private-key-redacted>", safe_text(value))
    redacted = re.sub(
        r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=]+",
        "<image-data-redacted>",
        redacted,
        flags=re.IGNORECASE,
    )
    redacted = re.sub(
        r"(?i)(authorization\s*:\s*)(?:bearer|basic|token|apikey)\s+[^\s,;]+",
        r"\1***",
        redacted,
    )
    redacted = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1***", redacted)
    redacted = re.sub(
        rf"(?i)(\b[\"']?{_SENSITIVE_KEY}[\"']?\s*[=:]\s*)"
        rf"{_QUOTED_OR_BARE_SECRET}",
        r"\1***",
        redacted,
    )
    redacted = re.sub(
        rf"(?i)(--(?:api[-_]?key|access[-_]?token|token|secret|password|"
        rf"authorization)\s*(?:=|\s)\s*){_QUOTED_OR_BARE_SECRET}",
        r"\1***",
        redacted,
    )
    redacted = re.sub(
        r"(?i)\b(https?://)[^/\s:@]+:[^/\s@]+@",
        r"\1***:***@",
        redacted,
    )
    redacted = re.sub(
        rf"(?i)([?&][\"']?{_SENSITIVE_KEY}[\"']?=)[^&#\s]*",
        r"\1***",
        redacted,
    )
    redacted = _RAW_CREDENTIAL.sub("<credential-redacted>", redacted)
    redacted = _JWT_CREDENTIAL.sub("<jwt-redacted>", redacted)
    redacted = re.sub(
        r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{240,}={0,2}(?![A-Za-z0-9+/])",
        "<base64-redacted>",
        redacted,
    )
    return redacted


def safe_redacted_text(value: object, maximum_bytes: int, marker: str) -> str:
    fallback = (
        f"{type(value).__name__} message unavailable"
        if isinstance(value, BaseException)
        else f"{type(value).__name__} text unavailable"
    )
    redacted = redact_sensitive_text(safe_text(value, fallback=fallback))
    return bound_utf8(redacted, maximum_bytes, marker)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on"}
