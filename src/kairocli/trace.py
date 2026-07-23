from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import LlmResponse
from .paths import KairoPaths, reject_symlink_components
from .private_lock import private_file_lock
from .text_safety import bound_utf8, safe_text

MAX_TRACE_FILE_BYTES = 10 * 1024 * 1024
MAX_TRACE_FILES = 10
MAX_TRACE_REASONING_CHARS = 100_000
MAX_TRACE_ERROR_CHARS = 10_000


class LlmTraceLogger:
    """Opt-in private JSONL diagnostics that never record prompts or answer content."""

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

    @classmethod
    def from_environment(cls, paths: KairoPaths) -> LlmTraceLogger:
        return cls(
            paths.user_dir / "traces",
            enabled=_env_bool("KAIROCLI_TRACE_ENABLED", False),
            include_reasoning=_env_bool("KAIROCLI_TRACE_REASONING", False),
        )

    async def record_success(
        self,
        *,
        scope: str,
        provider: str,
        model: str,
        duration_ms: int,
        message_count: int,
        tool_schema_count: int,
        response: LlmResponse,
    ) -> None:
        if not self.enabled:
            return
        event: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "trace_id": f"trace_{uuid.uuid4().hex[:16]}",
            "scope": scope or "unknown",
            "provider": provider,
            "model": model,
            "status": "success",
            "duration_ms": max(0, duration_ms),
            "message_count": max(0, message_count),
            "tool_schema_count": max(0, tool_schema_count),
            "response": {
                "content_chars": len(response.content),
                "reasoning_chars": len(response.reasoning_content or ""),
                "tool_calls": len(response.tool_calls),
                "streamed": response.streamed,
            },
            "usage": {
                "input_tokens": max(0, response.usage.input_tokens),
                "output_tokens": max(0, response.usage.output_tokens),
                "cached_tokens": max(0, response.usage.cache_tokens),
            },
        }
        if self.include_reasoning and response.reasoning_content:
            event["reasoning"] = redact_sensitive_text(
                response.reasoning_content[:MAX_TRACE_REASONING_CHARS]
            )
            event["reasoning_truncated"] = (
                len(response.reasoning_content) > MAX_TRACE_REASONING_CHARS
            )
        await self._append(event)

    async def record_error(
        self,
        *,
        scope: str,
        provider: str,
        model: str,
        duration_ms: int,
        message_count: int,
        tool_schema_count: int,
        error: BaseException,
    ) -> None:
        if not self.enabled:
            return
        await self._append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "trace_id": f"trace_{uuid.uuid4().hex[:16]}",
                "scope": scope or "unknown",
                "provider": provider,
                "model": model,
                "status": "error",
                "duration_ms": max(0, duration_ms),
                "message_count": max(0, message_count),
                "tool_schema_count": max(0, tool_schema_count),
                "error": {
                    "type": type(error).__name__,
                    "message": safe_redacted_text(
                        error, MAX_TRACE_ERROR_CHARS, "...[trace error truncated]"
                    ),
                },
            }
        )

    async def _append(self, event: dict[str, Any]) -> None:
        try:
            await asyncio.to_thread(self._append_sync, event)
        except (OSError, RecursionError, TypeError, UnicodeError, ValueError):
            # Diagnostics must never change Agent success or failure semantics.
            return

    def _append_sync(self, event: dict[str, Any]) -> None:
        serialized = (
            json.dumps(
                event,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        encoded = serialized.encode("utf-8")
        with private_file_lock(self.directory, ".trace.lock", "LLM trace"):
            target = self._target(len(encoded))
            reject_symlink_components(target, "LLM trace")
            flags = os.O_CREAT | os.O_APPEND | os.O_WRONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(target, flags, 0o600)
            with os.fdopen(descriptor, "ab") as stream:
                if os.name == "posix":
                    os.fchmod(stream.fileno(), 0o600)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            self._prune()

    def _target(self, incoming_bytes: int) -> Path:
        stem = f"llm-trace-{datetime.now(UTC).date().isoformat()}"
        for suffix in range(MAX_TRACE_FILES):
            target = self.directory / f"{stem}{f'-{suffix}' if suffix else ''}.jsonl"
            reject_symlink_components(target, "LLM trace")
            try:
                size = target.stat().st_size
            except OSError:
                size = 0
            if size + incoming_bytes <= MAX_TRACE_FILE_BYTES:
                return target
        return self.directory / f"{stem}-{uuid.uuid4().hex[:8]}.jsonl"

    def _prune(self) -> None:
        files_with_mtime: list[tuple[float, Path]] = []
        for candidate in self.directory.glob("llm-trace-*.jsonl"):
            try:
                reject_symlink_components(candidate, "LLM trace")
                files_with_mtime.append((candidate.stat().st_mtime, candidate))
            except (OSError, ValueError):
                continue
        files = [path for _, path in sorted(files_with_mtime, reverse=True)]
        for stale in files[MAX_TRACE_FILES:]:
            try:
                stale.unlink()
            except OSError:
                continue


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
