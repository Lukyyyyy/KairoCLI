from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from .json_boundary import decode_strict_json
from .paths import KairoPaths, reject_symlink_components
from .private_lock import private_file_lock
from .trace import redact_sensitive_text

MAX_AUDIT_EVENT_BYTES = 1024 * 1024
MAX_AUDIT_FILE_BYTES = 10 * 1024 * 1024
MAX_AUDIT_FILES_PER_DAY = 5
MAX_AUDIT_DETAIL_CHARS = 10_000


class PolicyDenied(PermissionError):
    pass


class ApprovalDecision(StrEnum):
    APPROVED = "approved"
    APPROVED_ALL = "approved_all"
    APPROVED_ALL_BY_SERVER = "approved_all_by_server"
    REJECTED = "rejected"
    MODIFIED = "modified"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class ApprovalResult:
    decision: ApprovalDecision
    modified_arguments: dict[str, Any] | None = None
    reason: str | None = None

    @classmethod
    def approve(cls) -> ApprovalResult:
        return cls(ApprovalDecision.APPROVED)

    @classmethod
    def approve_all(cls) -> ApprovalResult:
        return cls(ApprovalDecision.APPROVED_ALL)

    @classmethod
    def approve_all_by_server(cls) -> ApprovalResult:
        return cls(ApprovalDecision.APPROVED_ALL_BY_SERVER)

    @classmethod
    def reject(cls, reason: str | None = None) -> ApprovalResult:
        return cls(ApprovalDecision.REJECTED, reason=reason)

    @classmethod
    def modify(cls, arguments: dict[str, Any]) -> ApprovalResult:
        return cls(ApprovalDecision.MODIFIED, modified_arguments=arguments)

    @classmethod
    def skip(cls) -> ApprovalResult:
        return cls(ApprovalDecision.SKIPPED)

    @property
    def approved(self) -> bool:
        return self.decision in {
            ApprovalDecision.APPROVED,
            ApprovalDecision.APPROVED_ALL,
            ApprovalDecision.APPROVED_ALL_BY_SERVER,
            ApprovalDecision.MODIFIED,
        }

    def effective_arguments(self, original: dict[str, Any]) -> dict[str, Any]:
        if self.decision == ApprovalDecision.MODIFIED and self.modified_arguments is not None:
            return self.modified_arguments
        return original


class PathGuard:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()

    def resolve(self, value: str | Path, *, must_exist: bool = False) -> Path:
        self._reject_blank(value)
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self.workspace)
        except ValueError as exc:
            raise PolicyDenied(f"Path escapes workspace: {value}") from exc
        if must_exist and not resolved.exists():
            raise FileNotFoundError(resolved)
        return resolved

    def resolve_for_write(self, value: str | Path) -> Path:
        self._reject_blank(value)
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        lexical = Path(os.path.abspath(candidate))
        try:
            relative = lexical.relative_to(self.workspace)
        except ValueError as exc:
            raise PolicyDenied(f"Path escapes workspace: {value}") from exc
        current = self.workspace
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise PolicyDenied(f"Write path cannot traverse a symlink: {value}")
            if current.exists() and current != lexical and not current.is_dir():
                raise PolicyDenied(f"Write path parent is not a directory: {value}")
        return lexical

    @staticmethod
    def _reject_blank(value: str | Path) -> None:
        if value is None or (isinstance(value, str) and not value.strip()):
            raise PolicyDenied("Path cannot be empty")


class CommandGuard:
    _denied = (
        re.compile(r"\bsudo\b", re.I),
        re.compile(
            r"\brm\s+"
            r"(?=[^;&|]*(?:-[a-z]*r|--recursive))"
            r"(?=[^;&|]*(?:-[a-z]*f|--force))"
            r"[^;&|]*\s(?:/|~|\$home)",
            re.I,
        ),
        re.compile(r"\b(?:mkfs(?:\.|\b)|fdisk\b)", re.I),
        re.compile(r"\bdd\b.*\bof=/dev/", re.I),
        re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", re.I),
        re.compile(r"\b(?:curl|wget)\b[^|]*\|\s*(?:sh|bash|zsh|fish|ksh)\b", re.I),
        re.compile(r"\bfind\s+(?:/|~|\$home)", re.I),
        re.compile(r"\bchmod\s+-R\s+777\s+(?:/|~)", re.I),
        re.compile(r"\b(?:shutdown|reboot|halt|poweroff)\b", re.I),
    )

    def check(self, command: str) -> None:
        normalized = re.sub(r"\s+", " ", command).strip()
        if not normalized:
            raise PolicyDenied("Command cannot be empty")
        for pattern in self._denied:
            if pattern.search(normalized):
                raise PolicyDenied("Command rejected by safety policy")


@dataclass(slots=True)
class AuditEntry:
    timestamp: str
    tool: str
    decision: str
    arguments: dict[str, Any]
    detail: str = ""


class AuditLog:
    def __init__(self, paths: KairoPaths) -> None:
        self.directory = paths.audit_dir

    def append(self, tool: str, decision: str, arguments: dict[str, Any], detail: str = "") -> None:
        try:
            redacted_arguments = _redact_arguments(tool, arguments)
        except (RecursionError, TypeError, ValueError):
            redacted_arguments = {"_serialization_error": True}
        entry = AuditEntry(
            datetime.now(UTC).isoformat(),
            tool,
            decision,
            redacted_arguments,
            redact_sensitive_text(detail)[:MAX_AUDIT_DETAIL_CHARS],
        )
        try:
            serialized = json.dumps(
                asdict(entry),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            encoded = (serialized + "\n").encode("utf-8")
        except (RecursionError, TypeError, UnicodeError, ValueError):
            return
        if len(encoded) > MAX_AUDIT_EVENT_BYTES:
            entry.arguments = {
                "_truncated": True,
                "original_bytes": len(encoded),
                "keys": sorted(str(key)[:128] for key in arguments)[:100],
            }
            try:
                encoded = (
                    json.dumps(
                        asdict(entry),
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
            except (RecursionError, TypeError, UnicodeError, ValueError):
                return
        try:
            with private_file_lock(self.directory, ".audit.lock", "Audit log"):
                target = self.directory / f"audit-{datetime.now().date().isoformat()}.jsonl"
                reject_symlink_components(target, "Audit log")
                if target.is_file() and target.stat().st_size + len(encoded) > MAX_AUDIT_FILE_BYTES:
                    if not self._rotate(target):
                        return
                flags = os.O_CREAT | os.O_APPEND | os.O_WRONLY
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                descriptor = os.open(target, flags, 0o600)
                with os.fdopen(descriptor, "ab") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                if os.name == "posix":
                    target.chmod(0o600)
        except (OSError, ValueError):
            # Audit diagnostics must not change tool execution semantics.
            return

    def _rotate(self, target: Path) -> bool:
        for index in range(MAX_AUDIT_FILES_PER_DAY - 1, 0, -1):
            source = target if index == 1 else target.with_name(f"{target.stem}.{index - 1}.jsonl")
            destination = target.with_name(f"{target.stem}.{index}.jsonl")
            if source.is_symlink() or destination.is_symlink():
                return False
            if source.exists():
                source.replace(destination)
        return True


def read_recent_audit(paths: KairoPaths, limit: int = 10) -> str:
    bounded_limit = max(1, min(limit, 100))
    target = paths.audit_dir / f"audit-{datetime.now().date().isoformat()}.jsonl"
    try:
        reject_symlink_components(target, "Audit log")
    except ValueError:
        return "Audit log is unavailable because the file is a symlink."
    if not target.is_file():
        return "No audit entries today."
    try:
        with private_file_lock(paths.audit_dir, ".audit.lock", "Audit log"):
            size = target.stat().st_size
            with target.open("rb") as stream:
                stream.seek(max(0, size - 1024 * 1024))
                raw = stream.read(1024 * 1024)
    except (OSError, ValueError) as exc:
        return f"Audit log unavailable: {type(exc).__name__}"
    lines = raw.splitlines()
    if size > len(raw) and lines:
        lines = lines[1:]
    selected: list[str] = []
    for line in reversed(lines):
        try:
            payload = decode_strict_json(
                line,
                max_bytes=MAX_AUDIT_EVENT_BYTES,
                max_depth=32,
                max_nodes=100_000,
            )
        except (RecursionError, TypeError, UnicodeError, ValueError):
            continue
        if not _valid_audit_payload(payload):
            continue
        selected.append(
            json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        )
        if len(selected) >= bounded_limit:
            break
    selected.reverse()
    return "\n".join(selected) or "No audit entries today."


class ApprovalPolicy:
    DANGEROUS_TOOLS = {
        "write_file",
        "apply_patch",
        "execute_command",
        "shell_exec",
        "create_project",
        "install_skill",
        "revert_turn",
        "browser_connect",
        "browser_disconnect",
    }

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._approved_tools: set[str] = set()
        self._approved_mcp_servers: set[str] = set()

    def needs_approval(self, tool_name: str) -> bool:
        if not self.enabled or tool_name in self._approved_tools:
            return False
        server = self.mcp_server_name(tool_name)
        if server is not None and server in self._approved_mcp_servers:
            return False
        return tool_name in self.DANGEROUS_TOOLS or server is not None

    @classmethod
    def governs(cls, tool_name: str) -> bool:
        """Return whether a tool participates in the HITL approval policy."""
        return tool_name in cls.DANGEROUS_TOOLS or cls.mcp_server_name(tool_name) is not None

    def remember(self, tool_name: str, result: ApprovalResult) -> None:
        if result.decision == ApprovalDecision.APPROVED_ALL:
            self._approved_tools.add(tool_name)
        elif result.decision == ApprovalDecision.APPROVED_ALL_BY_SERVER:
            server = self.mcp_server_name(tool_name)
            if server is not None:
                self._approved_mcp_servers.add(server)

    def clear_session_approvals(self) -> None:
        self._approved_tools.clear()
        self._approved_mcp_servers.clear()

    def clear_mcp_server_approvals(self, server_name: str) -> None:
        """Invalidate every cached approval tied to one restarted MCP server."""
        self._approved_mcp_servers.discard(server_name)
        self._approved_tools = {
            tool for tool in self._approved_tools if self.mcp_server_name(tool) != server_name
        }

    @staticmethod
    def mcp_server_name(tool_name: str) -> str | None:
        if not tool_name.startswith("mcp__"):
            return None
        parts = tool_name.split("__", 2)
        return parts[1] if len(parts) == 3 and parts[1] else None


def _redact_arguments(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    sensitive = re.compile(r"(?:api[_-]?key|token|secret|password|authorization)", re.I)
    assignment = re.compile(
        r"(?i)(\b(?:api[_-]?key|token|secret|password|authorization)\b\s*[=:]\s*)([^\s]+)"
    )
    flags = re.compile(r"(?i)(--(?:api-key|token|secret|password|authorization)\s+)([^\s]+)")
    nodes = [0]

    def redact(value: Any, key: str, depth: int) -> Any:
        nodes[0] += 1
        if nodes[0] > 100_000 or depth > 32:
            raise ValueError("Audit arguments are too complex")
        if sensitive.search(key):
            return "***"
        if (tool == "write_file" and key == "content") or (
            tool == "apply_patch" and key == "patch"
        ):
            return f"<{key} redacted>"
        if isinstance(value, str):
            return flags.sub(r"\1***", assignment.sub(r"\1***", value))
        if value is None or isinstance(value, bool) or isinstance(value, int):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("Audit arguments must contain finite numbers")
            return value
        if isinstance(value, list):
            return [redact(item, "", depth + 1) for item in value]
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for nested_key, nested_value in value.items():
                if not isinstance(nested_key, str):
                    raise TypeError("Audit argument keys must be strings")
                result[nested_key] = redact(nested_value, nested_key, depth + 1)
            return result
        raise TypeError("Audit arguments must be standard JSON values")

    return {key: redact(value, key, 1) for key, value in arguments.items()}


def _valid_audit_payload(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("timestamp"), str)
        and isinstance(payload.get("tool"), str)
        and isinstance(payload.get("decision"), str)
        and isinstance(payload.get("arguments"), dict)
        and isinstance(payload.get("detail", ""), str)
    )
