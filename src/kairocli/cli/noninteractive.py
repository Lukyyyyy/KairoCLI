from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
from typing import Any

from ..agent import Agent, AgentCanceled, AgentOrchestrator, PlanExecuteAgent
from ..config import (
    AppConfig,
    normalize_provider_name,
)
from ..image import prepare_image_input
from ..llm import MAX_LLM_RESPONSE_BYTES
from ..mcp import (
    McpServerManager,
)
from ..paths import KairoPaths
from ..policy import ApprovalPolicy, ApprovalResult
from ..rendering.terminal import sanitize_terminal_text
from ..sessions import SessionStore, apply_session
from ..snapshot import SnapshotError, SnapshotService, turn_snapshot_messages
from ..text_safety import safe_text
from ..todos import SessionTodoController
from ..trace import redact_sensitive_text
from ..user_input import (
    MAX_USER_INPUT_BYTES,
    UserInputError,
    normalize_user_input,
)
from .bootstrap import _inject_mcp_resource_index, _register_browser_agent_tools, make_agent
from .interactive import _safe_cli_error, expand_local_mentions

log = logging.getLogger(__name__)
MAX_INTERACTIVE_ERROR_BYTES = 4_000


MAX_NONINTERACTIVE_INPUT_BYTES = MAX_USER_INPUT_BYTES
MAX_NONINTERACTIVE_RESULT_BYTES = MAX_LLM_RESPONSE_BYTES
MAX_NONINTERACTIVE_ERROR_BYTES = 4_000
MAX_NONINTERACTIVE_WARNING_BYTES = 2_000
MAX_NONINTERACTIVE_WARNINGS = 100


async def noninteractive(
    paths: KairoPaths,
    config: AppConfig,
    prompt_argument: str,
    *,
    provider: str | None = None,
    output_format: str = "text",
    mode: str = "agent",
    resume_id: str | None = None,
    continue_session: bool = False,
    save_session: bool = False,
    allowed_tools: list[str] | None = None,
    bypass_approvals: bool = False,
) -> int:
    started = time.monotonic()
    warnings: list[str] = []
    prompt = prompt_argument
    if not prompt:
        prompt = await asyncio.to_thread(_read_noninteractive_stdin)
    try:
        prompt = normalize_user_input(prompt)
    except UserInputError as exc:
        return _emit_noninteractive_error(
            output_format,
            "InputLimitError",
            _safe_cli_error(exc),
            started,
        )
    if not prompt.strip():
        return _emit_noninteractive_error(
            output_format,
            "InputError",
            "Noninteractive input cannot be empty",
            started,
        )
    allow = set(allowed_tools or [])
    if any(not re.fullmatch(r"[A-Za-z0-9_.:-]+", name) for name in allow):
        return _emit_noninteractive_error(
            output_format,
            "ApprovalConfigError",
            "--allow-tool names may contain only letters, digits, _, ., :, and -",
            started,
        )

    async def approve(tool_name: str, _arguments: dict[str, Any]) -> ApprovalResult:
        if tool_name in allow:
            return ApprovalResult.approve()
        return ApprovalResult.reject(
            f"Noninteractive mode denied {tool_name}; use --allow-tool {tool_name} "
            "or --dangerously-skip-approvals only when explicitly authorized"
        )

    persistent = bool(save_session or resume_id or continue_session)
    agent: Agent | None = None
    try:
        store = SessionStore(paths.session_database) if persistent else None
        restored = None
        if store is not None and resume_id:
            restored = await asyncio.to_thread(store.load, resume_id, paths.workspace)
            if restored is None:
                return _emit_noninteractive_error(
                    output_format,
                    "SessionError",
                    "Session was not found in the current workspace",
                    started,
                )
        elif store is not None and continue_session:
            restored = await asyncio.to_thread(store.latest, paths.workspace)
        selected_provider = normalize_provider_name(provider or config.default_provider)
        if restored is not None and provider is None and restored.meta.provider in config.providers:
            selected_provider = restored.meta.provider
        todo_controller = (
            SessionTodoController(store, paths.workspace) if store is not None else None
        )
        approvals = ApprovalPolicy(not bypass_approvals)

        agent = make_agent(
            paths,
            config,
            selected_provider,
            approvals,
            approve,
            todo_controller=todo_controller,
        )
        session_id: str | None = None
        if restored is not None:
            apply_session(agent, restored)
            session_id = restored.meta.id
        elif store is not None:
            created = await asyncio.to_thread(
                store.create,
                paths.workspace,
                agent.llm.provider,
                agent.llm.model,
            )
            session_id = created.meta.id
        if todo_controller is not None and session_id is not None:
            todo_controller.attach(session_id)
    except Exception as exc:
        if agent is not None:
            try:
                await agent.tools.close()
            except Exception as cleanup_exc:
                warnings.append(
                    "Tool shutdown warning: " + _noninteractive_exception_text(cleanup_exc)
                )
        return _emit_noninteractive_error(
            output_format,
            type(exc).__name__,
            _noninteractive_exception_text(exc),
            started,
            warnings=warnings,
        )
    assert agent is not None

    manager = McpServerManager(paths, agent.tools)
    browser_guard = agent.tools.browser_guard
    if browser_guard is not None:
        _register_browser_agent_tools(agent, browser_guard.session, manager)
    start_event = {
        "schema_version": 1,
        "type": "start",
        "status": "running",
        "mode": mode,
        "provider": agent.llm.provider,
        "model": agent.llm.model,
        "session_id": session_id,
    }
    if output_format == "jsonl":
        _write_json(start_event)
    answer = ""
    error: BaseException | None = None
    captured_snapshot = False
    post_snapshot_captured = False
    snapshots: SnapshotService = agent.tools.snapshot_service
    pre_message, post_message = turn_snapshot_messages(mode, prompt)
    try:
        await manager.start_all()
        _inject_mcp_resource_index(agent, manager)
        prepared = await prepare_image_input(
            prompt,
            paths.workspace,
            paths.user_dir / "cache" / "clipboard",
        )
        image_urls = list(prepared.image_urls)
        expanded = expand_local_mentions(prepared.text, paths.workspace)
        expanded = await manager.expand_resource_mentions(expanded)
        if snapshots.config.enabled:
            try:
                await snapshots.capture(pre_message)
                captured_snapshot = True
            except SnapshotError as exc:
                warnings.append("Snapshot warning: " + _noninteractive_exception_text(exc))
        if mode == "plan":
            answer = await PlanExecuteAgent(agent).run(expanded, image_urls)
        elif mode == "team":
            answer = await AgentOrchestrator(agent).run(expanded, image_urls)
        else:
            answer = await agent.run(expanded, image_urls)
        if captured_snapshot:
            try:
                await snapshots.capture(post_message)
                post_snapshot_captured = True
            except SnapshotError as exc:
                warnings.append("Snapshot warning: " + _noninteractive_exception_text(exc))
    except Exception as exc:
        error = exc
    finally:
        if captured_snapshot and not post_snapshot_captured:
            try:
                await snapshots.capture(post_message)
            except SnapshotError as exc:
                warnings.append("Snapshot warning: " + _noninteractive_exception_text(exc))
        if store is not None and session_id is not None:
            snapshot = store.capture_snapshot(agent)
            try:
                await asyncio.to_thread(
                    store.save_snapshot,
                    session_id,
                    paths.workspace,
                    snapshot,
                )
            except Exception as exc:
                warnings.append("Session warning: " + _noninteractive_exception_text(exc))
        try:
            await manager.close()
        except Exception as exc:
            warnings.append("MCP shutdown warning: " + _noninteractive_exception_text(exc))
        try:
            await agent.tools.close()
        except Exception as exc:
            warnings.append("Tool shutdown warning: " + _noninteractive_exception_text(exc))
    duration_ms = int((time.monotonic() - started) * 1_000)
    usage = {
        "input_tokens": agent.total_input_tokens,
        "output_tokens": agent.total_output_tokens,
        "cached_tokens": agent.total_cached_tokens,
        "llm_calls": agent.llm_call_count,
        "compactions": agent.compaction_count,
    }
    if error is not None:
        status = "canceled" if isinstance(error, AgentCanceled) else "error"
        payload = {
            **start_event,
            "type": "result",
            "status": status,
            "error": {
                "type": type(error).__name__,
                "message": _noninteractive_exception_text(error),
            },
            "usage": usage,
            "duration_ms": duration_ms,
            "warnings": warnings,
        }
        _emit_noninteractive(output_format, payload)
        return 130 if status == "canceled" else 1
    payload = {
        **start_event,
        "type": "result",
        "status": "success",
        "result": answer,
        "usage": usage,
        "duration_ms": duration_ms,
        "warnings": warnings,
    }
    _emit_noninteractive(output_format, payload)
    return 0


def _emit_noninteractive(output_format: str, payload: dict[str, Any]) -> None:
    if output_format == "text":
        payload = _normalize_noninteractive_payload(payload)
        if payload["status"] == "success":
            print(sanitize_terminal_text(str(payload.get("result", ""))))
        else:
            error = payload.get("error", {})
            print(
                sanitize_terminal_text(f"Kairo CLI: {error.get('message', 'unknown error')}"),
                file=sys.stderr,
            )
        for warning in payload.get("warnings", []):
            print(sanitize_terminal_text(str(warning)), file=sys.stderr)
    else:
        _write_json(payload)


def _read_noninteractive_stdin() -> str:
    binary = getattr(sys.stdin, "buffer", None)
    if binary is not None:
        raw = bytes(binary.read(MAX_NONINTERACTIVE_INPUT_BYTES + 1))
        return raw.decode("utf-8", errors="replace")
    return str(sys.stdin.read(MAX_NONINTERACTIVE_INPUT_BYTES + 1))


def _emit_noninteractive_error(
    output_format: str,
    error_type: str,
    message: str,
    started: float,
    status: str = "error",
    warnings: list[str] | None = None,
) -> int:
    payload = {
        "schema_version": 1,
        "type": "result",
        "status": status,
        "error": {"type": error_type, "message": message},
        "duration_ms": int((time.monotonic() - started) * 1_000),
        "warnings": list(warnings or []),
    }
    _emit_noninteractive(output_format, payload)
    return 1


def _write_json(payload: dict[str, Any]) -> None:
    safe_payload = _normalize_noninteractive_payload(payload)
    encoded = json.dumps(
        safe_payload,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    print(encoded, flush=True)


def _normalize_noninteractive_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {
        "schema_version": 1,
        "type": _bounded_noninteractive_text(payload.get("type"), 32),
        "status": _bounded_noninteractive_text(payload.get("status"), 32),
    }
    for key, limit in (("mode", 32), ("provider", 256), ("model", 1_024)):
        if key in payload:
            normalized[key] = _bounded_noninteractive_text(payload.get(key), limit)
    if "session_id" in payload:
        session_id = payload.get("session_id")
        normalized["session_id"] = (
            _bounded_noninteractive_text(session_id, 256) if isinstance(session_id, str) else None
        )
    if "result" in payload:
        normalized["result"] = _bounded_noninteractive_text(
            payload.get("result"), MAX_NONINTERACTIVE_RESULT_BYTES
        )
    error = payload.get("error")
    if isinstance(error, dict):
        normalized["error"] = {
            "type": _bounded_noninteractive_text(error.get("type"), 256).replace("\n", " "),
            "message": _bounded_noninteractive_text(
                error.get("message"),
                MAX_NONINTERACTIVE_ERROR_BYTES,
                redact=True,
            ),
        }
    usage = payload.get("usage")
    if isinstance(usage, dict):
        normalized["usage"] = {
            key: _noninteractive_counter(usage.get(key))
            for key in (
                "input_tokens",
                "output_tokens",
                "cached_tokens",
                "llm_calls",
                "compactions",
            )
        }
    if "duration_ms" in payload:
        normalized["duration_ms"] = _noninteractive_counter(payload.get("duration_ms"))
    raw_warnings = payload.get("warnings")
    if isinstance(raw_warnings, list):
        normalized["warnings"] = [
            _bounded_noninteractive_text(
                warning,
                MAX_NONINTERACTIVE_WARNING_BYTES,
                redact=True,
            )
            for warning in raw_warnings[:MAX_NONINTERACTIVE_WARNINGS]
        ]
        omitted = len(raw_warnings) - MAX_NONINTERACTIVE_WARNINGS
        if omitted > 0:
            normalized["warnings"].append(f"{omitted} additional warning(s) omitted")
    return normalized


def _bounded_noninteractive_text(value: Any, maximum_bytes: int, *, redact: bool = False) -> str:
    if not isinstance(value, str):
        return ""
    unicode_safe = value.encode("utf-8", errors="replace").decode("utf-8")
    if redact:
        unicode_safe = redact_sensitive_text(unicode_safe)
    raw = unicode_safe.encode("utf-8")
    if len(raw) <= maximum_bytes:
        return unicode_safe
    marker = f"...[truncated: original_bytes={len(raw)}]"
    marker_bytes = marker.encode("utf-8")
    if len(marker_bytes) >= maximum_bytes:
        return marker_bytes[:maximum_bytes].decode("utf-8", errors="ignore")
    available = max(0, maximum_bytes - len(marker_bytes))
    prefix = raw[:available].decode("utf-8", errors="ignore")
    return prefix + marker


def _noninteractive_counter(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, min(value, 1_000_000_000_000))


def _noninteractive_exception_text(exc: BaseException) -> str:
    message = safe_text(exc, fallback="Exception message was unavailable")
    return message or type(exc).__name__
