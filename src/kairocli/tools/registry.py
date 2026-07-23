from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..browser import BrowserGuard
from ..cancellation import AgentCanceled, raise_if_canceled, wait_with_cancellation
from ..json_boundary import decode_strict_json
from ..lsp import LspManager
from ..models import FileDiff, ToolOutput
from ..policy import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalResult,
    AuditLog,
    CommandGuard,
    PathGuard,
    PolicyDenied,
)
from ..rag import CodeIndex
from ..text_safety import safe_text
from ..tool_result import is_failed_tool_text
from ..trace import redact_sensitive_text
from .filesystem import (
    _atomic_write_text,
    _create_project_atomic,
    _glob_paths,
    _read_file_range,
    _suggested_reads,
    _take_text_budget,
)
from .limits import (
    MAX_GREP_MAX_CHARS,
    MAX_PATCH_BYTES,
    MAX_READ_FILE_CHARS,
    MAX_READ_FILE_LINES,
)
from .process import _read_bounded_stream, _terminate_process_tree
from .schemas import (
    _command_schema,
    _create_schema,
    _glob_schema,
    _grep_schema,
    _list_schema,
    _lsp_inspect_schema,
    _lsp_workspace_schema,
    _patch_schema,
    _path_schema,
    _query_schema,
    _shell_exec_schema,
    _shell_id_schema,
    _shell_start_schema,
    _url_schema,
    _web_search_schema,
    _write_schema,
)
from .shell import MAX_SHELL_COMMAND_BYTES, ShellSessionManager
from .validation import ToolArgumentsError, _validate_patch, _validate_tool_arguments
from .web import WebClient

ToolHandler = Callable[[dict[str, Any]], Awaitable[Any]]
ApprovalHandler = Callable[[str, dict[str, Any]], Awaitable[bool | ApprovalResult]]
_cancel_event: ContextVar[asyncio.Event | None] = ContextVar("tool_cancel_event", default=None)
log = logging.getLogger(__name__)
DEFAULT_READ_FILE_CHARS = 100_000
MAX_COMMAND_OUTPUT_BYTES = 50_000
MAX_GIT_APPLY_OUTPUT_BYTES = 50_000
DEFAULT_GREP_MAX_CHARS = 24_000
MAX_GREP_FILE_BYTES = 5 * 1024 * 1024
MAX_TOOL_OUTPUT_CHARS = 200_000
MAX_TOOL_ERROR_CHARS = 4_000
MAX_DISPLAY_DIFF_CHARS = 200_000
MAX_DISPLAY_DIFF_LINES = 2_000
MAX_DISPLAY_DIFF_FILES = 20
TOOL_CANCEL_GRACE_SECONDS = 0.1
MCP_TIMEOUT_GRACE_SECONDS = 5.0
SERIALIZED_WORKSPACE_MUTATION_TOOLS = frozenset(
    {
        "write_file",
        "apply_patch",
        "create_project",
        "install_skill",
        "revert_turn",
        "execute_command",
        "shell_exec",
    }
)


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class _AsyncServerGate:
    """Fair shared/exclusive gate for MCP calls and lifecycle transitions."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    async def acquire_shared(self) -> None:
        async with self._condition:
            await self._condition.wait_for(lambda: not self._writer and self._waiting_writers == 0)
            self._readers += 1

    async def release_shared(self) -> None:
        async with self._condition:
            if self._readers <= 0:
                raise RuntimeError("MCP shared gate released without ownership")
            self._readers -= 1
            if self._readers == 0:
                self._condition.notify_all()

    async def acquire_exclusive(self) -> None:
        async with self._condition:
            self._waiting_writers += 1
            try:
                await self._condition.wait_for(lambda: not self._writer and self._readers == 0)
                self._writer = True
            finally:
                self._waiting_writers -= 1

    async def release_exclusive(self) -> None:
        async with self._condition:
            if not self._writer:
                raise RuntimeError("MCP exclusive gate released without ownership")
            self._writer = False
            self._condition.notify_all()


async def _acquire_lock_with_cancellation(
    lock: asyncio.Lock, cancel_event: asyncio.Event | None
) -> None:
    """Acquire a lock without leaking ownership at a cancellation race."""
    raise_if_canceled(cancel_event)
    acquire = asyncio.create_task(lock.acquire())
    try:
        await wait_with_cancellation(acquire, cancel_event)
    except BaseException:
        if not acquire.done():
            acquire.cancel()
            await asyncio.gather(acquire, return_exceptions=True)
        if not acquire.cancelled() and acquire.exception() is None and acquire.result():
            lock.release()
        raise


async def _acquire_mcp_shared_with_cancellation(
    gate: _AsyncServerGate, cancel_event: asyncio.Event | None
) -> None:
    """Acquire shared ownership without leaking a reader at cancellation."""
    raise_if_canceled(cancel_event)
    acquire = asyncio.create_task(gate.acquire_shared())
    try:
        await wait_with_cancellation(acquire, cancel_event)
    except BaseException:
        if not acquire.done():
            acquire.cancel()
            await asyncio.gather(acquire, return_exceptions=True)
        if not acquire.cancelled() and acquire.exception() is None:
            await _release_mcp_shared_safely(gate)
        raise


async def _release_mcp_shared_safely(gate: _AsyncServerGate) -> None:
    release = asyncio.create_task(gate.release_shared())
    try:
        await asyncio.shield(release)
    except asyncio.CancelledError:
        await release
        raise


async def _release_mcp_exclusive_safely(gate: _AsyncServerGate) -> None:
    release = asyncio.create_task(gate.release_exclusive())
    try:
        await asyncio.shield(release)
    except asyncio.CancelledError:
        await release
        raise


class ToolRegistry:
    def __init__(
        self,
        workspace: Path,
        audit: AuditLog | None = None,
        approval_policy: ApprovalPolicy | None = None,
        approver: ApprovalHandler | None = None,
        tool_timeout_seconds: float = 90,
        browser_guard: BrowserGuard | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.path_guard = PathGuard(self.workspace)
        self.command_guard = CommandGuard()
        self.audit = audit
        self.approval_policy = approval_policy
        self.approver = approver
        self.tool_timeout_seconds = max(0.1, tool_timeout_seconds)
        self.browser_guard = browser_guard
        self.code_index = CodeIndex(
            self.workspace, self.workspace / ".kairocli" / "rag" / "codebase.db"
        )
        self.snapshot_service: Any = None
        self.current_provider = ""
        self.current_model = ""
        self.web_client = WebClient(workspace=self.workspace)
        self.lsp = LspManager(self.workspace)
        self.shell_sessions = ShellSessionManager(self.workspace)
        self._tools: dict[str, ToolDefinition] = {}
        self._orphaned_tool_tasks: set[asyncio.Task[ToolOutput]] = set()
        # Approval UIs are single-consumer surfaces. Keep the cache check, prompt,
        # argument revalidation, and cache update atomic across parallel tool calls.
        self._approval_lock = asyncio.Lock()
        # Browser policy depends on mutable page/tab state. Keep the guard check,
        # approval, remote operation, and successful state commit in one sequence.
        self._browser_operation_lock = asyncio.Lock()
        self._mcp_server_gates: dict[str, _AsyncServerGate] = {}
        self._workspace_mutation_lock = asyncio.Lock()
        self._active_tool_tasks: dict[asyncio.Task[Any], int] = {}
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None
        self._register_builtin_tools()

    def register(self, definition: ToolDefinition) -> None:
        if self._closing:
            raise RuntimeError("Tool registry is closed")
        self._tools[definition.name] = definition

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def _mcp_server_gate(self, server_name: str) -> _AsyncServerGate:
        gate = self._mcp_server_gates.get(server_name)
        if gate is None:
            gate = _AsyncServerGate()
            self._mcp_server_gates[server_name] = gate
        return gate

    @asynccontextmanager
    async def mcp_server_transition(self, server_name: str) -> AsyncIterator[None]:
        """Wait for in-flight calls and block new calls during one server transition."""
        if self._closing:
            raise RuntimeError("Tool registry is closed")
        gate = self._mcp_server_gate(server_name)
        await gate.acquire_exclusive()
        try:
            yield
        finally:
            await _release_mcp_exclusive_safely(gate)

    @asynccontextmanager
    async def mcp_server_call(self, server_name: str) -> AsyncIterator[None]:
        """Hold one server stable for a non-tool MCP read operation."""
        if self._closing:
            raise RuntimeError("Tool registry is closed")
        gate = self._mcp_server_gate(server_name)
        await _acquire_mcp_shared_with_cancellation(gate, None)
        try:
            yield
        finally:
            await _release_mcp_shared_safely(gate)

    def schemas(self) -> list[dict[str, Any]]:
        if self._closing:
            return []
        return [tool.schema() for tool in self._tools.values()]

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        cancel_event: asyncio.Event | None = None,
    ) -> str:
        return (await self.execute_output(name, arguments, cancel_event)).text

    async def execute_output(
        self,
        name: str,
        arguments: dict[str, Any],
        cancel_event: asyncio.Event | None = None,
    ) -> ToolOutput:
        started_at = time.monotonic()
        log.info("tool_start name=%s", name)
        if self._closing:
            log.info("tool_finish name=%s status=closed duration_ms=0", name)
            return _closed_registry_output()
        current_task = asyncio.current_task()
        if current_task is not None:
            self._active_tool_tasks[current_task] = self._active_tool_tasks.get(current_task, 0) + 1
        context_token = _cancel_event.set(cancel_event)
        effective_arguments = arguments
        browser_operation_locked = False
        browser_operation_started = False
        mcp_gate: _AsyncServerGate | None = None
        mcp_gate_acquired = False
        try:
            raise_if_canceled(cancel_event)
            mcp_server = ApprovalPolicy.mcp_server_name(name)
            if mcp_server is not None:
                mcp_gate = self._mcp_server_gates.get(mcp_server)
                if mcp_gate is None and name not in self._tools:
                    return ToolOutput(json.dumps({"error": f"Unknown tool: {name}"}))
                if mcp_gate is None:
                    mcp_gate = self._mcp_server_gate(mcp_server)
                await _acquire_mcp_shared_with_cancellation(mcp_gate, cancel_event)
                mcp_gate_acquired = True
                raise_if_canceled(cancel_event)
            # A lifecycle transition may replace or remove the definition while
            # this call waits. Resolve it only after shared ownership is granted.
            definition = self._tools.get(name)
            if definition is None:
                return ToolOutput(json.dumps({"error": f"Unknown tool: {name}"}))
            _validate_tool_arguments(definition.parameters, arguments)
            self._preflight(name, arguments)
            if BrowserGuard.is_chrome_tool(name):
                await _acquire_lock_with_cancellation(self._browser_operation_lock, cancel_event)
                browser_operation_locked = True
                raise_if_canceled(cancel_event)
            browser_check = (
                self.browser_guard.check(name, arguments)
                if self.browser_guard is not None
                else None
            )
            if browser_check is not None and browser_check.blocked:
                raise PolicyDenied(browser_check.reason)
            force_browser_approval = bool(
                browser_check and browser_check.requires_per_call_approval
            )
            approval_governed = force_browser_approval or bool(
                self.approval_policy and self.approval_policy.governs(name)
            )
            if approval_governed:
                await _acquire_lock_with_cancellation(self._approval_lock, cancel_event)
                try:
                    raise_if_canceled(cancel_event)
                    # State may have changed while another request owned the prompt.
                    # Revalidate before deciding whether its session approval applies.
                    _validate_tool_arguments(definition.parameters, arguments)
                    self._preflight(name, arguments)
                    browser_check = (
                        self.browser_guard.check(name, arguments)
                        if self.browser_guard is not None
                        else None
                    )
                    if browser_check is not None and browser_check.blocked:
                        raise PolicyDenied(browser_check.reason)
                    force_browser_approval = bool(
                        browser_check and browser_check.requires_per_call_approval
                    )
                    approval_required = force_browser_approval or bool(
                        self.approval_policy and self.approval_policy.needs_approval(name)
                    )
                    if approval_required:
                        approval_arguments = arguments
                        if force_browser_approval and browser_check is not None:
                            approval_arguments = {
                                **arguments,
                                "_kairocli_approval_notice": browser_check.reason,
                            }
                        raw_result = (
                            await wait_with_cancellation(
                                self.approver(name, approval_arguments), cancel_event
                            )
                            if self.approver is not None
                            else ApprovalResult.reject(
                                browser_check.reason
                                if force_browser_approval and browser_check is not None
                                else "No approval handler is configured"
                            )
                        )
                        if raw_result is True:
                            result = ApprovalResult.approve()
                        elif raw_result is False:
                            result = ApprovalResult.reject()
                        elif isinstance(raw_result, ApprovalResult):
                            result = raw_result
                        else:
                            result = ApprovalResult.reject(
                                "Approval handler returned an invalid decision"
                            )
                        if not result.approved:
                            detail = _bounded_tool_error(
                                result.reason
                                or (
                                    "Approval skipped"
                                    if result.decision == ApprovalDecision.SKIPPED
                                    else "Approval denied"
                                )
                            )
                            if self.audit:
                                self.audit.append(name, result.decision.value, arguments, detail)
                            return ToolOutput(
                                json.dumps(
                                    {
                                        "error": detail,
                                        "approval_denied": True,
                                        "approval_decision": result.decision.value,
                                    }
                                )
                            )
                        effective_arguments = result.effective_arguments(arguments)
                        effective_arguments.pop("_kairocli_approval_notice", None)
                        # Modified parameters are untrusted user input and must pass policy again.
                        _validate_tool_arguments(definition.parameters, effective_arguments)
                        self._preflight(name, effective_arguments)
                        if self.browser_guard is not None:
                            modified_check = self.browser_guard.check(name, effective_arguments)
                            if modified_check.blocked:
                                raise PolicyDenied(modified_check.reason)
                        if self.approval_policy is not None and not force_browser_approval:
                            self.approval_policy.remember(name, result)
                finally:
                    self._approval_lock.release()
            if name in SERIALIZED_WORKSPACE_MUTATION_TOOLS:
                await _acquire_lock_with_cancellation(self._workspace_mutation_lock, cancel_event)
                try:
                    raise_if_canceled(cancel_event)
                    # Recheck policy at the serialization boundary so queued writes
                    # cannot rely only on stale pre-approval filesystem state.
                    self._preflight(name, effective_arguments)
                    result = await wait_with_cancellation(
                        definition.handler(effective_arguments), cancel_event
                    )
                finally:
                    self._workspace_mutation_lock.release()
            else:
                if BrowserGuard.is_chrome_tool(name):
                    browser_operation_started = True
                result = await wait_with_cancellation(
                    definition.handler(effective_arguments), cancel_event
                )
            raise_if_canceled(cancel_event)
            if isinstance(result, ToolOutput):
                output = _bound_tool_output(result)
            elif isinstance(result, str):
                output = _bound_tool_output(ToolOutput(result))
            else:
                output = _bounded_json_output(result)
            failed = is_failed_tool_text(output.text)
            if failed:
                output = _redact_failed_tool_output(output)
            if self.browser_guard is not None and not failed:
                self.browser_guard.apply_after_execution(name, effective_arguments, output.text)
            if self.audit and _requires_audit(name):
                detail = output.text if failed else ""
                if browser_check is not None and BrowserGuard.is_chrome_tool(name):
                    browser_detail = _browser_audit_detail(self.browser_guard, browser_check)
                    detail = f"{browser_detail}; {detail}" if detail else browser_detail
                self.audit.append(
                    name,
                    "error" if failed else "allowed",
                    effective_arguments,
                    detail,
                )
            return output
        except ToolArgumentsError as exc:
            raise_if_canceled(cancel_event)
            detail = _bounded_tool_error(exc)
            if self.audit and _requires_audit(name):
                self.audit.append(name, "error", effective_arguments, detail)
            return ToolOutput(
                json.dumps(
                    {
                        "error": detail,
                        "invalid_arguments": True,
                    }
                )
            )
        except PolicyDenied as exc:
            raise_if_canceled(cancel_event)
            detail = _bounded_tool_error(exc)
            if self.audit:
                self.audit.append(name, "denied", effective_arguments, detail)
            return ToolOutput(json.dumps({"error": detail, "policy_denied": True}))
        except AgentCanceled:
            if browser_operation_started and self.browser_guard is not None:
                self.browser_guard.apply_after_cancellation(name)
            if self.audit and _requires_audit(name):
                self.audit.append(
                    name,
                    "canceled",
                    effective_arguments,
                    "Tool execution canceled; side effects may have completed",
                )
            raise
        except asyncio.CancelledError:
            if browser_operation_started and self.browser_guard is not None:
                self.browser_guard.apply_after_cancellation(name)
            raise
        except Exception as exc:
            raise_if_canceled(cancel_event)
            detail = _bounded_tool_error(exc)
            if self.audit and _requires_audit(name):
                self.audit.append(name, "error", effective_arguments, detail)
            return ToolOutput(json.dumps({"error": detail, "type": type(exc).__name__}))
        finally:
            log.info(
                "tool_finish name=%s duration_ms=%d",
                name,
                int((time.monotonic() - started_at) * 1000),
            )
            if browser_operation_locked:
                self._browser_operation_lock.release()
            if mcp_gate is not None and mcp_gate_acquired:
                await _release_mcp_shared_safely(mcp_gate)
            _cancel_event.reset(context_token)
            if current_task is not None:
                depth = self._active_tool_tasks.get(current_task, 0)
                if depth <= 1:
                    self._active_tool_tasks.pop(current_task, None)
                else:
                    self._active_tool_tasks[current_task] = depth - 1

    def _preflight(self, name: str, arguments: dict[str, Any]) -> None:
        path_tools = {
            "read_file",
            "list_dir",
            "lsp_inspect",
            "lsp_workspace_diagnostics",
        }
        if name in path_tools and "path" in arguments:
            self.path_guard.resolve(str(arguments["path"]))
        if name in {"write_file", "create_project"} and "path" in arguments:
            write_path = self.path_guard.resolve_for_write(str(arguments["path"]))
            if name == "create_project" and write_path.exists():
                if not write_path.is_dir() or any(write_path.iterdir()):
                    raise PolicyDenied("create_project target must not exist or must be empty")
        if name in {"execute_command", "shell_exec"}:
            command = str(arguments.get("command", ""))
            self.command_guard.check(command)
            if len(command.encode()) > MAX_SHELL_COMMAND_BYTES:
                raise PolicyDenied(f"{name} command exceeds the 100 KiB safety limit")
        if name == "shell_start" and arguments.get("cwd") is not None:
            path = self.path_guard.resolve(str(arguments["cwd"]), must_exist=True)
            if not path.is_dir():
                raise PolicyDenied("shell_start cwd must be a directory")
        content_size = len(str(arguments.get("content", "")).encode())
        if name == "write_file" and content_size > 5 * 1024 * 1024:
            raise PolicyDenied("write_file content exceeds the 5 MiB safety limit")
        if name == "apply_patch":
            patch = str(arguments.get("patch", ""))
            if len(patch.encode()) > MAX_PATCH_BYTES:
                raise PolicyDenied("apply_patch input exceeds the 1 MiB safety limit")
            _validate_patch(patch, self.path_guard)

    async def execute_many(
        self,
        calls: list[tuple[str, dict[str, Any]]],
        max_concurrency: int = 4,
        cancel_event: asyncio.Event | None = None,
    ) -> list[str]:
        outputs = await self.execute_many_outputs(
            calls, max_concurrency=max_concurrency, cancel_event=cancel_event
        )
        return [output.text for output in outputs]

    async def execute_many_outputs(
        self,
        calls: list[tuple[str, dict[str, Any]]],
        max_concurrency: int = 4,
        cancel_event: asyncio.Event | None = None,
    ) -> list[ToolOutput]:
        semaphore = asyncio.Semaphore(max(1, max_concurrency))

        async def run(name: str, arguments: dict[str, Any]) -> ToolOutput:
            async with semaphore:
                raise_if_canceled(cancel_event)
                started = time.monotonic()
                timeout = self.tool_timeout_seconds
                if name.startswith("mcp__") and "timeout" in arguments:
                    try:
                        raw_mcp_timeout = float(arguments["timeout"])
                        if not raw_mcp_timeout > 0:
                            raise ValueError
                        requested_mcp_timeout = (
                            raw_mcp_timeout / 1_000
                            if name.startswith("mcp__chrome-devtools__") or raw_mcp_timeout > 300
                            else raw_mcp_timeout
                        )
                        timeout = min(
                            timeout,
                            max(0.1, requested_mcp_timeout) + MCP_TIMEOUT_GRACE_SECONDS,
                        )
                    except (TypeError, ValueError):
                        pass
                if name in {"execute_command", "shell_exec"}:
                    try:
                        command_timeout = min(max(float(arguments.get("timeout", 60)), 0.1), 300)
                    except (TypeError, ValueError):
                        command_timeout = 60
                    timeout = max(timeout, command_timeout + 5)
                call_cancel = asyncio.Event()
                try:
                    output = await self._execute_with_hard_timeout(
                        self.execute_output(name, arguments, call_cancel),
                        timeout,
                        call_cancel,
                    )
                except TimeoutError:
                    elapsed_ms = int((time.monotonic() - started) * 1_000)
                    if self.audit and _requires_audit(name):
                        self.audit.append(
                            name,
                            "error",
                            arguments,
                            f"Tool execution exceeded {timeout:g}s timeout",
                        )
                    return ToolOutput(
                        json.dumps(
                            {
                                "error": f"Tool execution exceeded {timeout:g}s timeout",
                                "timed_out": True,
                            }
                        ),
                        timed_out=True,
                        elapsed_ms=elapsed_ms,
                    )
                return replace(output, elapsed_ms=int((time.monotonic() - started) * 1_000))

        try:
            return await wait_with_cancellation(
                asyncio.gather(*(run(name, args) for name, args in calls)), cancel_event
            )
        except AgentCanceled:
            if self.audit:
                for name, arguments in calls:
                    if _requires_audit(name):
                        self.audit.append(
                            name,
                            "canceled",
                            arguments,
                            "Tool batch canceled; side effects may have completed",
                        )
            raise

    async def _execute_with_hard_timeout(
        self,
        awaitable: Coroutine[Any, Any, ToolOutput],
        wall_timeout: float,
        call_cancel: asyncio.Event,
    ) -> ToolOutput:
        task: asyncio.Task[ToolOutput] = asyncio.create_task(awaitable)
        try:
            done, _ = await asyncio.wait({task}, timeout=wall_timeout)
            if task in done:
                return task.result()
            call_cancel.set()
            await self._cancel_tool_task(task)
            raise TimeoutError
        except asyncio.CancelledError:
            call_cancel.set()
            await self._cancel_tool_task(task)
            raise

    async def _cancel_tool_task(self, task: asyncio.Task[ToolOutput]) -> None:
        if task.done():
            await asyncio.gather(task, return_exceptions=True)
            return
        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=TOOL_CANCEL_GRACE_SECONDS)
        if task in done:
            await asyncio.gather(task, return_exceptions=True)
            return
        self._orphaned_tool_tasks.add(task)

        def consume(completed: asyncio.Task[ToolOutput]) -> None:
            self._orphaned_tool_tasks.discard(completed)
            try:
                completed.exception()
            except (asyncio.CancelledError, Exception):
                pass

        task.add_done_callback(consume)

    def _register_builtin_tools(self) -> None:
        specs: list[tuple[str, str, dict[str, Any], ToolHandler]] = [
            (
                "read_file",
                "Read a bounded UTF-8 text range within the workspace. Use offset/limit to "
                "continue when partial is reported; binary files are rejected.",
                _path_schema(),
                self._read_file,
            ),
            (
                "write_file",
                "Write a UTF-8 file within the workspace.",
                _write_schema(),
                self._write_file,
            ),
            (
                "apply_patch",
                "Apply a focused unified Git diff whose headers and path markers agree, inside "
                "the workspace after an atomic check.",
                _patch_schema(),
                self._apply_patch,
            ),
            (
                "lsp_inspect",
                "Read current parser/language-server diagnostics and bounded code-action "
                "summaries for one workspace file. Suggestions are never executed automatically.",
                _lsp_inspect_schema(),
                self._lsp_inspect,
            ),
            (
                "lsp_workspace_diagnostics",
                "Run bounded parser checks and supported LSP workspace diagnostics under "
                "one workspace directory. This tool is read-only.",
                _lsp_workspace_schema(),
                self._lsp_workspace_diagnostics,
            ),
            (
                "list_dir",
                "List a bounded page of directory entries within the workspace.",
                _list_schema(),
                self._list_dir,
            ),
            (
                "glob_files",
                "Find workspace files using a glob pattern.",
                _glob_schema(),
                self._glob_files,
            ),
            (
                "grep_code",
                "Search bounded workspace text using a regex or literal pattern; inspect partial "
                "metadata before assuming results are complete.",
                _grep_schema(),
                self._grep_code,
            ),
            (
                "execute_command",
                "Run a bounded command in the workspace.",
                _command_schema(),
                self._execute_command,
            ),
            (
                "shell_start",
                "Start a bounded persistent shell for commands that must share cwd or "
                "environment. Prefer execute_command for independent commands.",
                _shell_start_schema(),
                self._shell_start,
            ),
            (
                "shell_exec",
                "Execute one bounded command in a persistent shell session. State such as "
                "cd and exported variables persists until the session is stopped.",
                _shell_exec_schema(),
                self._shell_exec,
            ),
            (
                "shell_list",
                "List active persistent shell sessions without creating one.",
                {"type": "object", "properties": {}, "additionalProperties": False},
                self._shell_list,
            ),
            (
                "shell_stop",
                "Stop one persistent shell and its complete process group.",
                _shell_id_schema(),
                self._shell_stop,
            ),
            (
                "create_project",
                "Atomically create a Python, Node, or Java starter project in a new "
                "or empty workspace directory.",
                _create_schema(),
                self._create_project,
            ),
            (
                "search_code",
                "Search indexed code semantically.",
                _query_schema(),
                self._search_code,
            ),
            (
                "web_search",
                "Search the web for current information.",
                _web_search_schema(),
                self._web_search,
            ),
            ("web_fetch", "Fetch and extract text from a URL.", _url_schema(), self._web_fetch),
            (
                "revert_turn",
                "Restore the Nth most recent pre-turn workspace snapshot; "
                "first saves an undo point.",
                {
                    "type": "object",
                    "properties": {"steps": {"type": "integer", "minimum": 1, "maximum": 50}},
                },
                self._revert_turn,
            ),
        ]
        for name, description, parameters, handler in specs:
            self.register(ToolDefinition(name, description, parameters, handler))

    async def _read_file(self, args: dict[str, Any]) -> str:
        path = self.path_guard.resolve(str(args["path"]), must_exist=True)
        if not path.is_file():
            raise ValueError("read_file path is not a regular file")
        start = max(int(args.get("offset", args.get("start_line", 1))), 1)
        requested_limit = args.get("limit")
        if requested_limit is None and int(args.get("end_line", 0)) > 0:
            requested_limit = int(args["end_line"]) - start + 1
        limit = min(max(int(requested_limit or MAX_READ_FILE_LINES), 1), MAX_READ_FILE_LINES)
        max_chars = min(
            max(int(args.get("max_chars", DEFAULT_READ_FILE_CHARS)), 1_000),
            MAX_READ_FILE_CHARS,
        )
        return await asyncio.to_thread(_read_file_range, path, start, limit, max_chars)

    async def _write_file(self, args: dict[str, Any]) -> ToolOutput:
        path = self.path_guard.resolve_for_write(str(args["path"]))
        content = str(args.get("content", ""))
        before, before_error = await asyncio.to_thread(_capture_diff_text, path)
        await asyncio.to_thread(_atomic_write_text, path, content, self.workspace)
        after, after_error = _bounded_diff_value(content)
        diagnostics, diagnostic_error = await self._post_edit_diagnostics(path)
        payload = {
            "path": str(path.relative_to(self.workspace)),
            "bytes": len(content.encode()),
            "diagnostics": [
                {
                    "line": item.line,
                    "column": item.column,
                    "severity": item.severity,
                    "message": item.message,
                    "source": item.source,
                }
                for item in diagnostics
            ],
        }
        if diagnostic_error is not None:
            payload["diagnostics_warning"] = diagnostic_error
        output = _bounded_json_output(payload)
        return replace(
            output,
            diffs=(
                FileDiff(
                    str(path.relative_to(self.workspace)),
                    before,
                    after,
                    before_error or after_error,
                ),
            ),
        )

    async def _lsp_inspect(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self.path_guard.resolve(str(args["path"]), must_exist=True)
        if not path.is_file():
            raise ValueError("lsp_inspect path is not a regular file")
        start_line = max(1, int(args.get("start_line", 1)))
        raw_end = args.get("end_line")
        end_line = max(start_line, int(raw_end)) if raw_end is not None else None
        return await self.lsp.inspect_file_async(path, start_line=start_line, end_line=end_line)

    async def _lsp_workspace_diagnostics(self, args: dict[str, Any]) -> dict[str, Any]:
        raw_path = str(args.get("path", "."))
        root = self.path_guard.resolve(raw_path, must_exist=True)
        if not root.is_dir():
            raise ValueError("lsp_workspace_diagnostics path must be a directory")
        return await self.lsp.workspace_diagnostics_async(
            root,
            max_files=int(args.get("max_files", 500)),
            max_diagnostics=int(args.get("max_diagnostics", 1_000)),
        )

    async def _apply_patch(self, args: dict[str, Any]) -> ToolOutput:
        patch = str(args["patch"])
        paths = _validate_patch(patch, self.path_guard)
        git = shutil.which("git")
        if git is None:
            raise RuntimeError("Git is required to apply unified patches")
        display_paths = paths[:MAX_DISPLAY_DIFF_FILES]
        before = {
            relative: await asyncio.to_thread(
                _capture_diff_text, self.path_guard.resolve_for_write(relative)
            )
            for relative in display_paths
        }
        await self._run_git_apply(git, patch, check=True)
        await self._run_git_apply(git, patch, check=False)
        changed: list[str] = []
        deleted: list[str] = []
        diagnostics: list[dict[str, Any]] = []
        diagnostic_errors: list[dict[str, str]] = []
        for relative in paths:
            path = self.path_guard.resolve_for_write(relative)
            if not path.exists():
                deleted.append(relative)
                continue
            changed.append(relative)
            file_diagnostics, diagnostic_error = await self._post_edit_diagnostics(path)
            for item in file_diagnostics:
                diagnostics.append(
                    {
                        "path": relative,
                        "line": item.line,
                        "column": item.column,
                        "severity": item.severity,
                        "message": item.message,
                        "source": item.source,
                    }
                )
            if diagnostic_error is not None:
                diagnostic_errors.append({"path": relative, "error": diagnostic_error})
        payload = {
            "changed": changed,
            "deleted": deleted,
            "files": len(paths),
            "patch_bytes": len(patch.encode()),
            "diagnostics": diagnostics,
        }
        if diagnostic_errors:
            payload["diagnostics_warnings"] = diagnostic_errors
        diffs: list[FileDiff] = []
        for relative in display_paths:
            path = self.path_guard.resolve_for_write(relative)
            after_text, after_error = await asyncio.to_thread(_capture_diff_text, path)
            before_text, before_error = before[relative]
            diffs.append(
                FileDiff(
                    relative,
                    before_text,
                    after_text,
                    before_error or after_error,
                )
            )
        if len(paths) > len(display_paths):
            diffs.append(
                FileDiff(
                    f"{len(paths) - len(display_paths)} additional file(s)",
                    None,
                    None,
                    f"only the first {MAX_DISPLAY_DIFF_FILES} changed files are displayed",
                )
            )
        output = _bounded_json_output(payload)
        return replace(output, diffs=tuple(diffs))

    async def _post_edit_diagnostics(self, path: Path) -> tuple[list[Any], str | None]:
        try:
            return await self.lsp.diagnose_file_async(path), None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return [], _bounded_tool_error(exc)

    async def _run_git_apply(self, git: str, patch: str, *, check: bool) -> None:
        command = [git, "apply", "--recount", "--whitespace=nowarn"]
        if check:
            command.append("--check")
        command.append("-")
        process_options: dict[str, Any] = {}
        if os.name == "posix":
            process_options["start_new_session"] = True
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=self.workspace,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **process_options,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        stdin = process.stdin
        stdout_reader = asyncio.create_task(
            _read_bounded_stream(process.stdout, MAX_GIT_APPLY_OUTPUT_BYTES)
        )
        stderr_reader = asyncio.create_task(
            _read_bounded_stream(process.stderr, MAX_GIT_APPLY_OUTPUT_BYTES)
        )

        async def feed_patch() -> None:
            try:
                stdin.write(patch.encode())
                await stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                stdin.close()
                try:
                    await stdin.wait_closed()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            await process.wait()

        try:
            await wait_with_cancellation(
                asyncio.wait_for(feed_patch(), 30),
                _cancel_event.get(),
            )
        except (AgentCanceled, asyncio.CancelledError, TimeoutError):
            await _terminate_process_tree(process)
            await asyncio.gather(stdout_reader, stderr_reader, return_exceptions=True)
            raise
        stdout, stderr = await asyncio.gather(stdout_reader, stderr_reader)
        if process.returncode:
            detail = stderr.preview.decode(errors="replace").strip()
            if not detail:
                detail = stdout.preview.decode(errors="replace").strip()
            phase = "check" if check else "apply"
            raise ValueError(f"Patch {phase} failed: {detail or 'unknown Git error'}")

    async def _list_dir(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self.path_guard.resolve(str(args.get("path", ".")), must_exist=True)
        if not path.is_dir():
            raise ValueError("list_dir path is not a directory")
        offset = max(int(args.get("offset", 0)), 0)
        limit = min(max(int(args.get("limit", 200)), 1), 1_000)
        entries = [
            {"name": child.name, "type": "directory" if child.is_dir() else "file"}
            for child in sorted(
                path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())
            )
        ]
        page = entries[offset : offset + limit]
        next_offset = offset + len(page)
        return {
            "entries": page,
            "offset": offset,
            "next_offset": next_offset if next_offset < len(entries) else None,
            "total": len(entries),
            "partial": next_offset < len(entries),
        }

    async def _glob_files(self, args: dict[str, Any]) -> dict[str, Any]:
        pattern = str(args.get("pattern", "**/*"))
        limit = min(max(int(args.get("max_results", 50)), 1), 200)
        root = self.path_guard.resolve(str(args.get("path", ".")), must_exist=True)
        if not root.is_dir():
            raise ValueError("glob_files path is not a directory")
        result = await asyncio.to_thread(_glob_paths, self.workspace, root, pattern, limit + 1)
        partial = len(result) > limit
        return {
            "matches": result[:limit],
            "partial": partial,
            "partial_reason": f"Reached max_results={limit}" if partial else "",
        }

    async def _grep_code(self, args: dict[str, Any]) -> dict[str, Any]:
        pattern = str(args["pattern"])
        if not pattern:
            raise ValueError("grep_code pattern cannot be empty")
        max_results = min(max(int(args.get("max_results", 50)), 1), 200)
        use_regex = bool(args.get("regex", False))
        case_sensitive = bool(args.get("case_sensitive", True))
        context_lines = min(max(int(args.get("context_lines", 0)), 0), 5)
        head_limit = min(max(int(args.get("head_limit", 20)), 1), 50)
        max_chars = min(
            max(int(args.get("max_chars", DEFAULT_GREP_MAX_CHARS)), 1_000),
            MAX_GREP_MAX_CHARS,
        )
        root = self.path_guard.resolve(str(args.get("path", ".")), must_exist=True)
        if not root.is_dir():
            raise ValueError("grep_code path is not a directory")
        rg = shutil.which("rg")
        if rg:
            command = [rg, "--line-number", "--no-heading", "--color", "never"]
            if not use_regex:
                command.append("--fixed-strings")
            if not case_sensitive:
                command.append("--ignore-case")
            if context_lines:
                command.extend(["--context", str(context_lines)])
            command.extend(["--max-count", str(head_limit)])
            file_glob = args.get("glob")
            if file_glob:
                command.extend(["--glob", str(file_glob)])
            command.extend([pattern, str(root)])
            process_options: dict[str, Any] = {}
            if os.name == "posix":
                process_options["start_new_session"] = True
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **process_options,
            )
            assert process.stdout is not None
            assert process.stderr is not None
            stdout_reader = asyncio.create_task(_read_bounded_stream(process.stdout, max_chars * 2))
            stderr_reader = asyncio.create_task(_read_bounded_stream(process.stderr, 10_000))
            timed_out = False
            try:
                await wait_with_cancellation(
                    asyncio.wait_for(process.wait(), 15), _cancel_event.get()
                )
            except TimeoutError:
                timed_out = True
                await _terminate_process_tree(process)
            except (AgentCanceled, asyncio.CancelledError):
                await _terminate_process_tree(process)
                await asyncio.gather(stdout_reader, stderr_reader, return_exceptions=True)
                raise
            stdout, stderr = await asyncio.gather(stdout_reader, stderr_reader)
            if process.returncode not in {0, 1} and not timed_out:
                raise RuntimeError(
                    stderr.preview.decode(errors="replace").strip()
                    or f"ripgrep failed with exit code {process.returncode}"
                )
            raw_lines = stdout.preview.decode(errors="replace").splitlines()
            normalized = [line.replace(str(self.workspace) + os.sep, "", 1) for line in raw_lines]
            matches, char_limited = _take_text_budget(normalized, max_results, max_chars)
            reasons: list[str] = []
            if len(normalized) > max_results:
                reasons.append(f"Reached max_results={max_results}")
            if char_limited or stdout.truncated:
                reasons.append(f"Reached max_chars={max_chars}")
            if timed_out:
                reasons.append("ripgrep timed out after 15s")
            return {
                "matches": matches,
                "partial": bool(reasons),
                "partial_reason": "; ".join(reasons),
                "engine": "rg",
                "suggested_reads": _suggested_reads(matches),
            }
        expression = re.compile(
            pattern if use_regex else re.escape(pattern),
            0 if case_sensitive else re.IGNORECASE,
        )
        fallback_matches: list[str] = []
        files = await asyncio.to_thread(
            _glob_paths, self.workspace, root, str(args.get("glob", "**/*")), 10_001
        )
        fallback_reasons: list[str] = []
        if len(files) > 10_000:
            files = files[:10_000]
            fallback_reasons.append("Scanned first 10000 files")
        skipped_oversized = 0
        reached_results = False
        for file_name in files:
            path = self.workspace / file_name
            content, oversized = _read_grep_text(path)
            if oversized:
                skipped_oversized += 1
            if content is None:
                continue
            for number, line in enumerate(content.splitlines(), 1):
                if expression.search(line):
                    fallback_matches.append(f"{file_name}:{number}:{line}")
                    if len(fallback_matches) >= max_results:
                        reached_results = True
                        break
            if reached_results:
                break
        if reached_results:
            fallback_reasons.append(f"Reached max_results={max_results}")
        if skipped_oversized:
            fallback_reasons.append(
                f"Skipped {skipped_oversized} file(s) above {MAX_GREP_FILE_BYTES} bytes"
            )
        rendered, char_limited = _take_text_budget(fallback_matches, max_results, max_chars)
        if char_limited:
            fallback_reasons.append(f"Reached max_chars={max_chars}")
        return {
            "matches": rendered,
            "partial": bool(fallback_reasons),
            "partial_reason": "; ".join(fallback_reasons),
            "engine": "python",
            "suggested_reads": _suggested_reads(rendered),
        }

    async def _execute_command(self, args: dict[str, Any]) -> dict[str, Any]:
        command = str(args["command"])
        self.command_guard.check(command)
        timeout = min(max(float(args.get("timeout", 60)), 0.1), 300)
        process_options: dict[str, Any] = {}
        if os.name == "posix":
            process_options["start_new_session"] = True
        elif os.name == "nt":
            process_options["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=self.workspace,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **process_options,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        stdout_reader = asyncio.create_task(
            _read_bounded_stream(process.stdout, MAX_COMMAND_OUTPUT_BYTES)
        )
        stderr_reader = asyncio.create_task(
            _read_bounded_stream(process.stderr, MAX_COMMAND_OUTPUT_BYTES)
        )
        timed_out = False
        completion = asyncio.gather(process.wait(), stdout_reader, stderr_reader)
        try:
            await wait_with_cancellation(
                asyncio.wait_for(asyncio.shield(completion), timeout),
                _cancel_event.get(),
            )
        except TimeoutError:
            timed_out = True
            await _terminate_process_tree(process)
        except (AgentCanceled, asyncio.CancelledError):
            await _terminate_process_tree(process)
            await asyncio.gather(completion, return_exceptions=True)
            raise
        except Exception:
            await _terminate_process_tree(process)
            await asyncio.gather(completion, return_exceptions=True)
            raise
        _, stdout, stderr = await completion
        stderr_text = stderr.preview.decode(errors="replace")
        if timed_out:
            timeout_message = f"Command exceeded {timeout:g}s timeout"
            stderr_text = f"{stderr_text}\n{timeout_message}".strip()
        return {
            "exit_code": None if timed_out else process.returncode,
            "timed_out": timed_out,
            "canceled": False,
            "stdout": stdout.preview.decode(errors="replace"),
            "stderr": stderr_text,
            "stdout_bytes": stdout.total_bytes,
            "stderr_bytes": stderr.total_bytes,
            "stdout_truncated": stdout.truncated,
            "stderr_truncated": stderr.truncated,
        }

    async def _shell_start(self, args: dict[str, Any]) -> dict[str, Any]:
        cwd = None
        if args.get("cwd") is not None:
            cwd = self.path_guard.resolve(str(args["cwd"]), must_exist=True)
            if not cwd.is_dir():
                raise ValueError("shell_start cwd must be a directory")
        return await self.shell_sessions.start(cwd)

    async def _shell_exec(self, args: dict[str, Any]) -> dict[str, Any]:
        command = str(args["command"])
        self.command_guard.check(command)
        timeout = min(max(float(args.get("timeout", 60)), 0.1), 300)
        return await self.shell_sessions.execute(
            str(args["session_id"]), command, timeout, _cancel_event.get()
        )

    async def _shell_list(self, _args: dict[str, Any]) -> list[dict[str, Any]]:
        return await self.shell_sessions.list()

    async def _shell_stop(self, args: dict[str, Any]) -> dict[str, Any]:
        session_id = str(args["session_id"])
        return {
            "session_id": session_id,
            "stopped": await self.shell_sessions.stop(session_id),
        }

    async def _create_project(self, args: dict[str, Any]) -> dict[str, Any]:
        root = self.path_guard.resolve_for_write(str(args["path"]))
        kind = str(args.get("kind", "python")).lower()
        return _create_project_atomic(root, kind, self.workspace)

    async def _search_code(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args["query"])
        if not self.code_index.store.paths():
            return {"query": query, "matches": [], "index_required": True}
        return {"query": query, "matches": await self.code_index.search(query)}

    async def _web_search(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args["query"])
        limit = min(max(int(args.get("limit", args.get("top_k", 10))), 1), 10)
        preferred = await self._run_preferred_search_mcp(
            "web_search",
            {"query": query},
            limit,
            ("top_k", "topK", "max_results", "num_results", "limit", "count"),
        )
        if preferred is not None:
            tool_name, text = preferred
            return {
                "query": query,
                "provider": "step-mcp",
                "results": [
                    {
                        "title": "Step Search",
                        "url": "",
                        "snippet": text,
                        "source": tool_name,
                    }
                ],
            }
        results = await self.web_client.search(query, limit)
        return {
            "query": query,
            "provider": results[0].source if results else "",
            "results": [
                {
                    "title": item.title,
                    "url": item.url,
                    "snippet": item.snippet,
                    "source": item.source,
                }
                for item in results
            ],
        }

    async def _web_fetch(self, args: dict[str, Any]) -> dict[str, Any]:
        url = str(args["url"])
        max_chars = int(args.get("max_chars", 100_000))
        preferred = await self._run_preferred_search_mcp(
            "web_fetch",
            {"url": url},
            max_chars,
            ("max_chars", "maxChars", "limit", "max_length", "maxLength"),
        )
        if preferred is not None:
            tool_name, text = preferred
            return {
                "url": url,
                "title": "",
                "content": text,
                "source": tool_name,
                "partial": False,
                "body_empty": not text.strip(),
            }
        return await self.web_client.fetch(url, max_chars)

    async def _run_preferred_search_mcp(
        self,
        raw_name: str,
        arguments: dict[str, Any],
        numeric_value: int,
        numeric_names: tuple[str, ...],
    ) -> tuple[str, str] | None:
        if self.current_provider != "step" or not self.current_model.startswith("step-3.7-flash"):
            return None
        suffix = f"__{raw_name}"
        candidates = sorted(
            (name for name in self._tools if name.startswith("mcp__") and name.endswith(suffix)),
            key=lambda name: ("step" not in name.casefold(), name),
        )
        for name in candidates:
            definition = self._tools[name]
            properties = definition.parameters.get("properties", {})
            forwarded = dict(arguments)
            if isinstance(properties, dict):
                for numeric_name in numeric_names:
                    if numeric_name in properties:
                        forwarded[numeric_name] = numeric_value
                        break
            output = await self.execute_output(name, forwarded, _cancel_event.get())
            text = output.text.strip()
            if text and not is_failed_tool_text(text):
                return name, text
        return None

    async def _revert_turn(self, args: dict[str, Any]) -> dict[str, Any]:
        if self.snapshot_service is None:
            return {"restored": False, "error": "No snapshot service is configured"}
        result = await self.snapshot_service.restore_pre_turn(max(int(args.get("steps", 1)), 1))
        return {
            "restored": result.success,
            "revision": result.revision,
            "message": result.message,
            "restored_files": list(result.restored_files),
            "removed_files": list(result.removed_files),
        }

    async def close(self) -> None:
        self._closing = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._close_once(), name="kairo-tool-registry-shutdown"
            )
        await _await_registry_shutdown(self._close_task)

    async def _close_once(self) -> None:
        active = tuple(self._active_tool_tasks)
        for task in active:
            task.cancel()
        if active:
            await asyncio.wait(active, timeout=TOOL_CANCEL_GRACE_SECONDS)
        orphaned = tuple(self._orphaned_tool_tasks)
        for task in orphaned:
            task.cancel()
        if orphaned:
            await asyncio.wait(orphaned, timeout=TOOL_CANCEL_GRACE_SECONDS)
        closers: list[Coroutine[Any, Any, None]] = [
            self.shell_sessions.close(),
            self.lsp.close(),
        ]
        if self.snapshot_service is not None:
            closers.append(self.snapshot_service.close())
        await asyncio.gather(*closers, return_exceptions=True)
        self._tools.clear()
        self._mcp_server_gates.clear()


def _closed_registry_output() -> ToolOutput:
    return ToolOutput(json.dumps({"error": "Tool registry is closed", "registry_closed": True}))


async def _await_registry_shutdown(task: asyncio.Task[None]) -> None:
    canceled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            canceled = True
    task.result()
    if canceled:
        raise asyncio.CancelledError


def _bound_tool_output(output: ToolOutput) -> ToolOutput:
    if len(output.text) <= MAX_TOOL_OUTPUT_CHARS:
        return output
    head_chars = MAX_TOOL_OUTPUT_CHARS * 2 // 3
    tail_chars = MAX_TOOL_OUTPUT_CHARS - head_chars
    marker = (
        f"\n...[tool output truncated: original_chars={len(output.text)}, "
        f"max_chars={MAX_TOOL_OUTPUT_CHARS}; narrow the query or paginate]...\n"
    )
    preview = output.text[:head_chars] + marker + output.text[-tail_chars:]
    return ToolOutput(
        preview,
        output.image_urls,
        True,
        len(output.text),
        output.timed_out,
        output.elapsed_ms,
        output.diffs,
    )


def _bounded_tool_error(value: Any) -> str:
    redacted = redact_sensitive_text(
        safe_text(value, fallback=f"{type(value).__name__} message unavailable")
    )
    if len(redacted) <= MAX_TOOL_ERROR_CHARS:
        return redacted
    marker = f"...[tool error truncated: original_chars={len(redacted)}]"
    available = max(0, MAX_TOOL_ERROR_CHARS - len(marker) - 1)
    return f"{redacted[:available]}\n{marker}"


def _redact_failed_tool_output(output: ToolOutput) -> ToolOutput:
    original = output.text
    try:
        parsed = decode_strict_json(
            original,
            max_bytes=MAX_TOOL_OUTPUT_CHARS * 4,
            max_depth=32,
            max_nodes=100_000,
        )
    except (RecursionError, TypeError, UnicodeError, ValueError):
        redacted = redact_sensitive_text(original)
        if len(redacted) <= MAX_TOOL_ERROR_CHARS:
            return replace(output, text=redacted)
        bounded = _bounded_tool_error(redacted)
    else:
        sanitized = _redact_failed_json_value(parsed)
        serialized = json.dumps(sanitized, ensure_ascii=False)
        if len(serialized) <= MAX_TOOL_ERROR_CHARS:
            return replace(output, text=serialized)
        if isinstance(sanitized, dict):
            error = sanitized.get("error")
            detail = _bounded_tool_error(error if isinstance(error, str) else serialized)
            envelope: dict[str, Any] = {"error": detail, "tool_error": True}
            for key in (
                "isError",
                "approval_denied",
                "approval_decision",
                "policy_denied",
                "invalid_arguments",
                "type",
            ):
                if key in sanitized:
                    envelope[key] = sanitized[key]
            bounded = json.dumps(envelope, ensure_ascii=False)
        else:
            bounded = _bounded_tool_error(serialized)
    return replace(
        output,
        text=bounded,
        truncated=True,
        original_chars=output.original_chars or len(original),
    )


def _redact_failed_json_value(value: Any, depth: int = 0) -> Any:
    if depth > 32:
        return "<nested-error-data-redacted>"
    if isinstance(value, str):
        return redact_sensitive_text(value)
    if isinstance(value, list):
        return [_redact_failed_json_value(item, depth + 1) for item in value]
    if isinstance(value, dict):
        return {
            redact_sensitive_text(str(key)): _redact_failed_json_value(item, depth + 1)
            for key, item in value.items()
        }
    return value


def _capture_diff_text(path: Path) -> tuple[str | None, str]:
    if not path.exists():
        return None, ""
    if path.is_symlink() or not path.is_file():
        return None, "target is not a regular non-symlink file"
    try:
        max_bytes = MAX_DISPLAY_DIFF_CHARS * 4
        if path.stat().st_size > max_bytes:
            return None, f"file exceeds {MAX_DISPLAY_DIFF_CHARS} display characters"
        with path.open("rb") as stream:
            encoded = stream.read(max_bytes + 1)
        if len(encoded) > max_bytes:
            return None, f"file exceeds {MAX_DISPLAY_DIFF_CHARS} display characters"
        value = encoded.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        return None, f"file is not bounded UTF-8 text ({type(exc).__name__})"
    return _bounded_diff_value(value)


def _read_grep_text(path: Path) -> tuple[str | None, bool]:
    try:
        if path.stat().st_size > MAX_GREP_FILE_BYTES:
            return None, True
        with path.open("rb") as stream:
            encoded = stream.read(MAX_GREP_FILE_BYTES + 1)
    except OSError:
        return None, False
    if len(encoded) > MAX_GREP_FILE_BYTES:
        return None, True
    try:
        return encoded.decode("utf-8"), False
    except UnicodeDecodeError:
        return None, False


def _bounded_diff_value(value: str) -> tuple[str | None, str]:
    if len(value) > MAX_DISPLAY_DIFF_CHARS:
        return None, f"content exceeds {MAX_DISPLAY_DIFF_CHARS} display characters"
    lines = value.count("\n") + 1
    if lines > MAX_DISPLAY_DIFF_LINES:
        return None, f"content exceeds {MAX_DISPLAY_DIFF_LINES} display lines"
    return value, ""


def _bounded_json_output(value: Any) -> ToolOutput:
    serialized = json.dumps(value, ensure_ascii=False)
    if len(serialized) <= MAX_TOOL_OUTPUT_CHARS:
        return ToolOutput(serialized)
    # JSON escaping can expand control-heavy content by up to six times.
    preview_limit = MAX_TOOL_OUTPUT_CHARS // 8
    preview = serialized[: preview_limit * 2 // 3] + serialized[-preview_limit // 3 :]
    envelope = json.dumps(
        {
            "partial": True,
            "partial_reason": "Serialized tool result exceeded the global character budget",
            "original_chars": len(serialized),
            "max_chars": MAX_TOOL_OUTPUT_CHARS,
            "json_preview": preview,
        },
        ensure_ascii=False,
    )
    return ToolOutput(envelope, truncated=True, original_chars=len(serialized))


def _browser_audit_detail(guard: BrowserGuard | None, check: Any) -> str:
    from urllib.parse import urlsplit, urlunsplit

    url = str(check.url or "")
    if url:
        parsed = urlsplit(url)
        url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    mode = guard.session.mode if guard is not None else "unknown"
    return (
        f"browser_mode={mode}; sensitive={bool(check.sensitive)}; "
        f"url={url or '-'}; pattern={check.pattern or '-'}"
    )


def _requires_audit(name: str) -> bool:
    return name in ApprovalPolicy.DANGEROUS_TOOLS or name.startswith("mcp__")
