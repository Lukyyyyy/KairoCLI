from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .agent import Agent, AgentCanceled, AgentOrchestrator, PlanExecuteAgent
from .brand import PRODUCT_NAME
from .browser import handle_browser_command
from .commands import SLASH_HELP
from .config import AppConfig, handle_config_command, handle_model_command
from .diff_display import render_file_diff
from .image import prepare_image_input
from .json_boundary import decode_strict_json
from .mcp import (
    McpServerManager,
    handle_mcp_command,
    refresh_agent_resource_index,
)
from .memory import handle_memory_command, handle_save_command
from .plan import ExecutionPlan
from .policy import ApprovalPolicy, ApprovalResult, read_recent_audit
from .prompts import initialize_project_memory
from .session_display import format_session_list
from .sessions import SessionStore, apply_session, write_session_export
from .skills import SkillRegistry, handle_skill_command
from .snapshot import SnapshotError, turn_snapshot_messages
from .tasks import (
    DurableTask,
    DurableTaskManager,
    DurableTaskStore,
    handle_task_command,
)
from .terminal import sanitize_terminal_text
from .text_safety import safe_text
from .todos import SessionTodoController
from .tool_display import format_tool_calls, format_tool_results
from .trace import redact_sensitive_text, safe_redacted_text
from .user_input import UserInputError, normalize_interactive_submission

MAX_TUI_ERROR_BYTES = 4_000
MAX_TUI_TREE_ENTRIES = 2_000
TUI_WORKER_SHUTDOWN_GRACE_SECONDS = 0.5
_DETACHED_TUI_WAITS: set[asyncio.Task[Any]] = set()


def _safe_tui_error(value: object) -> str:
    return safe_redacted_text(value, MAX_TUI_ERROR_BYTES, "...[error truncated]")


def _finish_detached_tui_wait(task: asyncio.Task[Any]) -> None:
    _DETACHED_TUI_WAITS.discard(task)
    try:
        task.exception()
    except (asyncio.CancelledError, Exception):
        pass


async def _await_tui_shutdown(task: asyncio.Task[None]) -> None:
    canceled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            canceled = True
    task.result()
    if canceled:
        raise asyncio.CancelledError


def _tui_submission_action(value: str, running: bool) -> str:
    normalized = value.strip().casefold()
    if not normalized:
        return "ignore"
    if normalized in {"/cancel", "cancel"}:
        return "cancel" if running else "idle-cancel"
    if running:
        return "busy"
    if normalized in {"/exit", "/quit", "exit", "quit"}:
        return "exit"
    return "run"


def _tui_run_mode(value: str) -> tuple[str, str]:
    stripped = value.strip()
    lowered = stripped.casefold()
    for command, mode in (("/plan", "plan"), ("/team", "team")):
        if lowered == command:
            return mode, ""
        if lowered.startswith(command + " "):
            return mode, stripped[len(command) :].strip()
    return "react", stripped


def _modified_approval_result(value: str) -> ApprovalResult:
    try:
        parsed = decode_strict_json(
            value,
            max_bytes=1024 * 1024,
            max_depth=32,
            max_nodes=100_000,
        )
    except (RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ValueError("Invalid JSON: " + _safe_tui_error(exc)) from exc
    if not isinstance(parsed, dict):
        raise ValueError("Replacement arguments must be a JSON object")
    return ApprovalResult.modify(parsed)


def _redact_tui_transcript_input(value: str) -> str:
    redacted = redact_sensitive_text(value)
    return re.sub(r"(?i)((?:^|\s)(?:--)?api[_-]?key\s+)(\S+)", r"\1***", redacted)


def _tui_config_summary(config: AppConfig | None) -> str:
    if config is None:
        return "Provider configuration is unavailable."
    lines = [f"Active provider: {config.default_provider}", ""]
    for name, provider in config.providers.items():
        marker = "*" if name == config.default_provider else " "
        lines.extend(
            (
                f"{marker} {name}",
                f"  model: {provider.model or 'unset'}",
                f"  base URL: {provider.base_url}",
                f"  temperature: {provider.temperature}",
                f"  max tokens: {provider.max_tokens}",
                f"  context window: {provider.context_window or 'default'}",
                "",
            )
        )
    return sanitize_terminal_text("\n".join(lines).rstrip())


def run_tui(
    agent: Agent,
    session_store: SessionStore | None = None,
    session_id: str | None = None,
    workspace: Path | None = None,
    todo_controller: SessionTodoController | None = None,
    mcp_manager: McpServerManager | None = None,
    task_store: DurableTaskStore | None = None,
    task_manager: DurableTaskManager | None = None,
    skill_registry: SkillRegistry | None = None,
    app_config: AppConfig | None = None,
    startup_notice: str = "",
) -> None:
    try:
        from rich.text import Text
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.containers import Horizontal, Vertical
        from textual.message import Message
        from textual.screen import ModalScreen
        from textual.widgets import (
            Button,
            DirectoryTree,
            Footer,
            Header,
            Input,
            RichLog,
            Static,
            TextArea,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install Kairo CLI dependencies for full-screen TUI") from exc

    class TerminalSafeRichLog(RichLog):
        def write(
            self,
            content: object,
            width: int | None = None,
            expand: bool = False,
            shrink: bool = True,
            scroll_end: bool | None = None,
            animate: bool = False,
        ) -> Any:
            if isinstance(content, str):
                content = Text(sanitize_terminal_text(content))
            return super().write(content, width, expand, shrink, scroll_end, animate)

    class WorkspaceTree(DirectoryTree):
        EXCLUDED = frozenset(
            {".git", ".kairocli", ".venv", "__pycache__", "node_modules", "target"}
        )

        def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
            root = (workspace or agent.tools.workspace).resolve()
            selected: list[Path] = []
            for path in paths:
                if len(selected) >= MAX_TUI_TREE_ENTRIES:
                    break
                if path.name in self.EXCLUDED or path.is_symlink():
                    continue
                try:
                    path.resolve().relative_to(root)
                except (OSError, RuntimeError, ValueError):
                    continue
                selected.append(path)
            return selected

    class Composer(TextArea):
        """Multiline task composer that keeps Enter as the submit shortcut."""

        BINDINGS = [
            Binding("enter", "submit", show=False, priority=True),
            Binding("shift+enter", "newline", show=False, priority=True),
        ]

        class Submitted(Message):
            def __init__(self, composer: Composer, value: str) -> None:
                self.composer = composer
                self.value = value
                super().__init__()

            @property
            def control(self) -> Composer:
                return self.composer

        @property
        def value(self) -> str:
            return self.text

        @value.setter
        def value(self, value: str) -> None:
            self.text = value
            self.move_cursor((value.count("\n"), len(value.rsplit("\n", 1)[-1])))

        def action_submit(self) -> None:
            self.post_message(self.Submitted(self, self.text))

        def action_newline(self) -> None:
            start, end = self.selection
            result = self.replace("\n", start, end, maintain_selection_offset=False)
            self.move_cursor(result.end_location)

    class ConfigScreen(ModalScreen[None]):
        BINDINGS = [("escape", "close", "Close")]
        CSS = """
        #config-panel { width: 85%; max-height: 90%; border: round cyan; padding: 1 2; }
        #config-summary { height: 1fr; overflow-y: auto; }
        #config-actions { height: auto; align-horizontal: right; }
        """

        def compose(self) -> ComposeResult:
            with Vertical(id="config-panel"):
                yield Static("Kairo CLI configuration", classes="screen-title")
                yield Static(_tui_config_summary(app_config), id="config-summary")
                with Horizontal(id="config-actions"):
                    yield Button("Close", id="config-close", variant="primary")

        def action_close(self) -> None:
            self.dismiss(None)

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "config-close":
                self.dismiss(None)

    class ApprovalScreen(ModalScreen[ApprovalResult]):
        CSS = """
        #approval { width: 90%; max-height: 90%; border: round cyan; padding: 1 2; }
        #approval-actions { height: auto; }
        #approval-error { height: auto; color: red; }
        #approval-arguments { margin: 1 0; }
        """

        def __init__(self, tool_name: str, summary: str) -> None:
            super().__init__()
            self.tool_name = tool_name
            self.summary = summary

        def compose(self) -> ComposeResult:
            with Vertical(id="approval"):
                yield Static(f"Allow {self.tool_name}?\n{self.summary[:12_000]}")
                yield Input(value=self.summary[:32_000], id="approval-arguments")
                yield Static("", id="approval-error")
                with Horizontal(id="approval-actions"):
                    yield Button("Allow once", id="approval-once", variant="success")
                    yield Button("Always tool", id="approval-all")
                    if ApprovalPolicy.mcp_server_name(self.tool_name):
                        yield Button("Always server", id="approval-server")
                    yield Button("Run edited", id="approval-modify", variant="warning")
                    yield Button("Skip", id="approval-skip")
                    yield Button("Reject", id="approval-reject", variant="error")

        def on_button_pressed(self, event: Button.Pressed) -> None:
            decisions: dict[str, Callable[[], ApprovalResult]] = {
                "approval-once": ApprovalResult.approve,
                "approval-all": ApprovalResult.approve_all,
                "approval-server": ApprovalResult.approve_all_by_server,
                "approval-skip": ApprovalResult.skip,
                "approval-reject": lambda: ApprovalResult.reject("User rejected the operation"),
            }
            if event.button.id == "approval-modify":
                try:
                    result = _modified_approval_result(
                        self.query_one("#approval-arguments", Input).value
                    )
                except ValueError as exc:
                    self.query_one("#approval-error", Static).update(_safe_tui_error(exc))
                    return
                self.dismiss(result)
                return
            factory = decisions.get(event.button.id or "")
            if factory is not None:
                self.dismiss(factory())

    class PlanReviewScreen(ModalScreen[bool | str]):
        BINDINGS = [("escape", "cancel", "Cancel plan")]
        CSS = """
        #plan-review { width: 90%; max-height: 90%; border: round cyan; padding: 1 2; }
        #plan-actions { height: auto; }
        #plan-feedback { margin: 1 0; }
        #plan-error { height: auto; color: red; }
        """

        def __init__(self, plan: ExecutionPlan) -> None:
            super().__init__()
            self.plan = plan

        def compose(self) -> ComposeResult:
            lines = ["Review execution plan:"]
            for task in self.plan.tasks.values():
                dependencies = ", ".join(sorted(task.dependencies)) or "none"
                lines.append(f"- {task.id}: {task.description} (depends: {dependencies})")
            with Vertical(id="plan-review"):
                yield Static("\n".join(lines)[:24_000])
                yield Input(
                    placeholder="Type feedback to regenerate the plan",
                    id="plan-feedback",
                )
                yield Static("", id="plan-error")
                with Horizontal(id="plan-actions"):
                    yield Button("Execute", id="plan-execute", variant="success")
                    yield Button("Replan", id="plan-replan", variant="warning")
                    yield Button("Cancel", id="plan-cancel", variant="error")

        def action_cancel(self) -> None:
            self.dismiss(False)

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "plan-execute":
                self.dismiss(True)
            elif event.button.id == "plan-cancel":
                self.dismiss(False)
            elif event.button.id == "plan-replan":
                feedback = self.query_one("#plan-feedback", Input).value.strip()
                if not feedback:
                    self.query_one("#plan-error", Static).update(
                        "Enter feedback before replanning."
                    )
                    return
                self.dismiss(feedback)

    class KairoApp(App[None]):
        TITLE = PRODUCT_NAME
        BINDINGS = [
            ("ctrl+b", "toggle_tree", "Files"),
            ("ctrl+comma", "show_config", "Config"),
        ]
        CSS = """
        #body { height: 1fr; }
        #workspace-tree {
            width: 28%;
            min-width: 24;
            max-width: 48;
            border-right: solid #505050;
            padding: 0 1;
        }
        #workspace-tree.hidden { display: none; }
        #main { width: 1fr; }
        #welcome {
            height: auto;
            margin: 1 1 0 1;
            padding: 1 2;
            border: round cyan;
            color: $text-muted;
        }
        #log { height: 1fr; margin: 0 1; }
        #prompt {
            height: 7;
            margin: 0 1;
            border-top: solid #808080;
            border-bottom: solid #808080;
            border-left: none;
            border-right: none;
            background: transparent;
        }
        #status { height: 1; margin: 0 2; color: $text-muted; }
        """

        def __init__(self) -> None:
            super().__init__()
            self.session_id = session_id
            self._turn_worker: Any | None = None
            self._mcp_worker: Any | None = None
            self._shutdown_task: asyncio.Task[None] | None = None
            if todo_controller is not None and session_id is not None:
                todo_controller.attach(session_id)

        def compose(self) -> ComposeResult:
            yield Header()
            with Horizontal(id="body"):
                yield WorkspaceTree(str(workspace or agent.tools.workspace), id="workspace-tree")
                with Vertical(id="main"):
                    yield Static(
                        "Welcome back!\n\n"
                        f"{agent.llm.provider}/{agent.llm.model or 'unset'} · "
                        f"{workspace or agent.tools.workspace}\n"
                        "/help for commands · /exit to quit",
                        id="welcome",
                    )
                    yield TerminalSafeRichLog(id="log", markup=False)
                    yield Composer(
                        placeholder="Message Kairo CLI · @path · @image · @resource",
                        id="prompt",
                        compact=True,
                    )
                    yield Static("ReAct · idle · ? for shortcuts", id="status")
            yield Footer()

        def action_toggle_tree(self) -> None:
            self.query_one("#workspace-tree", WorkspaceTree).toggle_class("hidden")

        def action_show_config(self) -> None:
            self.push_screen(ConfigScreen())

        def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
            root = (workspace or agent.tools.workspace).resolve()
            try:
                relative = event.path.resolve().relative_to(root).as_posix()
            except ValueError:
                return
            mention = (
                f"@<{relative}>"
                if any(character.isspace() for character in relative)
                else f"@{relative}"
            )
            prompt = self.query_one("#prompt", Composer)
            separator = " " if prompt.value and not prompt.value.endswith(" ") else ""
            prompt.value += separator + mention
            prompt.focus()

        @staticmethod
        def write_untrusted(log: Any, value: object, style: str | None = None) -> None:
            safe = sanitize_terminal_text(safe_text(value))
            log.write(Text(safe, style=style) if style else Text(safe))

        def on_mount(self) -> None:
            agent.tools.approval_policy = ApprovalPolicy(True)
            agent.tools.approver = self.approve_tool
            agent.on_tool_calls = self.display_tool_calls
            agent.on_tool_results = self.display_tool_results
            if startup_notice:
                self.query_one("#log", RichLog).write(startup_notice)
            if mcp_manager is not None:
                self._mcp_worker = self.run_worker(
                    self.start_mcp(), name="kairo-tui-mcp-start", exclusive=False
                )
            if task_manager is not None:
                task_manager.terminal_callback = self.on_task_terminal
                task_manager.start()

        def display_tool_calls(self, calls: Any) -> None:
            summary = format_tool_calls(calls)
            if summary:
                self.write_untrusted(self.query_one("#log", RichLog), summary, style="dim")

        def display_tool_results(self, calls: Any, results: Any) -> None:
            summary = format_tool_results(calls, results)
            if summary:
                self.write_untrusted(
                    self.query_one("#log", RichLog),
                    summary,
                    style="yellow" if summary.startswith("⚠") else "green",
                )
            for output in results:
                for diff in output.diffs:
                    self.write_untrusted(self.query_one("#log", RichLog), render_file_diff(diff))

        def on_task_terminal(self, task: DurableTask) -> None:
            log = self.query_one("#log", RichLog)
            detail = task.output or task.error
            preview = detail.replace("\n", " ")[:160] if detail else ""
            message = f"Background task {task.id} {task.status}."
            if preview:
                message += f" {preview}"
            self.write_untrusted(log, message)

        async def start_mcp(self) -> None:
            if mcp_manager is None:
                return
            status = self.query_one("#status", Static)
            log = self.query_one("#log", RichLog)
            status.update("ReAct · starting MCP")
            try:
                await mcp_manager.start_all()
                refresh_agent_resource_index(agent, mcp_manager)
                states = mcp_manager.status()
                ready = sum(value == "RUNNING" for value in states.values())
                failed = sum(value.startswith("ERROR:") for value in states.values())
                if states:
                    log.write(f"MCP ready: {ready}; unavailable/disabled: {failed}.")
            except Exception as exc:
                await mcp_manager.close()
                log.write(f"[yellow]MCP startup warning: {type(exc).__name__}[/yellow]")
            finally:
                if self._turn_worker is None or not self._turn_worker.is_running:
                    status.update("ReAct · idle")

        async def review_plan(self, plan: ExecutionPlan) -> bool | str:
            return await self.push_screen_wait(PlanReviewScreen(plan))

        async def on_unmount(self) -> None:
            if self._shutdown_task is None:
                self._shutdown_task = asyncio.create_task(
                    self._shutdown_once(), name="kairo-tui-shutdown"
                )
            await _await_tui_shutdown(self._shutdown_task)

        async def _shutdown_once(self) -> None:
            agent.cancel()
            waits: list[asyncio.Task[Any]] = []
            for worker in (self._turn_worker, self._mcp_worker):
                if worker is None or not worker.is_running:
                    continue
                worker.cancel()
                waits.append(asyncio.create_task(worker.wait()))
            if waits:
                done, pending = await asyncio.wait(waits, timeout=TUI_WORKER_SHUTDOWN_GRACE_SECONDS)
                if done:
                    await asyncio.gather(*done, return_exceptions=True)
                for task in pending:
                    task.cancel()
                    _DETACHED_TUI_WAITS.add(task)
                    task.add_done_callback(_finish_detached_tui_wait)
            try:
                await self.save_session()
            finally:
                closers = []
                if task_manager is not None:
                    closers.append(task_manager.close())
                if mcp_manager is not None:
                    closers.append(mcp_manager.close())
                closers.append(agent.tools.close())
                await asyncio.gather(*closers, return_exceptions=True)

        async def save_session(self) -> None:
            if session_store is None or self.session_id is None or workspace is None:
                return
            snapshot = session_store.capture_snapshot(agent)
            try:
                await asyncio.to_thread(
                    session_store.save_snapshot,
                    self.session_id,
                    workspace,
                    snapshot,
                )
            except Exception as exc:  # Keep the active TUI usable on persistence failure.
                self.notify("Session warning: " + _safe_tui_error(exc), severity="warning")

        async def handle_session(self, payload: str, log: Any) -> None:
            if session_store is None or self.session_id is None or workspace is None:
                log.write("Session persistence is unavailable.")
                return
            operation, _, argument = (payload or "status").partition(" ")
            if operation in {"", "status", "save"}:
                await self.save_session()
                state = await asyncio.to_thread(session_store.load, self.session_id, workspace)
                if state is not None:
                    log.write(
                        f"{state.meta.id} · {state.meta.message_count} messages · "
                        f"{state.meta.title}"
                    )
                return
            if operation == "list":
                list_argument = argument.strip()
                if list_argument not in {"", "--all"}:
                    log.write("Usage: /session list [--all]")
                    return
                sessions = await asyncio.to_thread(session_store.list, workspace, 100)
                log.write(
                    format_session_list(
                        sessions,
                        self.session_id,
                        show_all=list_argument == "--all",
                        width=88,
                    )
                )
                return
            if operation == "new":
                await self.save_session()
                state = await asyncio.to_thread(
                    session_store.create,
                    workspace,
                    agent.llm.provider,
                    agent.llm.model,
                )
                apply_session(agent, state)
                self.session_id = state.meta.id
                if todo_controller is not None:
                    todo_controller.attach(state.meta.id)
                log.write(f"Started {state.meta.id}.")
                return
            if operation == "resume" and argument.strip():
                state = await asyncio.to_thread(session_store.load, argument.strip(), workspace)
                if state is None:
                    log.write("Session was not found in the current workspace.")
                    return
                await self.save_session()
                apply_session(agent, state)
                self.session_id = state.meta.id
                if todo_controller is not None:
                    todo_controller.attach(state.meta.id)
                log.write(f"Resumed {state.meta.id} with {len(state.messages)} messages.")
                return
            if operation == "delete":
                targets = argument.split()
                if not targets:
                    log.write("Usage: /session delete <SESSION_ID...|--empty>")
                    return
                if targets == ["--empty"]:
                    deleted = await asyncio.to_thread(
                        session_store.delete_empty,
                        workspace,
                        exclude_session_id=self.session_id,
                    )
                    if deleted:
                        noun = "session" if deleted == 1 else "sessions"
                        log.write(f"Deleted {deleted} empty {noun}.")
                    else:
                        log.write("No empty sessions to delete.")
                    return
                if any(target.startswith("--") for target in targets):
                    log.write("Usage: /session delete <SESSION_ID...|--empty>")
                    return
                if self.session_id in targets:
                    log.write("Cannot delete the active session; start a new session first.")
                    return
                try:
                    deleted = await asyncio.to_thread(session_store.delete_many, targets, workspace)
                except ValueError as exc:
                    log.write(_safe_tui_error(exc))
                else:
                    noun = "session" if deleted == 1 else "sessions"
                    log.write(f"Deleted {deleted} {noun}.")
                return
            log.write("Usage: /session [status|save|list|new|resume ID|delete ID...]")

        async def approve_tool(self, tool_name: str, arguments: dict[str, Any]) -> ApprovalResult:
            summary = json.dumps(arguments, ensure_ascii=False, default=str)
            return await self.push_screen_wait(ApprovalScreen(tool_name, summary))

        async def handle_shell(self, payload: str, log: Any) -> None:
            operation, _, argument = (payload or "list").strip().partition(" ")
            if operation in {"", "list"}:
                name, arguments = "shell_list", {}
            elif operation == "start":
                name = "shell_start"
                arguments = {"cwd": argument.strip()} if argument.strip() else {}
            elif operation == "exec":
                identifier, separator, command = argument.strip().partition(" ")
                if not separator or not command.strip():
                    log.write("Usage: /shell exec <SHELL_ID> <command>")
                    return
                name = "shell_exec"
                arguments = {"session_id": identifier, "command": command.strip()}
            elif operation == "stop" and argument.strip():
                name = "shell_stop"
                arguments = {"session_id": argument.strip()}
            else:
                log.write("Usage: /shell [list|start [cwd]|exec ID COMMAND|stop ID]")
                return
            self.write_untrusted(log, await agent.tools.execute(name, arguments))

        async def on_composer_submitted(self, event: Composer.Submitted) -> None:
            log = self.query_one("#log", RichLog)
            event.composer.value = ""
            try:
                prompt = normalize_interactive_submission(event.value)
            except UserInputError as exc:
                self.write_untrusted(log, "Input error: " + _safe_tui_error(exc), style="red")
                return
            running = self._turn_worker is not None and self._turn_worker.is_running
            action = _tui_submission_action(prompt, running)
            if action == "ignore":
                return
            self.write_untrusted(log, f"> {_redact_tui_transcript_input(prompt)}", style="dim")
            if action == "cancel":
                agent.cancel()
                self.query_one("#status", Static).update("ReAct · canceling")
                log.write("Cancellation requested for the active task.")
                return
            if action == "idle-cancel":
                log.write("No task is currently running.")
                return
            if action == "busy":
                log.write("A task is still running. Wait for it or enter /cancel.")
                return
            if action == "exit":
                self.exit()
                return
            self._turn_worker = self.run_worker(
                self.process_prompt(prompt), name="kairo-tui-turn", exclusive=False
            )

        async def process_prompt(self, prompt: str) -> None:
            log = self.query_one("#log", RichLog)
            status = self.query_one("#status", Static)
            snapshots = agent.tools.snapshot_service
            command, _, payload = prompt.strip().partition(" ")
            command = command.casefold()
            if command == "/clear":
                agent.clear()
                if agent.tools.approval_policy is not None:
                    agent.tools.approval_policy.clear_session_approvals()
                log.clear()
                log.write("Conversation history and session approvals cleared.")
                await self.save_session()
                return
            if command in {"/context", "/ctx"}:
                log.write(agent.context_status())
                return
            if command == "/compact":
                try:
                    changed = await agent.compact()
                    log.write("Context compacted." if changed else "No compaction was needed.")
                except AgentCanceled:
                    log.write("Compaction canceled.")
                await self.save_session()
                return
            if command == "/init":
                argument = payload.strip().casefold()
                if argument not in {"", "--force"}:
                    log.write("Usage: /init [--force]")
                    return
                try:
                    target = initialize_project_memory(
                        workspace or agent.tools.workspace, argument == "--force"
                    )
                    log.write(f"Created {target.name}")
                except FileExistsError as exc:
                    self.write_untrusted(log, _safe_tui_error(exc))
                return
            if command == "/hitl":
                operation = payload.strip().casefold()
                policy = agent.tools.approval_policy
                if policy is None:
                    log.write("HITL approvals are unavailable.")
                    return
                if operation == "on":
                    policy.enabled = True
                elif operation == "off":
                    policy.enabled = False
                    policy.clear_session_approvals()
                elif operation:
                    log.write("Usage: /hitl [on|off]")
                    return
                log.write(f"HITL approvals: {'on' if policy.enabled else 'off'}")
                return
            if command in {"/help", "?"}:
                log.write(SLASH_HELP)
                return
            if command == "/session":
                await self.handle_session(payload.strip(), log)
                return
            if command == "/todo":
                if todo_controller is None:
                    log.write("Todo persistence is unavailable.")
                else:
                    try:
                        log.write(await todo_controller.command(payload.strip()))
                    except ValueError as exc:
                        self.write_untrusted(log, "Todo error: " + _safe_tui_error(exc))
                return
            if command == "/shell":
                await self.handle_shell(payload.strip(), log)
                return
            if command == "/trace":
                logger = agent.trace_logger
                operation = payload.strip().casefold() or "status"
                if logger is None:
                    log.write("Model tracing is unavailable for this Agent.")
                elif operation == "on":
                    logger.enabled = True
                elif operation == "off":
                    logger.enabled = False
                    logger.include_reasoning = False
                elif operation == "reasoning on":
                    logger.enabled = True
                    logger.include_reasoning = True
                elif operation == "reasoning off":
                    logger.include_reasoning = False
                elif operation != "status":
                    log.write("Usage: /trace [status|on|off|reasoning on|reasoning off]")
                    return
                if logger is not None:
                    log.write(
                        f"Model trace: {'on' if logger.enabled else 'off'} · reasoning: "
                        f"{'on' if logger.include_reasoning else 'off'} · {logger.directory}"
                    )
                return
            if command == "/mcp":
                if mcp_manager is None:
                    log.write("MCP management is unavailable.")
                    return
                status.update("MCP · working")
                try:
                    log.write(
                        await handle_mcp_command(
                            payload.strip(),
                            mcp_manager,
                            agent,
                            agent.tools.approval_policy,
                        )
                    )
                finally:
                    status.update("ReAct · idle")
                return
            if command == "/browser":
                guard = agent.tools.browser_guard
                if guard is None:
                    log.write("Browser session management is unavailable.")
                else:
                    status.update("Browser · working")
                    try:
                        log.write(
                            await handle_browser_command(
                                payload.strip(),
                                guard.session,
                                mcp_manager=mcp_manager,
                                approval_policy=agent.tools.approval_policy,
                                tools=agent.tools,
                            )
                        )
                    finally:
                        status.update("ReAct · idle")
                return
            if command == "/index":
                try:
                    target = (
                        agent.tools.path_guard.resolve(payload.strip(), must_exist=True)
                        if payload.strip()
                        else agent.tools.workspace
                    )
                    if not target.is_dir():
                        log.write("Index target must be a workspace directory.")
                        return

                    def progress(position: int, total: int, path: str) -> None:
                        status.update(f"Index · {position}/{total} · {path[:60]}")

                    result = await agent.tools.code_index.index(target, progress)
                    log.write(f"Indexed {result['files']} files into {result['chunks']} chunks.")
                except (OSError, ValueError) as exc:
                    self.write_untrusted(log, "Index error: " + _safe_tui_error(exc))
                finally:
                    status.update("ReAct · idle")
                return
            if command == "/search":
                query = payload.strip()
                if not query:
                    log.write("Usage: /search <query>")
                    return
                try:
                    matches = await agent.tools.code_index.search(query)
                except ValueError as exc:
                    self.write_untrusted(log, "Search error: " + _safe_tui_error(exc))
                    return
                if not matches:
                    log.write("No indexed code matches.")
                for match in matches:
                    log.write(
                        f"{match['path']}:{match['start_line']}-{match['end_line']} "
                        f"score={match['score']}\n{match['content']}"
                    )
                return
            if command == "/graph":
                symbol = payload.strip()
                if not symbol:
                    log.write("Usage: /graph <symbol>")
                    return
                relations = agent.tools.code_index.graph(symbol)
                if not relations:
                    log.write("No indexed symbol relations.")
                for item in relations:
                    log.write(f"{item['kind']} {item['path']}:{item['line']} {item['text']}")
                return
            if command == "/task":
                if task_store is None or task_manager is None:
                    log.write("Background task management is unavailable.")
                else:
                    log.write(handle_task_command(payload.strip(), task_store, task_manager))
                return
            if command in {"/memory", "/mem"}:
                if agent.memory_store is None:
                    log.write("Long-term memory is unavailable.")
                else:
                    log.write(handle_memory_command(payload.strip(), agent.memory_store))
                return
            if command == "/save":
                if agent.memory_store is None:
                    log.write("Long-term memory is unavailable.")
                else:
                    try:
                        log.write(handle_save_command(payload.strip(), agent.memory_store))
                    except ValueError as exc:
                        self.write_untrusted(log, "Memory error: " + _safe_tui_error(exc))
                return
            if command == "/skill":
                if skill_registry is None:
                    log.write("Skill management is unavailable.")
                else:
                    log.write(
                        await asyncio.to_thread(
                            handle_skill_command,
                            payload.strip(),
                            skill_registry,
                            agent,
                        )
                    )
                return
            if command == "/policy":
                policy = agent.tools.approval_policy
                paths = agent.memory_store.paths if agent.memory_store is not None else None
                log.write(
                    f"Workspace fence: {agent.tools.workspace}\n"
                    f"HITL: {'on' if policy is not None and policy.enabled else 'off'}\n"
                    f"Audit: {paths.audit_dir if paths is not None else 'unavailable'}"
                )
                return
            if command == "/audit":
                paths = agent.memory_store.paths if agent.memory_store is not None else None
                if paths is None:
                    log.write("Audit log is unavailable.")
                else:
                    limit = int(payload.strip()) if payload.strip().isdigit() else 10
                    log.write(read_recent_audit(paths, limit))
                return
            if command == "/export":
                paths = agent.memory_store.paths if agent.memory_store is not None else None
                if paths is None:
                    log.write("Session export is unavailable.")
                else:
                    try:
                        target = write_session_export(paths, agent.export_markdown())
                    except (OSError, ValueError) as exc:
                        self.write_untrusted(
                            log,
                            "Session export failed: " + _safe_tui_error(exc),
                        )
                    else:
                        log.write(f"Exported to {target}")
                return
            if command == "/model":
                paths = agent.memory_store.paths if agent.memory_store is not None else None
                if app_config is None or paths is None:
                    log.write("Provider configuration is unavailable.")
                else:
                    log.write(handle_model_command(payload.strip(), app_config, paths))
                return
            if command == "/config":
                paths = agent.memory_store.paths if agent.memory_store is not None else None
                if app_config is None or paths is None:
                    log.write("Provider configuration is unavailable.")
                elif not payload.strip():
                    self.push_screen(ConfigScreen())
                else:
                    log.write(handle_config_command(payload.strip(), app_config, paths))
                return
            if command == "/snapshot" and snapshots is not None:
                try:
                    if payload == "status":
                        log.write(await snapshots.status())
                    elif payload == "clean":
                        cleaned = await snapshots.clean()
                        log.write(
                            "Snapshot history cleaned." if cleaned else "No snapshot history."
                        )
                    else:
                        for item in await snapshots.list():
                            log.write(
                                f"{item.short_revision} {item.phase} {item.turn_id} "
                                f"{item.created_at}"
                            )
                except SnapshotError as exc:
                    self.write_untrusted(
                        log, "Snapshot warning: " + _safe_tui_error(exc), style="yellow"
                    )
                return
            if command == "/restore" and snapshots is not None:
                if not payload.isdigit():
                    log.write("Usage: /restore <N>")
                    return
                try:
                    result = await snapshots.restore_pre_turn(int(payload))
                    log.write(result.message)
                except SnapshotError as exc:
                    self.write_untrusted(
                        log, "Snapshot warning: " + _safe_tui_error(exc), style="yellow"
                    )
                return
            if command.startswith("/") and command not in {"/plan", "/team"}:
                log.write(f"Unknown or unavailable TUI command: {command}. Use /help.")
                return
            mode, task = _tui_run_mode(prompt)
            if mode != "react" and not task:
                log.write(f"Usage: /{mode} <task>")
                return
            mode_label = mode.capitalize()
            status.update(f"{mode_label} · thinking")
            pre_message, post_message = turn_snapshot_messages(mode, task)
            captured = False
            try:
                prepared = await prepare_image_input(
                    task,
                    workspace or agent.tools.workspace,
                    agent.image_cache_dir,
                )
                from .cli import expand_local_mentions

                expanded = expand_local_mentions(prepared.text, workspace or agent.tools.workspace)
                if mcp_manager is not None:
                    expanded = await mcp_manager.expand_resource_mentions(expanded)
                image_urls = list(prepared.image_urls)
                if snapshots is not None and snapshots.config.enabled:
                    try:
                        await snapshots.capture(pre_message)
                        captured = True
                    except SnapshotError as exc:
                        self.write_untrusted(
                            log,
                            "Snapshot warning: " + _safe_tui_error(exc),
                            style="yellow",
                        )
                if mode == "plan":
                    answer = await PlanExecuteAgent(agent, review_handler=self.review_plan).run(
                        expanded, image_urls
                    )
                elif mode == "team":
                    answer = await AgentOrchestrator(agent).run(expanded, image_urls)
                else:
                    answer = await agent.run(expanded, image_urls)
                self.write_untrusted(log, answer)
            except AgentCanceled:
                log.write("Task canceled.")
            except Exception as exc:
                self.write_untrusted(log, "Error: " + _safe_tui_error(exc), style="red")
            finally:
                if captured:
                    snapshots.capture_background(
                        post_message,
                        lambda error: self.write_untrusted(
                            log,
                            "Snapshot warning: " + _safe_tui_error(error),
                            style="yellow",
                        ),
                    )
                status.update(
                    f"{mode_label} · idle · in/out "
                    f"{agent.total_input_tokens}/{agent.total_output_tokens}"
                )
                await self.save_session()

    KairoApp().run()
