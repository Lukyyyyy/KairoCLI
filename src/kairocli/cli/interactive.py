from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import re
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from ..agent import Agent, AgentCanceled, AgentOrchestrator, PlanExecuteAgent
from ..brand import PRODUCT_NAME, VERSION
from ..browser import (
    BrowserSession,
    handle_browser_command,
)
from ..commands import (
    SLASH_COMMAND_DESCRIPTIONS,
    SLASH_HELP,
    SLASH_SUBCOMMAND_DESCRIPTIONS,
    CommandType,
    ParsedCommand,
    parse_command,
)
from ..config import (
    AppConfig,
    handle_config_command,
    handle_model_command,
    normalize_provider_name,
)
from ..image import prepare_image_input
from ..json_boundary import decode_strict_json
from ..llm import LlmError
from ..mcp import (
    McpServerManager,
    handle_mcp_command,
)
from ..memory import MemoryStore, handle_memory_command, handle_save_command
from ..paths import KairoPaths
from ..plan import PlanReviewDecisionType, parse_plan_review_input
from ..policy import ApprovalPolicy, ApprovalResult, read_recent_audit
from ..prompts import initialize_project_memory
from ..rendering.diff_display import render_file_diff
from ..rendering.session_display import format_session_list
from ..rendering.terminal import TerminalStreamSanitizer, sanitize_terminal_text
from ..rendering.terminal_markdown import (
    ColorSystemName,
    RenderedTerminalText,
    TerminalMarkdownRenderer,
    render_code_search_match,
)
from ..rendering.thought_display import ThoughtDisplay
from ..rendering.tool_display import format_tool_calls, format_tool_results
from ..sessions import SessionStore, apply_session, write_session_export
from ..skills import SkillRegistry, handle_skill_command
from ..snapshot import SnapshotError, SnapshotService, turn_snapshot_messages
from ..tasks import DurableTaskManager, DurableTaskStore, handle_task_command
from ..text_safety import safe_text
from ..todos import SessionTodoController
from ..tools import ToolRegistry
from ..trace import safe_redacted_text
from ..user_input import (
    UserInputError,
    normalize_interactive_submission,
)
from .bootstrap import _inject_mcp_resource_index, _register_browser_agent_tools, make_agent
from .completion import (
    _completion_word,
    _highlight_input_line,
    _local_path_completion_candidates,
    _slash_completion_candidates,
)
from .history import (
    _append_input_history,
    _clear_input_history,
    _is_sensitive_history_input,
    _load_input_history,
    _prepare_input_history,
)
from .wechat_ui import QR_SCAN_PROMPT, workspace_prompt

log = logging.getLogger(__name__)
MAX_INTERACTIVE_ERROR_BYTES = 4_000


class _WechatSetupCanceled(Exception):
    """Raised when the user leaves the WeChat setup prompt."""


_WELCOME_PIXEL_WORDMARK = (
    "█▄▀ ▄▀▄ ▀█▀ █▀▄ ▄▀▄",
    "█▄  █▀█  █  █▀▄ █ █",
    "▀ ▀ ▀ ▀ ▀▀▀ ▀ ▀  ▀ ",
)
_WELCOME_ACCENT_STYLE = "bold #4a7fa7"
_WELCOME_SECTION_DIVIDER = "\x00section-divider\x00"


def _working_status_text(elapsed: float) -> str:
    return f"● Working ({max(0, int(elapsed))}s · esc to interrupt)"


def _mode_status_line(mode: str, status: str) -> str:
    label = {"plan": "Plan", "team": "Team"}.get(mode, "ReAct")
    return f"{label} · {status}"


def _plan_summary(plan: Any) -> str:
    tasks = list(plan.tasks.values())
    positions = {task.id: index for index, task in enumerate(tasks, start=1)}
    noun = "step" if len(tasks) == 1 else "steps"
    lines = [f"Plan · {len(tasks)} {noun}"]
    for index, task in enumerate(tasks, start=1):
        lines.append(f"  {index}  {task.description}")
        dependencies = [positions.get(task_id, task_id) for task_id in sorted(task.dependencies)]
        if dependencies:
            lines.append("     ↳ after " + ", ".join(str(value) for value in dependencies))
    return "\n".join(lines)


def _approval_arguments(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, ensure_ascii=False, indent=2, default=str)[:32_000]


def _print_approval_request(
    console: Any,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    has_server_scope: bool,
) -> None:
    """Render a stable approval card before temporarily taking terminal input."""

    summary = sanitize_terminal_text(_approval_arguments(arguments))
    if _has_rich(console):
        from rich.console import Group
        from rich.panel import Panel
        from rich.text import Text

        choices = Text()
        for key, label in (
            ("y", "allow once"),
            ("a", "always this tool"),
            *(((("v", "always this server"),)) if has_server_scope else ()),
            ("s", "skip"),
            ("m", "modify"),
            ("n", "reject"),
        ):
            if choices:
                choices.append("   ")
            choices.append(key, style="bold #4a7fa7")
            choices.append(f"  {label}", style="dim")
        body = Group(
            Text.assemble(("Tool  ", "dim"), (tool_name, "bold")),
            Text("Arguments", style="dim"),
            Text(summary, style="#888888"),
            Text(""),
            choices,
        )
        console.print(
            Panel(
                body,
                title=" Permission required ",
                title_align="left",
                border_style="#d4a72c",
                padding=(0, 1),
                expand=False,
            )
        )
        return
    choice_text = "y allow once · a always this tool"
    if has_server_scope:
        choice_text += " · v always this server"
    choice_text += " · s skip · m modify · n reject"
    console.print(
        f"Permission required\n  Tool: {tool_name}\n  Arguments:\n{summary}\n  {choice_text}"
    )


def _session_startup_notice(message_count: int) -> str | None:
    if message_count <= 0:
        return None
    noun = "message" if message_count == 1 else "messages"
    return f"Restored {message_count} {noun}"


def _resume_session_hint(session_id: str) -> str:
    return f"Resume this session with:\nkairocli --resume {session_id}"


class _TurnLifecycleState:
    """Own timing for one agent turn independently from transient renderers."""

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.monotonic
        self._started_at: float | None = None
        self._elapsed_before_start = 0.0

    @property
    def running(self) -> bool:
        return self._started_at is not None

    def start(self) -> None:
        self._elapsed_before_start = 0.0
        self._started_at = self._clock()

    def finish(self) -> None:
        self._started_at = None
        self._elapsed_before_start = 0.0

    def pause(self) -> None:
        if self._started_at is None:
            return
        self._elapsed_before_start += max(0.0, self._clock() - self._started_at)
        self._started_at = None

    def resume(self) -> None:
        if self._started_at is None:
            self._started_at = self._clock()

    def elapsed(self) -> float:
        if self._started_at is None:
            return self._elapsed_before_start
        return self._elapsed_before_start + max(0.0, self._clock() - self._started_at)


class _WorkingIndicator:
    """Transient renderer for the current agent turn's activity state."""

    def __init__(self, console: Any, turn_lifecycle: _TurnLifecycleState) -> None:
        self.console = console
        self.turn_lifecycle = turn_lifecycle
        self.live: Any = None
        self.refresh_task: asyncio.Task[None] | None = None
        self.stopped = False

    def start(self) -> None:
        if _has_rich(self.console):
            from rich.live import Live

            self.live = Live(
                self._render(),
                console=self.console,
                transient=True,
                refresh_per_second=8,
                redirect_stdout=False,
                redirect_stderr=False,
            )
            self.live.start(refresh=True)
            self.refresh_task = asyncio.create_task(
                self._refresh(), name="kairocli-working-indicator"
            )
            return
        _print_status(self.console, "\n" + self._text() + "\n", "dim")

    def stop(self) -> None:
        if self.stopped:
            return
        self.stopped = True
        if self.refresh_task is not None:
            self.refresh_task.cancel()
        if self.live is not None:
            self.live.update("")
            self.live.stop()

    def _text(self) -> str:
        return _working_status_text(self.turn_lifecycle.elapsed())

    def _render(self) -> Any:
        from rich.text import Text

        elapsed = int(self.turn_lifecycle.elapsed())
        return Text.assemble(
            "\n",
            ("● ", "#111111"),
            ("Working", "bold"),
            (f" ({elapsed}s · esc to interrupt)", "dim"),
        )

    async def _refresh(self) -> None:
        try:
            while not self.stopped:
                await asyncio.sleep(0.1)
                if self.live is not None:
                    self.live.update(self._render(), refresh=True)
        except asyncio.CancelledError:
            return


class _IndexProgress:
    """Render index progress in one transient terminal row."""

    def __init__(self, console: Any) -> None:
        self.output = console
        self.console = getattr(console, "console", console)
        self.progress: Any = None
        self.task_id: Any = None

    def start(self) -> None:
        if not _has_rich(self.console):
            self.output.print("Indexing workspace...")
            return
        from rich.padding import Padding
        from rich.progress import BarColumn, Progress, TextColumn
        from rich.table import Column

        class SpacedProgress(Progress):
            def get_renderable(self) -> Any:
                return Padding(super().get_renderable(), (1, 0, 0, 0))

        self.progress = SpacedProgress(
            TextColumn("● Indexing", style="bold"),
            BarColumn(
                bar_width=24,
                complete_style="green",
                finished_style="green",
                pulse_style="green",
            ),
            TextColumn(
                "{task.fields[state]}",
                markup=False,
                table_column=Column(ratio=1, no_wrap=True, overflow="ellipsis"),
            ),
            console=self.console,
            transient=True,
            expand=True,
        )
        self.progress.start()
        self.task_id = self.progress.add_task("", total=None, state="Scanning workspace...")

    def update(self, position: int, total: int, path: str) -> None:
        if self.progress is None:
            return
        display_path = sanitize_terminal_text(path).replace("\n", " ")
        self.progress.update(
            self.task_id,
            completed=position,
            total=total,
            state=f"{position}/{total} · {display_path}",
        )

    def stop(self) -> None:
        if self.progress is not None:
            self.progress.stop()


class _IndexDisplay:
    """Expandable summary for the latest code index."""

    def __init__(self) -> None:
        self.expanded = False
        self.finished = False
        self.files = 0
        self.chunks = 0
        self.tree = ""
        self.offset = 0

    def finish(self, files: int, chunks: int, paths: list[str]) -> None:
        self.files = files
        self.chunks = chunks
        self.tree = _format_file_tree(paths)
        self.offset = 0
        self.expanded = False
        self.finished = True

    def toggle(self) -> bool:
        if not self.finished:
            return False
        self.expanded = not self.expanded
        return True

    def dismiss(self) -> None:
        self.expanded = False
        self.finished = False
        self.offset = 0

    def scroll(self, amount: int, page_size: int) -> bool:
        lines = self.tree.splitlines()
        maximum = max(0, len(lines) - page_size)
        target = min(maximum, max(0, self.offset + amount))
        changed = target != self.offset
        self.offset = target
        return changed

    def visible_tree(self, page_size: int) -> str:
        lines = self.tree.splitlines()
        offset = min(self.offset, max(0, len(lines) - page_size))
        visible = lines[offset : offset + page_size]
        if len(lines) > page_size:
            end = min(offset + page_size, len(lines))
            visible.append(
                f"… lines {offset + 1}-{end} of {len(lines)} "
                "· ↑/↓, PageUp/PageDown (Fn+↑/↓) to browse"
            )
        return "\n".join(visible)

    def summary(self) -> str:
        action = "collapse" if self.expanded else "expand"
        return (
            f"Indexed {self.files} files into {self.chunks} chunks "
            f"(ctrl+o to {action})"
        )


def _format_file_tree(paths: list[str]) -> str:
    tree: dict[str, Any] = {}
    for path in sorted(set(paths)):
        node = tree
        for part in sanitize_terminal_text(path).replace("\n", " ").split("/"):
            if part:
                node = node.setdefault(part, {})

    lines: list[str] = []

    def render(node: dict[str, Any], prefix: str = "") -> None:
        entries = sorted(
            node.items(), key=lambda item: (not bool(item[1]), item[0].casefold())
        )
        for position, (name, children) in enumerate(entries):
            last = position == len(entries) - 1
            lines.append(f"{prefix}{'└── ' if last else '├── '}{name}{'/' if children else ''}")
            if children:
                render(children, prefix + ("    " if last else "│   "))

    render(tree)
    return "\n".join(lines)


class _StreamingAnswerDisplay:
    """Render one or more model answer blocks as streaming terminal output."""

    def __init__(
        self,
        console: Any,
        renderer: str,
        columns: int,
        thought_summary: Callable[[], str] | None = None,
    ) -> None:
        self.console = console
        self.renderer = renderer
        self.columns = columns
        self.thought_summary = thought_summary
        self.sanitizer = TerminalStreamSanitizer()
        self.markdown_renderer: TerminalMarkdownRenderer | None = None
        self.block_open = False
        self.has_streamed_content = False
        self.content_committed = False
        self.thought_header_written = False
        self.block_has_thought_header = False
        self._plain_line_start = False

    def start_turn(self) -> None:
        self.has_streamed_content = False
        self.content_committed = False
        self.thought_header_written = False
        self.block_has_thought_header = False

    def append(self, delta: str) -> None:
        if not delta:
            return
        self.has_streamed_content = True
        self.content_committed = True
        if not self.block_open:
            self.block_has_thought_header = not self.thought_header_written
            self.thought_header_written = True
        if not self.block_open:
            if self.block_has_thought_header and self.thought_summary is not None:
                _write_stream("\n")
                _print_status(
                    self.console,
                    "  " + self.thought_summary(),
                    "#888888",
                )
            _write_stream("\n")
            _print_answer_prefix(self.console)
            self.block_open = True
        if self.renderer == "inline":
            if self.markdown_renderer is None:
                self.markdown_renderer = TerminalMarkdownRenderer(
                    self.columns,
                    continuation_indent="  ",
                    color_system=_console_color_system(self.console),
                )
            safe = self.markdown_renderer.append(delta)
        else:
            safe = RenderedTerminalText(self._indent_plain(self.sanitizer.feed(delta)))
        if safe:
            _write_rendered_stream(safe)

    def finish_block(self) -> None:
        if not self.block_open:
            return
        if self.markdown_renderer is not None:
            tail = self.markdown_renderer.finish()
        else:
            tail = RenderedTerminalText(self._indent_plain(self.sanitizer.finish()))
        if tail:
            _write_rendered_stream(tail)
        _end_answer_block()
        self.sanitizer = TerminalStreamSanitizer()
        self.markdown_renderer = None
        self.block_open = False
        self.block_has_thought_header = False
        self._plain_line_start = False

    def print_complete(self, answer: str) -> None:
        if not answer:
            return
        rendered = _render_interactive_answer(
            answer,
            self.renderer,
            self.columns,
            color_system=_console_color_system(self.console),
        )
        _write_stream("\n")
        _print_answer_prefix(self.console)
        _write_rendered_stream(RenderedTerminalText(rendered + "\n"))
        _end_answer_block()

    def _indent_plain(self, value: str) -> str:
        output: list[str] = []
        for character in value:
            if self._plain_line_start and character != "\n":
                output.append("  ")
                self._plain_line_start = False
            output.append(character)
            if character == "\n":
                self._plain_line_start = True
        return "".join(output)


class _EscapeInterrupt:
    """Temporarily make a TTY's Escape key cancel the active agent turn."""

    def __init__(self, callback: Any) -> None:
        self.callback = callback
        self.input_backend: Any = None
        self.raw_context: Any = None
        self.active = False
        self.triggered = False
        self.loop: asyncio.AbstractEventLoop | None = None
        self.fd: int | None = None
        self.attributes: Any = None
        self.flush_handle: asyncio.TimerHandle | None = None

    def bind_input(self, input_backend: Any) -> None:
        if self.active:
            raise RuntimeError("Cannot replace interrupt input while active")
        self.input_backend = input_backend

    def start(self) -> None:
        if self.active:
            return
        self.triggered = False
        if self.input_backend is not None:
            try:
                self.loop = asyncio.get_running_loop()
                self.fd = int(self.input_backend.fileno())
                self.raw_context = self.input_backend.raw_mode()
                self.raw_context.__enter__()
                self.loop.add_reader(self.fd, self._read)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                if self.fd is not None and self.loop is not None:
                    self.loop.remove_reader(self.fd)
                self.fd = None
                self._close_bound_input()
            else:
                self.active = True
                return
        if os.name != "posix" or not sys.stdin.isatty():
            return
        fd: int | None = None
        attributes: Any = None
        try:
            import termios
            import tty

            fd = sys.stdin.fileno()
            attributes = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            loop = asyncio.get_running_loop()
            loop.add_reader(fd, self._read)
        except (AttributeError, OSError, RuntimeError, termios.error):
            if fd is not None and attributes is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, attributes)
                except (OSError, termios.error):
                    pass
            return
        self.fd = fd
        self.attributes = attributes
        self.loop = loop
        self.active = True

    def stop(self) -> None:
        if not self.active:
            return
        self.active = False
        if self.flush_handle is not None:
            self.flush_handle.cancel()
            self.flush_handle = None
        if self.fd is None:
            self._close_bound_input()
            return
        fd = self.fd
        self.fd = None
        if self.loop is not None:
            self.loop.remove_reader(fd)
        self._close_bound_input()
        if self.attributes is None:
            return
        try:
            import termios

            termios.tcsetattr(fd, termios.TCSADRAIN, self.attributes)
        except (OSError, termios.error):
            pass
        finally:
            self.attributes = None

    def _close_bound_input(self) -> None:
        raw_context, self.raw_context = self.raw_context, None
        if raw_context is not None:
            try:
                raw_context.__exit__(None, None, None)
            except (OSError, RuntimeError):
                pass

    def _trigger(self) -> None:
        if self.triggered:
            return
        self.triggered = True
        self.callback()

    def _read(self) -> None:
        if self.fd is None:
            return
        if self.input_backend is not None:
            try:
                key_presses = self.input_backend.read_keys()
            except (OSError, RuntimeError):
                self.stop()
                return
            if self.loop is not None:
                if self.flush_handle is not None:
                    self.flush_handle.cancel()
                self.flush_handle = self.loop.call_later(0.05, self._flush_keys)
            self._trigger_from_keys(key_presses)
            return
        try:
            value = os.read(self.fd, 32)
        except OSError:
            self.stop()
            return
        if value == b"\x1b" or b"\x03" in value:
            self._trigger()

    def _flush_keys(self) -> None:
        self.flush_handle = None
        if self.active and self.input_backend is not None:
            self._trigger_from_keys(self.input_backend.flush_keys())

    def _trigger_from_keys(self, key_presses: list[Any]) -> None:
        keys = [key_press.key for key_press in key_presses]
        if "c-c" in keys or keys == ["escape"]:
            self._trigger()


def _safe_cli_error(value: object) -> str:
    return safe_redacted_text(
        value,
        MAX_INTERACTIVE_ERROR_BYTES,
        "...[error truncated]",
    )


async def _close_components(*components: tuple[str, Any]) -> list[str]:
    if not components:
        return []
    results = await asyncio.gather(
        *(component.close() for _, component in components),
        return_exceptions=True,
    )
    return [
        f"{label} shutdown warning: {_safe_cli_error(result)}"
        for (label, _), result in zip(components, results, strict=True)
        if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError)
    ]


class _InteractiveWechatRuntime:
    """Own a WeChat channel running beside the interactive CLI event loop."""

    def __init__(self, paths: KairoPaths, config: AppConfig, console: Any) -> None:
        self.paths = paths
        self.config = config
        self.console = console
        self.channel: Any = None
        self.channel_task: asyncio.Task[None] | None = None
        self.manager: McpServerManager | None = None
        self.agent: Agent | None = None
        self.cleanup_task: asyncio.Task[None] | None = None
        self.input_session: Any = None

    async def command(self, payload: str | None) -> str:
        operation = (payload or "help").strip().casefold() or "help"
        if operation == "help":
            account = self._account_store().load()
            running = self.channel_task is not None and not self.channel_task.done()
            if account is None:
                status = "not connected"
            else:
                status = f"connected · channel {'running' if running else 'stopped'}"
            return (
                f"WeChat · {status}\n"
                "  /wechat setup    connect an account\n"
                "  /wechat start    start receiving messages\n"
                "  /wechat status   show connection status\n"
                "  /wechat stop     stop receiving messages\n"
                "Tip: type `/wechat ` to choose a command."
            )
        if operation == "status":
            account = self._account_store().load()
            if account is None:
                return "WeChat is not bound; run `/wechat setup`."
            running = self.channel_task is not None and not self.channel_task.done()
            return (
                f"WeChat bound to {account.account_id} in {account.workspace}; "
                f"channel={'running' if running else 'stopped'}."
            )
        if operation == "stop":
            if self.channel_task is None:
                return "WeChat channel is not running."
            await self.close()
            return "WeChat channel stopped."
        if operation == "setup":
            await self.close()
            try:
                await self._setup()
            except _WechatSetupCanceled:
                return "WeChat setup canceled."
            return await self._start()
        if operation == "start":
            return await self._start()
        return "Unknown WeChat command. Type `/wechat ` to choose setup, start, status, or stop."

    async def _setup(self) -> None:
        from datetime import UTC, datetime

        from ..channels.wechat import IlinkClient, WechatAccount

        entered = await _read_wechat_workspace_input(self.input_session, self.paths.workspace)
        workspace = await asyncio.to_thread(
            lambda: Path(entered.strip() or self.paths.workspace).expanduser().resolve()
        )
        if not await asyncio.to_thread(workspace.is_dir):
            raise ValueError(f"Workspace does not exist: {workspace}")
        client = IlinkClient()
        login = await client.start_qr_login()
        self.console.print(QR_SCAN_PROMPT)
        try:
            import qrcode  # type: ignore[import-untyped]

            qr = qrcode.QRCode(border=1)
            qr.add_data(login.qrcode_url)
            qr.print_ascii(invert=True)
        except Exception:
            pass
        self.console.print(login.qrcode_url)
        deadline = asyncio.get_running_loop().time() + 300
        result = None
        while asyncio.get_running_loop().time() < deadline:
            result = await client.poll_qr_status(login.qrcode_id)
            if result.connected or result.expired:
                break
            await asyncio.sleep(3)
        if result is None or not result.connected:
            raise RuntimeError(
                f"WeChat binding did not complete: {result.message if result else 'timeout'}"
            )
        self._account_store().save(
            WechatAccount(
                result.token,
                result.account_id,
                result.base_url,
                result.user_id,
                str(workspace),
                created_at=datetime.now(UTC).isoformat(),
            )
        )

    async def _start(self) -> str:
        from ..channels.wechat import IlinkClient, WechatChannel, WechatPolicy

        if self.channel_task is not None and not self.channel_task.done():
            return "WeChat channel is already running."
        if self.channel_task is not None:
            await self.close()
        store = self._account_store()
        account = store.load()
        if account is None:
            raise RuntimeError("No WeChat account is bound; run `/wechat setup` first")
        channel_paths = KairoPaths.discover(Path(account.workspace), self.paths.home)
        channel_config = AppConfig.load(channel_paths)
        policy = WechatPolicy()

        async def approve(name: str, arguments: dict[str, Any]) -> bool:
            return policy.allow_tool(name, arguments)

        agent = make_agent(
            channel_paths,
            channel_config,
            approval_policy=ApprovalPolicy(True),
            approver=approve,
        )
        manager = McpServerManager(channel_paths, agent.tools)
        try:
            await manager.start_all()
            _inject_mcp_resource_index(agent, manager)
        except BaseException:
            await _close_components(("MCP", manager), ("Tool", agent.tools))
            raise
        channel = WechatChannel(IlinkClient(), store, account, agent)
        self.agent = agent
        self.manager = manager
        self.channel = channel
        self.channel_task = asyncio.create_task(channel.run(), name="kairocli-interactive-wechat")
        self.channel_task.add_done_callback(self._on_channel_done)
        return f"WeChat channel started for {account.account_id}."

    def _on_channel_done(self, task: asyncio.Task[None]) -> None:
        if self.channel_task is task:
            self.cleanup_task = asyncio.create_task(
                self._cleanup_finished_channel(task),
                name="kairocli-interactive-wechat-cleanup",
            )

    async def _cleanup_finished_channel(self, task: asyncio.Task[None]) -> None:
        try:
            if self.channel_task is not task:
                return
            channel, manager, agent = self.channel, self.manager, self.agent
            self.channel_task = None
            self.channel = None
            self.manager = None
            self.agent = None
            if channel is not None:
                channel.running = False
            try:
                task.exception()
            except (asyncio.CancelledError, Exception):
                pass
            await self._close_wechat_components(manager, agent)
        finally:
            if self.cleanup_task is asyncio.current_task():
                self.cleanup_task = None

    @staticmethod
    async def _close_wechat_components(
        manager: McpServerManager | None, agent: Agent | None
    ) -> None:
        # The MCP manager may still reference the tool registry, so close in order.
        if manager is not None:
            await _close_components(("WeChat MCP", manager))
        if agent is not None:
            await _close_components(("WeChat tool", agent.tools))

    def _account_store(self) -> Any:
        from ..channels.wechat import WechatAccountStore

        return WechatAccountStore(self.paths)

    async def close(self) -> None:
        task, channel = self.channel_task, self.channel
        manager, agent = self.manager, self.agent
        cleanup_task = self.cleanup_task
        self.channel_task = None
        self.channel = None
        self.manager = None
        self.agent = None
        if channel is not None:
            channel.running = False
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self._close_wechat_components(manager, agent)
        if cleanup_task is not None and cleanup_task is not asyncio.current_task():
            await asyncio.gather(cleanup_task, return_exceptions=True)


async def _await_shutdown(task: asyncio.Task[list[str]]) -> list[str]:
    """Finish one shutdown transaction before propagating caller cancellation."""

    canceled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            canceled = True
    result = task.result()
    if canceled:
        raise asyncio.CancelledError
    return result


async def interactive(
    paths: KairoPaths,
    config: AppConfig,
    provider: str | None = None,
    resume_id: str | None = None,
    continue_session: bool = False,
    startup_notice: str = "",
    renderer_mode: str | None = None,
) -> int:
    selected_provider = normalize_provider_name(provider or config.default_provider)
    session_store = SessionStore(paths.session_database)
    todo_controller = SessionTodoController(session_store, paths.workspace)
    restored_session = None
    if resume_id:
        restored_session = await asyncio.to_thread(session_store.load, resume_id, paths.workspace)
        if restored_session is None:
            raise ValueError("Session was not found in the current workspace")
    elif continue_session:
        restored_session = await asyncio.to_thread(session_store.latest, paths.workspace)
    if restored_session is not None and provider is None:
        if restored_session.meta.provider in config.providers:
            selected_provider = restored_session.meta.provider
    approvals = ApprovalPolicy(True)

    async def approve(tool_name: str, arguments: dict[str, Any]) -> ApprovalResult:
        has_server_scope = ApprovalPolicy.mcp_server_name(tool_name) is not None
        was_running = turn_lifecycle.running
        turn_display.finish_block()
        stop_working()
        escape_interrupt.stop()
        if was_running:
            turn_lifecycle.pause()
        _print_approval_request(
            console,
            tool_name,
            arguments,
            has_server_scope=has_server_scope,
        )
        try:
            answer = (await _read_approval_input(session)).strip().lower()
            if answer in {"y", "yes"}:
                return ApprovalResult.approve()
            if answer in {"a", "all"}:
                return ApprovalResult.approve_all()
            if answer in {"v", "server"} and has_server_scope:
                return ApprovalResult.approve_all_by_server()
            if answer in {"s", "skip"}:
                return ApprovalResult.skip()
            if answer in {"m", "modify"}:
                edited = await _read_approval_input(session, "Replacement JSON  › ")
                try:
                    replacement = decode_strict_json(
                        edited,
                        max_bytes=1024 * 1024,
                        max_depth=32,
                        max_nodes=100_000,
                    )
                except (RecursionError, TypeError, UnicodeError, ValueError) as exc:
                    return ApprovalResult.reject(
                        "Invalid replacement JSON: " + _safe_cli_error(exc)
                    )
                if not isinstance(replacement, dict):
                    return ApprovalResult.reject("Replacement arguments must be a JSON object")
                return ApprovalResult.modify(replacement)
            return ApprovalResult.reject("User rejected the operation")
        except (EOFError, KeyboardInterrupt):
            return ApprovalResult.reject("Approval input was canceled")
        finally:
            if was_running:
                turn_lifecycle.resume()
                restart_working()
            escape_interrupt.start()

    skills = SkillRegistry(paths)
    skills.reload()
    browser = BrowserSession()
    agent = make_agent(
        paths,
        config,
        selected_provider,
        approvals,
        approve,
        skill_registry=skills,
        browser_session=browser,
        todo_controller=todo_controller,
    )
    if restored_session is None:
        restored_session = await asyncio.to_thread(
            session_store.create,
            paths.workspace,
            selected_provider,
            agent.llm.model,
        )
    else:
        apply_session(agent, restored_session)
    session_id = restored_session.meta.id
    todo_controller.attach(session_id)
    mcp_manager = McpServerManager(paths, agent.tools)
    _register_browser_agent_tools(agent, browser, mcp_manager)
    mcp_startup_warning = ""
    try:
        await mcp_manager.start_all()
        _inject_mcp_resource_index(agent, mcp_manager)
    except Exception as exc:
        await mcp_manager.close()
        mcp_startup_warning = f"MCP startup warning: {type(exc).__name__}: " + _safe_cli_error(exc)
    snapshots: SnapshotService = agent.tools.snapshot_service
    snapshots_enabled = snapshots.config.enabled
    plan_agent = PlanExecuteAgent(agent)
    team_agent = AgentOrchestrator(agent)
    memory = agent.memory_store or MemoryStore(paths)
    tasks = DurableTaskStore(paths.task_database)
    task_manager = DurableTaskManager(
        tasks,
        lambda: make_agent(paths, config, selected_provider),
        config.task_workers,
    )
    task_manager.start()
    pending_mode = "agent"
    console = _console()
    command_console = _CommandOutputConsole(console)
    wechat_runtime = _InteractiveWechatRuntime(paths, config, command_console)
    active_renderer = renderer_mode or config.renderer
    working_indicator: _WorkingIndicator | None = None
    turn_lifecycle = _TurnLifecycleState()
    thought_display = ThoughtDisplay()
    index_display = _IndexDisplay()
    turn_display = _StreamingAnswerDisplay(
        console,
        active_renderer,
        _terminal_columns(),
        thought_display.live_summary,
    )
    interrupt_notified = False

    def stop_working() -> None:
        if working_indicator is not None:
            working_indicator.stop()

    def restart_working() -> None:
        nonlocal working_indicator
        stop_working()
        working_indicator = _WorkingIndicator(console, turn_lifecycle)
        working_indicator.start()

    def request_interrupt() -> None:
        nonlocal interrupt_notified
        if interrupt_notified:
            return
        interrupt_notified = True
        agent.cancel()
        stop_working()
        _print_status(console, "\nCanceling current task...\n", "yellow")

    escape_interrupt = _EscapeInterrupt(request_interrupt)

    def bottom_status() -> str:
        return _mode_status_line(pending_mode, agent.status_line())

    def display_activity(message: str, style: str = "dim") -> None:
        if not message:
            return
        turn_display.finish_block()
        stop_working()
        _write_stream("\n")
        _print_status(console, message, style)
        _write_stream("\n")

    def display_turn_message(message: object) -> None:
        turn_display.finish_block()
        stop_working()
        escape_interrupt.stop()
        command_console.print(message)

    def display_content_delta(delta: str) -> None:
        thought_display.finish_thinking()
        stop_working()
        turn_display.append(delta)

    agent.on_content_delta = display_content_delta
    agent.on_reasoning = thought_display.add
    agent.on_reasoning_delta = thought_display.add_delta

    def display_tool_calls(calls: Any) -> None:
        summary = format_tool_calls(calls, compact=False)
        if summary:
            display_activity(summary)

    agent.on_tool_calls = display_tool_calls

    def display_tool_results(calls: Any, results: Any) -> None:
        summary = format_tool_results(calls, results)
        if summary:
            display_activity(summary, "yellow" if summary.startswith("⚠") else "green")
        for output in results:
            for diff in output.diffs:
                display_activity(render_file_diff(diff))
        restart_working()

    agent.on_tool_results = display_tool_results

    async def review_plan(execution_plan: Any) -> bool | str:
        turn_display.finish_block()
        stop_working()
        escape_interrupt.stop()
        turn_lifecycle.pause()
        command_console.print(_plan_summary(execution_plan))
        try:
            answer = await _read_plan_review_input(session)
        except (EOFError, KeyboardInterrupt):
            answer = "\x1b"
        decision = parse_plan_review_input(answer)
        if decision.type == PlanReviewDecisionType.CANCEL:
            return False
        turn_lifecycle.resume()
        restart_working()
        escape_interrupt.start()
        if decision.type == PlanReviewDecisionType.SUPPLEMENT:
            return decision.feedback or False
        return True

    plan_agent.review_handler = review_plan
    _print_welcome(
        console,
        selected_provider,
        agent.llm.model or "unset",
        paths.workspace,
    )
    if startup_notice:
        _print_status(console, _terminal_background_block(startup_notice), "on default")
    if mcp_startup_warning:
        _print_status(console, _terminal_background_block(mcp_startup_warning), "on default")
    session_notice = _session_startup_notice(len(agent.history))
    if session_notice is not None:
        _print_status(
            console,
            _terminal_background_block(session_notice),
            "dim on default",
        )
    _print_status(console, _terminal_background_block(""), "on default")

    shutdown_task: asyncio.Task[list[str]] | None = None

    async def shutdown() -> list[str]:
        nonlocal shutdown_task

        async def finish() -> list[str]:
            try:
                await _save_session(session_store, session_id, paths, agent, command_console)
            finally:
                warnings = await _close_components(
                    ("WeChat", wechat_runtime),
                    ("MCP", mcp_manager),
                    ("Task", task_manager),
                    ("Tool", agent.tools),
                )
            return warnings

        if shutdown_task is None:
            shutdown_task = asyncio.create_task(finish(), name="kairocli-interactive-shutdown")
        return await _await_shutdown(shutdown_task)

    async def finish_normal_exit() -> int:
        for warning in await shutdown():
            command_console.print(warning)
        _print_status(console, "\n" + _resume_session_hint(session_id), "dim")
        return 0

    try:
        session = _prompt_session(
            paths,
            bottom_status,
            mcp_manager,
            config,
            skills,
            thought_display,
            index_display,
            renderer=active_renderer,
        )
    except BaseException:
        await shutdown()
        raise
    wechat_runtime.input_session = session
    if session is not None:
        escape_interrupt.bind_input(session.app.input)
    while True:
        try:
            # The composer is the sole owner of terminal input while it is running.
            # This also repairs ownership after any exceptional turn teardown.
            escape_interrupt.stop()
            raw = normalize_interactive_submission(await _read_input(session))
            thought_display.dismiss()
            index_display.dismiss()
        except (EOFError, KeyboardInterrupt):
            command_console.print("Goodbye.")
            return await finish_normal_exit()
        except asyncio.CancelledError:
            agent.cancel()
            await shutdown()
            raise
        except UserInputError as exc:
            command_console.print("Input error: " + _safe_cli_error(exc))
            continue
        except Exception:
            await shutdown()
            raise
        if _is_blank_interactive_submission(raw):
            continue
        command = parse_command(raw)
        if command.type != CommandType.NONE:
            if command.type == CommandType.HISTORY_CLEAR:
                cleared = _clear_input_history(paths.history_file)
                session = _prompt_session(
                    paths,
                    bottom_status,
                    mcp_manager,
                    config,
                    skills,
                    thought_display,
                    index_display,
                    renderer=active_renderer,
                )
                wechat_runtime.input_session = session
                if session is not None:
                    escape_interrupt.bind_input(session.app.input)
                command_console.print(
                    "Input history cleared."
                    if cleared
                    else "Input history was not cleared because its path is unsafe."
                )
                continue
            if command.type == CommandType.SESSION:
                session_id = await _handle_session_command(
                    command.payload,
                    session_store,
                    session_id,
                    paths,
                    agent,
                    command_console,
                )
                todo_controller.attach(session_id)
                continue
            if command.type == CommandType.TODO:
                try:
                    command_console.print(await todo_controller.command(command.payload))
                except ValueError as exc:
                    command_console.print("Todo error: " + _safe_cli_error(exc))
                continue
            try:
                outcome = await _handle_command(
                    command,
                    paths,
                    config,
                    agent,
                    memory,
                    tasks,
                    task_manager,
                    approvals,
                    mcp_manager,
                    skills,
                    browser,
                    wechat_runtime,
                    command_console,
                    index_display=index_display,
                )
            except SnapshotError as exc:
                _print_snapshot_warning(command_console, exc)
                continue
            except Exception as exc:
                _print_untrusted(command_console, "Command error: " + _safe_cli_error(exc))
                continue
            if outcome == "exit":
                return await finish_normal_exit()
            if outcome in {"plan", "team"}:
                if command.payload:
                    raw = command.payload
                    pending_mode = outcome
                else:
                    pending_mode = outcome
                    command_console.print(f"Next task will use {outcome} mode.")
                    continue
            else:
                await _save_session(session_store, session_id, paths, agent, command_console)
                continue
        try:
            prepared_images = await prepare_image_input(
                raw,
                paths.workspace,
                paths.user_dir / "cache" / "clipboard",
            )
            image_urls = list(prepared_images.image_urls)
        except (OSError, ValueError) as exc:
            command_console.print("Image error: " + _safe_cli_error(exc))
            continue
        text_without_images = prepared_images.text
        expanded = expand_local_mentions(text_without_images, paths.workspace)
        try:
            expanded = await mcp_manager.expand_resource_mentions(expanded)
        except Exception as exc:
            command_console.print("MCP resource error: " + _safe_cli_error(exc))
            continue
        snapshot_mode = pending_mode
        pre_message, post_message = turn_snapshot_messages(snapshot_mode, text_without_images)
        pre_turn_snapshot = not snapshots_enabled
        try:
            thought_display.start(text_without_images)
            turn_display.start_turn()
            if snapshots_enabled:
                pre_turn_snapshot = await _try_capture_snapshot(
                    snapshots,
                    pre_message,
                    command_console,
                )
            turn_lifecycle.start()
            interrupt_notified = False
            working_indicator = _WorkingIndicator(console, turn_lifecycle)
            working_indicator.start()
            escape_interrupt.start()
            if pending_mode == "plan":
                answer = await plan_agent.run(expanded, image_urls)
            elif pending_mode == "team":
                answer = await team_agent.run(expanded, image_urls)
            else:
                answer = await agent.run(expanded, image_urls)
            thought_display.finish(
                answer,
                answer_streamed=turn_display.content_committed,
            )
            stop_working()
            escape_interrupt.stop()
        except asyncio.CancelledError:
            agent.cancel()
            display_turn_message("Task canceled.")
        except AgentCanceled:
            display_turn_message("Task canceled.")
        except (LlmError, RuntimeError) as exc:
            display_turn_message("Error: " + _safe_cli_error(exc))
        except Exception as exc:
            display_turn_message("Unexpected error: " + _safe_cli_error(exc))
        finally:
            if working_indicator is not None:
                working_indicator.stop()
            turn_lifecycle.finish()
            turn_display.finish_block()
            escape_interrupt.stop()
            if snapshots_enabled and pre_turn_snapshot:
                try:
                    snapshots.capture_background(
                        post_message,
                        lambda error: _print_snapshot_warning(command_console, error),
                    )
                except Exception as exc:
                    command_console.print(
                        "Snapshot warning: " + _safe_cli_error(exc),
                    )
            pending_mode = "agent"
            await _save_session(session_store, session_id, paths, agent, command_console)


async def _try_capture_snapshot(snapshots: SnapshotService, message: str, console: Any) -> bool:
    try:
        await snapshots.capture(message)
    except SnapshotError as exc:
        _print_snapshot_warning(console, exc)
        return False
    return True


def _print_snapshot_warning(console: Any, error: SnapshotError) -> None:
    warning = "Snapshot warning: " + _safe_cli_error(error)
    if _has_rich(console):
        console.print(warning, style="yellow")
    else:
        console.print(warning)


async def _save_session(
    store: SessionStore,
    session_id: str,
    paths: KairoPaths,
    agent: Agent,
    console: Any,
) -> bool:
    snapshot = store.capture_snapshot(agent)
    try:
        await asyncio.to_thread(
            store.save_snapshot,
            session_id,
            paths.workspace,
            snapshot,
        )
    except Exception as exc:  # Persistence failure must not discard the active conversation.
        console.print("Session warning: " + _safe_cli_error(exc))
        return False
    return True


async def _handle_session_command(
    payload: str | None,
    store: SessionStore,
    current_id: str,
    paths: KairoPaths,
    agent: Agent,
    console: Any,
) -> str:
    operation, _, argument = (payload or "status").strip().partition(" ")
    operation = operation.casefold() or "status"
    if operation in {"status", "save"}:
        saved = await _save_session(store, current_id, paths, agent, console)
        if saved:
            state = await asyncio.to_thread(store.load, current_id, paths.workspace)
            if state is not None:
                console.print(
                    f"Session {state.meta.id} · {state.meta.message_count} messages · "
                    f"{state.meta.provider}/{state.meta.model}\n{state.meta.title}"
                )
        return current_id
    if operation == "list":
        list_argument = argument.strip()
        if list_argument not in {"", "--all"}:
            console.print("Usage: /session list [--all]")
            return current_id
        sessions = await asyncio.to_thread(store.list, paths.workspace, 100)
        console.print(
            format_session_list(
                sessions,
                current_id,
                show_all=list_argument == "--all",
                width=_terminal_columns(),
            )
        )
        return current_id
    if operation == "new":
        await _save_session(store, current_id, paths, agent, console)
        state = await asyncio.to_thread(
            store.create,
            paths.workspace,
            agent.llm.provider,
            agent.llm.model,
        )
        apply_session(agent, state)
        console.print(f"Started session {state.meta.id}.")
        return state.meta.id
    if operation == "resume":
        target_id = argument.strip()
        if not target_id:
            console.print("Usage: /session resume <SESSION_ID>")
            return current_id
        try:
            state = await asyncio.to_thread(store.load, target_id, paths.workspace)
        except ValueError as exc:
            console.print(_safe_cli_error(exc))
            return current_id
        if state is None:
            console.print("Session was not found in the current workspace.")
            return current_id
        await _save_session(store, current_id, paths, agent, console)
        apply_session(agent, state)
        console.print(f"Resumed {target_id} with {len(state.messages)} messages.")
        if state.meta.provider != agent.llm.provider:
            console.print(
                f"Stored provider was {state.meta.provider}/{state.meta.model}; "
                f"continuing with active {agent.llm.provider}/{agent.llm.model}."
            )
        return target_id
    if operation == "delete":
        targets = argument.split()
        if not targets:
            console.print("Usage: /session delete <SESSION_ID...|--empty>")
        elif targets == ["--empty"]:
            deleted = await asyncio.to_thread(
                store.delete_empty,
                paths.workspace,
                exclude_session_id=current_id,
            )
            if deleted:
                noun = "session" if deleted == 1 else "sessions"
                console.print(f"Deleted {deleted} empty {noun}.")
            else:
                console.print("No empty sessions to delete.")
        elif any(target.startswith("--") for target in targets):
            console.print("Usage: /session delete <SESSION_ID...|--empty>")
        elif current_id in targets:
            console.print("Cannot delete the active session; start a new session first.")
        else:
            try:
                deleted = await asyncio.to_thread(store.delete_many, targets, paths.workspace)
            except ValueError as exc:
                console.print(_safe_cli_error(exc))
            else:
                noun = "session" if deleted == 1 else "sessions"
                console.print(f"Deleted {deleted} {noun}.")
        return current_id
    console.print("Usage: /session [status|save|list|new|resume ID|delete ID...]")
    return current_id


async def _handle_command(
    command: ParsedCommand,
    paths: KairoPaths,
    config: AppConfig,
    agent: Agent,
    memory: MemoryStore,
    tasks: DurableTaskStore,
    task_manager: DurableTaskManager,
    approvals: ApprovalPolicy,
    mcp_manager: McpServerManager,
    skills: SkillRegistry,
    browser: BrowserSession,
    wechat_runtime: _InteractiveWechatRuntime,
    console: Any,
    index_display: _IndexDisplay | None = None,
) -> str | None:
    payload = command.payload or ""
    if command.type == CommandType.EXIT:
        console.print("Goodbye.")
        return "exit"
    if command.type == CommandType.HELP:
        console.print(SLASH_HELP)
    elif command.type == CommandType.UNKNOWN:
        console.print(f"Unknown command: {payload}")
    elif command.type == CommandType.CANCEL:
        agent.cancel()
        console.print("Cancellation requested.")
    elif command.type == CommandType.CLEAR:
        agent.clear()
        console.print("Conversation cleared; long-term memory retained.")
    elif command.type == CommandType.COMPACT:
        compacted = await agent.compact()
        console.print("Conversation compacted." if compacted else "Nothing to compact.")
    elif command.type == CommandType.HISTORY:
        console.print("Input history · type `/history ` to choose clear.")
    elif command.type == CommandType.HISTORY_CLEAR:
        cleared = _clear_input_history(paths.history_file)
        console.print(
            "Input history cleared."
            if cleared
            else "Input history was not cleared because its path is unsafe."
        )
    elif command.type == CommandType.INIT:
        try:
            target = initialize_project_memory(paths.workspace, payload == "--force")
            console.print(f"Created {target.name}")
        except FileExistsError as exc:
            console.print(_safe_cli_error(exc))
    elif command.type == CommandType.MODEL:
        console.print(handle_model_command(payload, config, paths))
    elif command.type == CommandType.PLAN:
        return "plan"
    elif command.type == CommandType.TEAM:
        return "team"
    elif command.type == CommandType.HITL:
        if payload in {"on", "off"}:
            approvals.enabled = payload == "on"
            approvals.clear_session_approvals()
        console.print(f"HITL approvals: {'on' if approvals.enabled else 'off'}")
    elif command.type == CommandType.MEMORY:
        console.print(await handle_memory_command(payload, memory))
    elif command.type == CommandType.SAVE:
        console.print(await handle_save_command(payload, memory))
    elif command.type == CommandType.INDEX:
        target = paths.workspace if not payload else agent.tools.path_guard.resolve(payload)
        progress = _IndexProgress(console)
        indexed_paths: list[str] = []
        progress.start()
        try:
            result = await agent.tools.code_index.index(
                target, progress.update, indexed_paths.append
            )
        finally:
            progress.stop()
        if index_display is None:
            console.print(f"Indexed {result['files']} files into {result['chunks']} chunks.")
        else:
            index_display.finish(result["files"], result["chunks"], indexed_paths)
    elif command.type == CommandType.SEARCH:
        if not payload:
            console.print("Usage: /search <query>")
        else:
            matches = await agent.tools.code_index.search(payload)
            for match in matches:
                console.print(
                    render_code_search_match(
                        match["path"],
                        match["start_line"],
                        match["end_line"],
                        match["score"],
                        match["content"],
                    )
                )
    elif command.type == CommandType.GRAPH:
        if not payload:
            console.print("Usage: /graph <symbol>")
        else:
            relations = agent.tools.code_index.graph(payload)
            if not relations:
                console.print("No indexed symbol relations. Run /index after code changes.")
            for item in relations:
                console.print(
                    f"{item['from_name']} ── {item['kind']} --> [{item['to_name']}] "
                    f"({item['path']}:{item['line']})"
                )
    elif command.type == CommandType.CONTEXT:
        console.print(agent.context_status())
    elif command.type == CommandType.POLICY:
        hitl_state = "on" if approvals.enabled else "off"
        console.print(
            f"Workspace fence: {paths.workspace}\nHITL: {hitl_state}\nAudit: {paths.audit_dir}"
        )
    elif command.type == CommandType.SNAPSHOT:
        if agent.tools.snapshot_service is None:
            console.print("Snapshots are disabled.")
        elif payload == "status":
            console.print(await agent.tools.snapshot_service.status())
        elif payload == "clean":
            cleaned = await agent.tools.snapshot_service.clean()
            console.print("Snapshot history cleaned." if cleaned else "No snapshot history.")
        elif payload not in {None, "list"}:
            console.print("Usage: /snapshot [list|status|clean]")
        else:
            pre_turn_number = 0
            for snapshot in await agent.tools.snapshot_service.list():
                restore_hint = ""
                if snapshot.phase == "pre-turn":
                    pre_turn_number += 1
                    restore_hint = f" /restore {pre_turn_number}"
                console.print(
                    f"{snapshot.short_revision} {snapshot.phase} {snapshot.turn_id}"
                    f" {snapshot.created_at}{restore_hint}"
                )
    elif command.type == CommandType.RESTORE:
        if agent.tools.snapshot_service is None:
            console.print("Snapshots are disabled.")
        elif not payload.isdigit():
            console.print("Usage: /restore <N>")
        else:
            result = await agent.tools.snapshot_service.restore_pre_turn(int(payload))
            console.print(
                f"{result.message} Restored {len(result.restored_files)} file(s); "
                f"removed {len(result.removed_files)}."
                if result.success
                else result.message
            )
    elif command.type == CommandType.AUDIT:
        limit = int(payload) if payload.isdigit() else 10
        console.print(read_recent_audit(paths, limit))
    elif command.type == CommandType.TASK:
        _handle_task(payload, tasks, task_manager, console)
    elif command.type == CommandType.EXPORT:
        try:
            target = write_session_export(paths, agent.export_markdown())
        except (OSError, ValueError) as exc:
            console.print("Session export failed: " + _safe_cli_error(exc))
        else:
            console.print(f"Exported to {target}")
    elif command.type == CommandType.MCP:
        await _handle_mcp(payload, mcp_manager, console, agent)
    elif command.type == CommandType.SKILL:
        await _handle_skill(payload, skills, console, agent)
    elif command.type == CommandType.BROWSER:
        await _handle_browser(
            payload,
            browser,
            mcp_manager,
            approvals,
            agent.tools,
            console,
        )
    elif command.type == CommandType.SHELL:
        await _handle_shell(payload, agent.tools, console)
    elif command.type == CommandType.TRACE:
        _handle_trace(payload, agent, console)
    elif command.type == CommandType.CONFIG:
        _handle_config(payload, config, paths, console)
    elif command.type == CommandType.WECHAT:
        console.print(await wechat_runtime.command(payload))
    else:
        console.print(f"{command.type.value} is not available in this build yet.")
    return None


async def _handle_shell(payload: str, tools: ToolRegistry, console: Any) -> None:
    operation, _, argument = (payload or "list").strip().partition(" ")
    operation = operation.casefold() or "list"
    if operation == "list":
        result = await tools.execute("shell_list", {})
    elif operation == "start":
        args = {"cwd": argument.strip()} if argument.strip() else {}
        result = await tools.execute("shell_start", args)
    elif operation == "exec":
        identifier, separator, command = argument.strip().partition(" ")
        if not separator or not command.strip():
            console.print("Usage: /shell exec <SHELL_ID> <command>")
            return
        result = await tools.execute(
            "shell_exec",
            {"session_id": identifier, "command": command.strip()},
        )
    elif operation == "stop":
        if not argument.strip():
            console.print("Usage: /shell stop <SHELL_ID>")
            return
        result = await tools.execute("shell_stop", {"session_id": argument.strip()})
    else:
        console.print("Usage: /shell [list|start [cwd]|exec ID COMMAND|stop ID]")
        return
    try:
        parsed = json.loads(result)
    except json.JSONDecodeError:
        console.print(result)
    else:
        console.print(json.dumps(parsed, ensure_ascii=False, indent=2))


def _handle_trace(payload: str, agent: Agent, console: Any) -> None:
    logger = agent.trace_logger
    if logger is None:
        console.print("Model tracing is unavailable for this Agent.")
        return
    operation = (payload or "status").strip().casefold()
    if operation == "on":
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
        console.print("Usage: /trace [status|on|off|reasoning on|reasoning off]")
        return
    console.print(
        f"Model trace: {'on' if logger.enabled else 'off'} · reasoning: "
        f"{'on' if logger.include_reasoning else 'off'} · {logger.directory}"
    )


async def _handle_memory(payload: str, memory: MemoryStore, console: Any) -> None:
    console.print(await handle_memory_command(payload, memory))


def _handle_task(
    payload: str,
    tasks: DurableTaskStore,
    task_manager: DurableTaskManager,
    console: Any,
) -> None:
    console.print(handle_task_command(payload, tasks, task_manager))


async def _handle_mcp(
    payload: str,
    manager: McpServerManager,
    console: Any,
    agent: Agent | None = None,
) -> None:
    console.print(
        await handle_mcp_command(
            payload,
            manager,
            agent,
            getattr(agent.tools, "approval_policy", None) if agent is not None else None,
        )
    )


async def _handle_skill(
    payload: str, registry: SkillRegistry, console: Any, agent: Agent | None = None
) -> None:
    console.print(await asyncio.to_thread(handle_skill_command, payload, registry, agent))


async def _handle_browser(
    payload: str,
    browser: BrowserSession,
    mcp_manager: McpServerManager,
    approvals: ApprovalPolicy,
    tools: ToolRegistry,
    console: Any,
) -> None:
    console.print(
        await handle_browser_command(
            payload,
            browser,
            mcp_manager=mcp_manager,
            approval_policy=approvals,
            tools=tools,
        )
    )


def _handle_config(payload: str, config: AppConfig, paths: KairoPaths, console: Any) -> None:
    console.print(handle_config_command(payload, config, paths))


def expand_local_mentions(
    text: str,
    workspace: Path,
    max_chars: int = 50_000,
    *,
    max_mentions: int = 20,
    max_file_bytes: int = 120_000,
    max_dir_entries: int = 80,
) -> str:
    """Expand workspace paths without allowing repository data to forge prompt markup."""
    if not text or "@" not in text or max_chars <= 0 or max_mentions <= 0:
        return text
    pattern = re.compile(r"(^|\s)@(?![^\s<>]*:)(<[^>]+>|[^\s<>:]+)")
    try:
        root = workspace.resolve(strict=True)
    except OSError:
        root = workspace.resolve()
    remaining = max_chars
    expanded_count = 0
    expanded_paths: set[Path] = set()

    def bounded(block: str, original: str) -> str:
        nonlocal remaining
        if len(block) > remaining:
            marker = (
                f'{original}\n<local-context partial="true">'
                "context budget exhausted</local-context>"
            )
            if len(marker) > remaining:
                return original
            block = marker
        remaining -= len(block)
        return block

    def replace(match: re.Match[str]) -> str:
        nonlocal expanded_count
        leading, raw = match.group(1), match.group(2)
        original = f"@{raw}"
        value = raw[1:-1] if raw.startswith("<") and raw.endswith(">") else raw
        if (
            not value
            or value == "clipboard"
            or value.startswith("image:")
            or ":" in value
            or len(value) > 4096
            or expanded_count >= max_mentions
        ):
            return leading + original
        try:
            source = Path(value).expanduser()
            candidate = (source if source.is_absolute() else root / source).resolve(strict=True)
            candidate.relative_to(root)
        except (OSError, RuntimeError, ValueError):
            return leading + original
        display = "." if candidate == root else candidate.relative_to(root).as_posix()
        escaped_path = html.escape(display, quote=True)
        if candidate in expanded_paths:
            duplicate_block = (
                f'@&lt;{escaped_path}&gt;\n<local-context path="{escaped_path}" duplicate="true" />'
            )
            return leading + bounded(duplicate_block, original)

        block: str
        try:
            if candidate.is_file():
                with candidate.open("rb") as handle:
                    payload = handle.read(max_file_bytes + 1)
                truncated = len(payload) > max_file_bytes
                payload = payload[:max_file_bytes]
                if b"\x00" in payload:
                    block = (
                        f'@&lt;{escaped_path}&gt;\n<file path="{escaped_path}" '
                        'binary="true">binary content omitted</file>'
                    )
                else:
                    content = html.escape(payload.decode("utf-8", errors="replace"), quote=False)
                    partial = ' partial="true"' if truncated else ""
                    suffix = (
                        f"\n[file truncated by Kairo CLI at {max_file_bytes} bytes]"
                        if truncated
                        else ""
                    )
                    block = (
                        f'@&lt;{escaped_path}&gt;\n<file path="{escaped_path}"{partial}>\n'
                        f"{content}{suffix}\n</file>"
                    )
            elif candidate.is_dir():
                entries: list[tuple[str, bool]] = []
                with os.scandir(candidate) as iterator:
                    for entry in iterator:
                        entries.append((entry.name, entry.is_dir(follow_symlinks=False)))
                        if len(entries) > max_dir_entries:
                            break
                entries.sort(key=lambda item: item[0])
                truncated = len(entries) > max_dir_entries
                listing = "\n".join(
                    f"- {html.escape(name, quote=False)}{'/' if is_dir else ''}"
                    for name, is_dir in entries[:max_dir_entries]
                )
                if truncated:
                    listing += f"\n[directory truncated by Kairo CLI at {max_dir_entries} entries]"
                partial = ' partial="true"' if truncated else ""
                block = (
                    f'@&lt;{escaped_path}&gt;\n<directory path="{escaped_path}"{partial}>\n'
                    f"{listing}\n</directory>"
                )
            else:
                return leading + original
        except OSError:
            return leading + original

        expanded_count += 1
        expanded_paths.add(candidate)
        return leading + bounded(block, original)

    return pattern.sub(replace, text)


def _console() -> Any:
    try:
        from rich.console import Console

        class TerminalSafeConsole(Console):
            _kairo_rich = True

            def print(self, *objects: Any, **kwargs: Any) -> None:
                safe_objects = tuple(
                    sanitize_terminal_text(item) if isinstance(item, str) else item
                    for item in objects
                )
                kwargs.setdefault("markup", False)
                super().print(*safe_objects, **kwargs)

        return TerminalSafeConsole(
            color_system="truecolor",
            style="on default",
        )
    except ImportError:

        class PlainConsole:
            def print(self, value: object = "") -> None:
                print(value)

        return PlainConsole()


def _has_rich(console: Any) -> bool:
    return bool(
        getattr(console, "_kairo_rich", False) or console.__class__.__module__.startswith("rich.")
    )


def _console_color_system(console: Any) -> ColorSystemName | None:
    value = getattr(console, "color_system", None)
    if _has_rich(console) and not getattr(console, "no_color", False) and value in {
        "standard",
        "256",
        "truecolor",
        "windows",
    }:
        return cast(ColorSystemName, value)
    return None


def _write_stream(value: str) -> None:
    sys.stdout.write(sanitize_terminal_text(value))
    sys.stdout.flush()


def _write_rendered_stream(value: RenderedTerminalText) -> None:
    """Write text already sanitized by a renderer, preserving its generated ANSI styles."""
    if not isinstance(value, RenderedTerminalText):
        raise TypeError("rendered terminal output must cross the trusted renderer boundary")
    sys.stdout.write(value)
    sys.stdout.flush()


def _end_answer_block() -> None:
    """Leave one terminal-background row below an answer before the composer."""
    _write_stream("\n")


def _terminal_columns() -> int:
    return max(40, min(shutil.get_terminal_size(fallback=(120, 24)).columns, 1_000))


def _terminal_background_block(value: object, columns: int | None = None) -> str:
    width = columns if columns is not None else _terminal_columns()
    lines = safe_text(value).splitlines() or [""]
    return "\n".join(line.ljust(width) for line in lines)


def _erase_with_default_background(output: Any, command: str) -> None:
    """Erase terminal cells using its configured background, not a fixed color."""
    output.write_raw("\x1b[49m" + command + "\x1b[0m")


def _render_interactive_answer(
    value: str,
    renderer: str,
    columns: int,
    color_system: ColorSystemName | None = None,
) -> RenderedTerminalText:
    if renderer == "inline":
        rendered = TerminalMarkdownRenderer.render(
            value, columns, continuation_indent="  ", color_system=color_system
        )
        return RenderedTerminalText(rendered.rstrip("\n"))
    safe = sanitize_terminal_text(value)
    return RenderedTerminalText(
        "\n".join(
            ("  " + line if index and line else line)
            for index, line in enumerate(safe.split("\n"))
        )
    )


def _print_untrusted(console: Any, value: object = "") -> None:
    safe = sanitize_terminal_text(safe_text(value))
    if _has_rich(console):
        console.print(safe, markup=False, highlight=False)
    else:
        console.print(safe)


def _print_status(console: Any, value: object, style: str) -> None:
    safe = sanitize_terminal_text(safe_text(value))
    if _has_rich(console):
        console.print(safe, style=style, markup=False, highlight=False)
    else:
        console.print(safe)


def _print_answer_prefix(console: Any) -> None:
    if _has_rich(console):
        console.print("● ", style="#888888", markup=False, highlight=False, end="")
    else:
        _write_stream("● ")


def _print_command_output(console: Any, value: object = "") -> None:
    """Render local command output with the same spacing as a model answer."""
    rendered = _render_interactive_answer(safe_text(value), "plain", _terminal_columns())
    _write_stream("\n")
    _print_answer_prefix(console)
    _print_untrusted(console, rendered)
    _end_answer_block()


class _CommandOutputConsole:
    def __init__(self, console: Any) -> None:
        self.console = console
        self._kairo_rich = _has_rich(console)

    def print(self, value: object = "", **_kwargs: object) -> None:
        if self._kairo_rich and hasattr(value, "__rich_console__"):
            _write_stream("\n")
            _print_answer_prefix(self.console)
            self.console.print(value)
            _end_answer_block()
            return
        _print_command_output(self.console, value)


def _welcome_lines(provider: str, model: str, workspace: Path, columns: int = 120) -> list[str]:
    width = max(40, min(columns, 120))

    location = str(workspace)
    try:
        location = "~/" + str(workspace.relative_to(Path.home()))
    except (OSError, RuntimeError, ValueError):
        pass

    title = f"{PRODUCT_NAME} {VERSION}"
    top = f"╭─ {title} " + "─" * max(0, width - len(title) - 5) + "╮"

    def fit(value: str, cell_width: int, *, center: bool = False) -> str:
        maximum = max(0, cell_width - 2)
        visible = value if len(value) <= maximum else value[: maximum - 1] + "…"
        return f" {visible:^{maximum}} " if center else f" {visible:<{maximum}} "

    model_line = f"{provider}/{model} · API usage"
    tips = [
        "Tips for getting started",
        "Run /init to create KAIRO.md",
        "",
        _WELCOME_SECTION_DIVIDER,
        "What's new",
        "Plan, ReAct and Team modes",
        "MCP, memory and safe tools",
        "",
        "/help for commands",
        "/exit to quit",
    ]

    if width < 64:
        inner = width - 2
        left = [
            "Welcome back!",
            "",
            *_WELCOME_PIXEL_WORDMARK,
            "",
            model_line,
            location,
        ]
        rows = [f"│{fit(value, inner, center=True)}│" for value in left]
        rows.append("├" + "─" * inner + "┤")
        rows.extend(
            f"│{fit('─' * (inner - 2), inner)}│"
            if value == _WELCOME_SECTION_DIVIDER
            else f"│{fit(value, inner)}│"
            for value in tips[:10]
        )
        return [top, *rows, "╰" + "─" * inner + "╯"]

    inner = width - 2
    left_width = int((inner - 1) * 0.55)
    right_width = inner - left_width - 1
    left = [
        "",
        "Welcome back!",
        "",
        *_WELCOME_PIXEL_WORDMARK,
        "",
        model_line,
        location,
    ]
    row_count = max(len(left), len(tips))
    left.extend([""] * (row_count - len(left)))
    tips.extend([""] * (row_count - len(tips)))
    body = []
    for index, (left_value, right_value) in enumerate(zip(left, tips, strict=True)):
        divider = "╷" if index == 0 else "╵" if index == row_count - 1 else "│"
        right_content = (
            "─" * (right_width - 2) if right_value == _WELCOME_SECTION_DIVIDER else right_value
        )
        body.append(
            f"│{fit(left_value, left_width, center=True)}{divider}"
            f"{fit(right_content, right_width)}│"
        )
    return [top, *body, "╰" + "─" * inner + "╯"]


def _print_welcome(console: Any, provider: str, model: str, workspace: Path) -> None:
    lines = _welcome_lines(provider, model, workspace, _terminal_columns())
    if not _has_rich(console):
        for line in lines:
            _print_status(console, line, "")
        return

    from rich.text import Text

    accents = (
        f"{PRODUCT_NAME} {VERSION}",
        "Tips for getting started",
        "What's new",
        "/init",
        "/help",
        "/exit",
    )
    for line in lines:
        rendered = Text(line)
        for position, character in enumerate(line):
            if character in "─│╷╵╭╮╰╯┬┴├┤":
                rendered.stylize("dim", position, position + 1)
        for wordmark_row in _WELCOME_PIXEL_WORDMARK:
            start = line.find(wordmark_row)
            if start >= 0:
                rendered.stylize(_WELCOME_ACCENT_STYLE, start, start + len(wordmark_row))
        for accent in accents:
            start = line.find(accent)
            if start >= 0:
                rendered.stylize(_WELCOME_ACCENT_STYLE, start, start + len(accent))
        if "· API usage" in line or "~/" in line:
            left_border = line.find("│")
            divider = line.find("│", left_border + 1)
            end = divider if divider >= 0 else len(line) - 1
            rendered.stylize("dim", left_border + 1, end)
        console.print(rendered, markup=False, highlight=False)


def _prompt_session(
    paths: KairoPaths,
    bottom_toolbar: Any = None,
    mcp_manager: McpServerManager | None = None,
    config: AppConfig | None = None,
    skill_registry: SkillRegistry | None = None,
    thought_display: ThoughtDisplay | None = None,
    index_display: _IndexDisplay | None = None,
    *,
    renderer: str = "inline",
) -> Any:
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import Completer
        from prompt_toolkit.document import Document
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.formatted_text import FormattedText
        from prompt_toolkit.history import FileHistory, InMemoryHistory
        from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.keys import Keys
        from prompt_toolkit.layout import Dimension
        from prompt_toolkit.layout.containers import (
            ConditionalContainer,
            Window,
        )
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.lexers import Lexer
        from prompt_toolkit.output import ColorDepth, create_output
        from prompt_toolkit.output.vt100 import Vt100_Output
        from prompt_toolkit.styles import Style

        commands = list(SLASH_COMMAND_DESCRIPTIONS)

        # prompt_toolkit currently collapses xterm's modified Shift+Enter into
        # a plain Enter, and does not recognize the equivalent CSI-u sequence.
        # Preserve both as Control-J, which the composer binds to a newline.
        ANSI_SEQUENCES["\x1b[27;2;13~"] = Keys.ControlJ
        ANSI_SEQUENCES["\x1b[13;2u"] = Keys.ControlJ

        def slash_candidates(text: str) -> list[str]:
            if not text.startswith("/"):
                return []
            providers = tuple(config.providers) if config is not None else ()
            mcp_servers = tuple(mcp_manager.status()) if mcp_manager is not None else ()
            skill_names = tuple(skill_registry.skills) if skill_registry is not None else ()
            return _slash_completion_candidates(
                text,
                commands,
                providers=providers,
                mcp_servers=mcp_servers,
                skills=skill_names,
            )

        def mention_candidates(text: str) -> list[str]:
            word = _completion_word(text)
            if not word.startswith("@"):
                return []
            candidates: list[str] = []
            if mcp_manager is not None:
                for value, _display, _description in mcp_manager.resource_mentions():
                    if value.startswith(word):
                        candidates.append(value)
                    if len(candidates) >= 50:
                        return candidates
            remaining = 50 - len(candidates)
            candidates.extend(
                _local_path_completion_candidates(
                    paths.workspace,
                    word,
                    max_results=remaining,
                )
            )
            return list(dict.fromkeys(candidates))

        class KairoCompleter(Completer):
            def get_completions(self, document: Any, complete_event: Any) -> Any:
                return iter(())

        class KairoLexer(Lexer):
            def lex_document(self, document: Any) -> Any:
                def get_line(line_number: int) -> FormattedText:
                    return FormattedText(_highlight_input_line(document.lines[line_number]))

                return get_line

        class SafeFileHistory(FileHistory):
            def load_history_strings(self) -> list[str]:
                return _load_input_history(Path(os.fsdecode(self.filename)))

            def store_string(self, string: str) -> None:
                _append_input_history(Path(os.fsdecode(self.filename)), string)

            def append_string(self, string: str) -> None:
                if _is_sensitive_history_input(string):
                    return
                super().append_string(string)

        try:
            _prepare_input_history(paths.history_file)
            history: Any = SafeFileHistory(str(paths.history_file))
        except (OSError, ValueError):
            history = InMemoryHistory()
        prompt_output = create_output()
        if isinstance(prompt_output, Vt100_Output):
            prompt_output.enable_cpr = False
            mutable_output: Any = prompt_output
            mutable_output.erase_down = lambda: _erase_with_default_background(
                prompt_output, "\x1b[J"
            )
            mutable_output.erase_end_of_line = lambda: _erase_with_default_background(
                prompt_output, "\x1b[K"
            )
            mutable_output.erase_screen = lambda: _erase_with_default_background(
                prompt_output, "\x1b[2J"
            )

        session: Any = None
        menu_query = ""
        menu_selected_index = 0

        def completion_menu_state() -> tuple[str, str, list[str], int]:
            nonlocal menu_query, menu_selected_index
            if session is None:
                return "", "", [], 0
            text = session.default_buffer.document.text_before_cursor
            if text.startswith("/"):
                kind = "slash"
                candidates = slash_candidates(text)
            else:
                kind = "mention"
                candidates = mention_candidates(text)
            if text != menu_query:
                menu_query = text
                menu_selected_index = 0
            if candidates:
                menu_selected_index = min(menu_selected_index, len(candidates) - 1)
            else:
                menu_selected_index = 0
            return kind, text, candidates, menu_selected_index

        def completion_menu_is_visible() -> bool:
            return bool(completion_menu_state()[2])

        def multiline_input_navigation() -> bool:
            return bool(
                session is not None
                and session.default_buffer.document.line_count > 1
                and not completion_menu_is_visible()
            )

        def composer_overflows() -> bool:
            return bool(
                session is not None and session.default_buffer.document.line_count > 7
            )

        def selected_completion_candidate() -> str | None:
            _, _, candidates, selected = completion_menu_state()
            return candidates[selected] if candidates else None

        bindings = KeyBindings()

        def input_is_blank() -> bool:
            return bool(
                session is not None
                and _is_blank_interactive_submission(session.default_buffer.text)
            )

        @bindings.add("enter", filter=Condition(input_is_blank), eager=True)
        def ignore_blank_submission(_event: Any) -> None:
            return None

        @bindings.add("c-j")
        @bindings.add("escape", "enter")
        def insert_newline(event: Any) -> None:
            event.current_buffer.insert_text("\n")

        @bindings.add("c-o")
        def toggle_details(event: Any) -> None:
            toggled = bool(index_display is not None and index_display.toggle())
            if not toggled and thought_display is not None:
                toggled = thought_display.toggle()
            if toggled:
                event.app.invalidate()

        def details_are_expanded() -> bool:
            return bool(
                (index_display is not None and index_display.expanded)
                or (thought_display is not None and thought_display.expanded)
            )

        def details_are_available() -> bool:
            return bool(
                (index_display is not None and index_display.finished)
                or (thought_display is not None and thought_display.finished)
            )

        def index_tree_is_expanded() -> bool:
            return bool(index_display is not None and index_display.expanded)

        def index_tree_page_size() -> int:
            rows = prompt_output.get_size().rows
            composer_rows = min(
                max(len(session.default_buffer.document.lines) + 1, 2), 8
            )
            return max(1, rows - composer_rows - 6)

        @bindings.add("pageup", filter=Condition(index_tree_is_expanded), eager=True)
        def page_index_tree_up(event: Any) -> None:
            page_size = index_tree_page_size()
            if index_display is not None and index_display.scroll(-page_size, page_size):
                event.app.invalidate()

        @bindings.add("pagedown", filter=Condition(index_tree_is_expanded), eager=True)
        def page_index_tree_down(event: Any) -> None:
            page_size = index_tree_page_size()
            if index_display is not None and index_display.scroll(page_size, page_size):
                event.app.invalidate()

        @bindings.add("up", filter=Condition(index_tree_is_expanded), eager=True)
        def scroll_index_tree_up(event: Any) -> None:
            if index_display is not None and index_display.scroll(
                -1, index_tree_page_size()
            ):
                event.app.invalidate()

        @bindings.add("down", filter=Condition(index_tree_is_expanded), eager=True)
        def scroll_index_tree_down(event: Any) -> None:
            if index_display is not None and index_display.scroll(
                1, index_tree_page_size()
            ):
                event.app.invalidate()

        @bindings.add("escape", filter=Condition(details_are_expanded))
        def collapse_details(event: Any) -> None:
            if index_display is not None:
                index_display.expanded = False
            if thought_display is not None:
                thought_display.expanded = False
            event.app.invalidate()

        @bindings.add("enter", filter=Condition(details_are_expanded))
        def submit_with_details_collapsed(event: Any) -> None:
            if index_display is not None:
                index_display.expanded = False
            if thought_display is not None:
                thought_display.expanded = False
            event.current_buffer.validate_and_handle()

        @bindings.add("down", filter=Condition(completion_menu_is_visible))
        @bindings.add("c-n", filter=Condition(completion_menu_is_visible))
        def select_next_completion(event: Any) -> None:
            nonlocal menu_selected_index
            _, _, candidates, selected = completion_menu_state()
            menu_selected_index = min(selected + 1, len(candidates) - 1)
            event.app.invalidate()

        @bindings.add("up", filter=Condition(completion_menu_is_visible))
        @bindings.add("c-p", filter=Condition(completion_menu_is_visible))
        def select_previous_completion(event: Any) -> None:
            nonlocal menu_selected_index
            _, _, _, selected = completion_menu_state()
            menu_selected_index = max(0, selected - 1)
            event.app.invalidate()

        @bindings.add("up", filter=Condition(multiline_input_navigation))
        def move_cursor_up_in_composer(event: Any) -> None:
            event.current_buffer.cursor_up()

        @bindings.add("down", filter=Condition(multiline_input_navigation))
        def move_cursor_down_in_composer(event: Any) -> None:
            event.current_buffer.cursor_down()

        @bindings.add("pageup", filter=Condition(multiline_input_navigation))
        def move_cursor_page_up_in_composer(event: Any) -> None:
            event.current_buffer.cursor_up(count=7)

        @bindings.add("pagedown", filter=Condition(multiline_input_navigation))
        def move_cursor_page_down_in_composer(event: Any) -> None:
            event.current_buffer.cursor_down(count=7)

        def completion_candidate_can_be_inserted() -> bool:
            candidate = selected_completion_candidate()
            if candidate is None or session is None:
                return False
            kind, text, _, _ = completion_menu_state()
            current = text if kind == "slash" else _completion_word(text)
            return candidate != current

        def insert_selected_completion(event: Any) -> None:
            candidate = selected_completion_candidate()
            if candidate is None:
                return
            if thought_display is not None:
                thought_display.expanded = False
            if index_display is not None:
                index_display.expanded = False
            kind, _, _, _ = completion_menu_state()
            if kind == "slash":
                event.current_buffer.document = Document(candidate, cursor_position=len(candidate))
                return
            document = event.current_buffer.document
            word = _completion_word(document.text_before_cursor)
            start = document.cursor_position - len(word)
            value = document.text[:start] + candidate + document.text[document.cursor_position :]
            event.current_buffer.document = Document(
                value,
                cursor_position=start + len(candidate),
            )

        @bindings.add("tab", filter=Condition(completion_menu_is_visible))
        def accept_completion_with_tab(event: Any) -> None:
            insert_selected_completion(event)

        @bindings.add("enter", filter=Condition(completion_candidate_can_be_inserted))
        def accept_completion_with_enter(event: Any) -> None:
            insert_selected_completion(event)

        session = PromptSession(
            message=FormattedText(
                [
                    ("class:composer.input", "\n"),
                    ("class:composer.prompt", "› "),
                ]
            ),
            history=history,
            completer=KairoCompleter(),
            key_bindings=bindings,
            lexer=KairoLexer(),
            prompt_continuation=FormattedText([("class:composer.input", "  ")]),
            complete_while_typing=True,
            reserve_space_for_menu=0,
            output=prompt_output,
            mouse_support=Condition(composer_overflows),
            color_depth=ColorDepth.TRUE_COLOR,
            style=Style.from_dict(
                {
                    "": "fg:#111111 bg:default",
                    "composer.input": "fg:#cecdc3 bg:#303030",
                    "composer.prompt": "fg:#cecdc3 bg:#303030 bold",
                    "plan-review.input": "fg:#111111 bg:#eaf5f8",
                    "plan-review.marker": "fg:#007a9f bg:#eaf5f8 bold",
                    "plan-review.title": "fg:#005f7a bg:#eaf5f8 bold",
                    "plan-review.key": "fg:#007a9f bg:#eaf5f8 bold",
                    "plan-review.hint": "fg:#66777d bg:#eaf5f8",
                    "plan-review.prompt": "fg:#007a9f bg:#eaf5f8 bold",
                    "approval.input": "fg:#111111 bg:#fff8e6",
                    "approval.marker": "fg:#a66b00 bg:#fff8e6 bold",
                    "approval.prompt": "fg:#6b4b00 bg:#fff8e6 bold",
                    "wechat.input": "fg:#17332b bg:#eef8f4",
                    "wechat.marker": "fg:#07a85a bg:#eef8f4 bold",
                    "wechat.title": "fg:#087f49 bg:#eef8f4 bold",
                    "wechat.label": "fg:#527066 bg:#eef8f4",
                    "wechat.path": "fg:#17332b bg:#eef8f4 bold",
                    "wechat.hint": "fg:#668078 bg:#eef8f4",
                    "wechat.prompt": "fg:#07a85a bg:#eef8f4 bold",
                    "bottom-toolbar": "fg:#888888 bg:default noreverse",
                    "thought": "fg:#888888 bg:default",
                    "answer.prefix": "fg:#888888 bg:default",
                    "slash-menu.command": "fg:#cecdc3 bg:default",
                    "slash-menu.description": "fg:#888888 bg:default",
                    "slash-menu.current": "fg:#007a9f bg:default bold",
                    "completion-menu": "bg:default",
                    "completion-menu.completion": "fg:#111111 bg:default",
                    "completion-menu.completion.current": ("fg:#006d91 bg:default bold"),
                    "completion-menu.meta.completion": "fg:#888888 bg:default",
                    "completion-menu.meta.completion.current": ("fg:#006d91 bg:default bold"),
                }
            ),
        )
        normal_container = session.app.layout.container
        prompt_container = normal_container.children[0]
        prompt_container.alternative_content.style = "class:composer.input"

        def render_completed_turn() -> FormattedText:
            if index_display is not None and index_display.finished:
                index_fragments = [
                    ("", "\n"),
                    ("class:answer.prefix", "● "),
                    ("", index_display.summary() + "\n"),
                ]
                if index_display.expanded and index_display.tree:
                    tree = "\n".join(
                        "  " + line
                        for line in index_display.visible_tree(
                            index_tree_page_size()
                        ).splitlines()
                    )
                    index_fragments.append(("class:thought", "\n" + tree + "\n"))
                return FormattedText(index_fragments)
            if thought_display is None or not thought_display.finished:
                return FormattedText([])
            latest = thought_display.turns[-1] if thought_display.turns else None
            answer_streamed = bool(latest is not None and latest.answer_streamed)
            fragments: list[tuple[str, str]] = []
            if not answer_streamed or thought_display.expanded:
                fragments.append(("class:thought", "\n  " + thought_display.summary() + "\n"))
            if thought_display.expanded:
                details = (
                    thought_display.details or "No model reasoning or tool activity was returned."
                )
                fragments.append(
                    (
                        "class:thought",
                        "\n" + "\n".join("  " + line for line in details.splitlines()) + "\n",
                    )
                )
            if latest is not None and latest.answer and not latest.answer_streamed:
                rendered_answer = _render_interactive_answer(
                    latest.answer, renderer, _terminal_columns()
                )
                fragments.extend(
                    [
                        ("", "\n"),
                        ("class:answer.prefix", "● "),
                        ("", rendered_answer + "\n"),
                    ]
                )
            return FormattedText(fragments)

        def completion_menu_height() -> Dimension:
            return Dimension.exact(min(len(completion_menu_state()[2]), 8))

        def render_completion_menu() -> FormattedText:
            kind, _, candidates, selected = completion_menu_state()
            start = max(0, min(selected - 7, len(candidates) - 8))
            visible = candidates[start : start + 8]
            command_width = max(16, *(len(value) + 2 for value in visible))
            fragments: list[tuple[str, str]] = []
            for offset, value in enumerate(visible):
                is_current = start + offset == selected
                command_style = (
                    "class:slash-menu.current" if is_current else "class:slash-menu.command"
                )
                if kind == "slash":
                    normalized_value = value.casefold()
                    command = normalized_value.split(" ", 1)[0]
                    description = SLASH_SUBCOMMAND_DESCRIPTIONS.get(
                        normalized_value,
                        SLASH_COMMAND_DESCRIPTIONS.get(
                            normalized_value,
                            SLASH_COMMAND_DESCRIPTIONS.get(command, ""),
                        ),
                    )
                    description_style = (
                        "class:slash-menu.current" if is_current else "class:slash-menu.description"
                    )
                    fragments.append((command_style, f"  {value:<{command_width}}"))
                    fragments.append((description_style, description))
                else:
                    fragments.append((command_style, f"  {value}"))
                if offset + 1 < len(visible):
                    fragments.append(("", "\n"))
            return FormattedText(fragments)

        def composer_height() -> Dimension:
            line_count = len(session.default_buffer.document.lines)
            return Dimension.exact(min(max(line_count + 1, 2), 8))

        session.app.layout.current_window.height = composer_height
        session.app.layout.current_window.style = "class:composer.input"
        prompt_container.alternative_content.content.children.insert(
            0,
            ConditionalContainer(
                Window(
                    FormattedTextControl(render_completed_turn),
                    style="bg:default",
                    wrap_lines=True,
                    dont_extend_height=True,
                    always_hide_cursor=True,
                    get_line_prefix=lambda _line, wrap_count: (
                        FormattedText([("class:thought", "  ")])
                        if wrap_count
                        else FormattedText([])
                    ),
                ),
                filter=Condition(details_are_available),
            ),
        )
        prompt_container.alternative_content.content.children.append(
            ConditionalContainer(
                Window(
                    FormattedTextControl(render_completion_menu),
                    height=completion_menu_height,
                    style="bg:default",
                    dont_extend_height=True,
                ),
                filter=Condition(completion_menu_is_visible),
            )
        )
        normal_container.children.append(
            ConditionalContainer(
                Window(
                    FormattedTextControl(
                        lambda: (
                            "  " + str(bottom_toolbar()) + " " if callable(bottom_toolbar) else ""
                        )
                    ),
                    height=Dimension.exact(1),
                    style="class:bottom-toolbar",
                    dont_extend_height=True,
                ),
                filter=Condition(lambda: not completion_menu_is_visible()),
            )
        )
        return session
    except ImportError:
        return None


async def _read_input(session: Any) -> str:
    if session is None:
        return await asyncio.to_thread(input, "❯ ")
    output = session.app.output
    try:
        # Ask modern terminals to preserve modifier information for keys that
        # would otherwise collapse to the same control byte. xterm/iTerm-style
        # terminals use modifyOtherKeys; Kitty-compatible terminals use CSI-u.
        output.write_raw("\x1b[>4;1m\x1b[>1u")
        output.flush()
        return str(await session.prompt_async())
    finally:
        # Both protocols are stack/reset based. Always restore them, including
        # cancellation and EOF paths, so the parent shell keeps its key setup.
        output.write_raw("\x1b[<u\x1b[>4m")
        output.flush()


async def _read_approval_input(session: Any, prompt: str = "Decision  › ") -> str:
    """Read approval input without sharing ownership with the main composer."""

    if session is None:
        return await asyncio.to_thread(input, prompt)

    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import DummyCompleter
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Dimension

    bindings = KeyBindings()

    @bindings.add("escape")
    @bindings.add("c-c")
    def reject_approval(event: Any) -> None:
        event.app.exit(result="n")

    approval_session: Any = PromptSession(
        message=FormattedText(
            [
                ("class:approval.marker", "  ● "),
                ("class:approval.prompt", prompt),
            ]
        ),
        key_bindings=bindings,
        completer=DummyCompleter(),
        complete_while_typing=False,
        reserve_space_for_menu=0,
        history=InMemoryHistory(),
        bottom_toolbar="",
        input=session.app.input,
        output=session.app.output,
        style=session.style,
        color_depth=session.color_depth,
    )
    approval_session.app.layout.current_window.style = "class:approval.input"
    approval_session.app.layout.current_window.height = Dimension.exact(1)
    approval_container = approval_session.app.layout.container.children[0]
    approval_container.alternative_content.style = "class:approval.input"
    return str(await approval_session.prompt_async())


async def _read_wechat_workspace_input(session: Any, default: Path) -> str:
    """Read a workspace path in a compact block matching the main composer."""

    if session is None:
        return await asyncio.to_thread(input, workspace_prompt(default))

    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import DummyCompleter
    from prompt_toolkit.filters import to_filter
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Dimension
    from prompt_toolkit.layout.containers import Window
    from prompt_toolkit.layout.controls import FormattedTextControl

    bindings = KeyBindings()

    @bindings.add("escape")
    @bindings.add("c-c")
    def cancel_setup(event: Any) -> None:
        event.app.exit(result="\x1b")

    workspace_session: Any = PromptSession(
        message=FormattedText(
            [
                ("class:wechat.prompt", "  › "),
            ]
        ),
        key_bindings=bindings,
        completer=DummyCompleter(),
        complete_while_typing=False,
        reserve_space_for_menu=0,
        history=InMemoryHistory(),
        bottom_toolbar="",
        input=session.app.input,
        output=session.app.output,
        style=session.style,
        color_depth=session.color_depth,
    )
    workspace_session.app.layout.current_window.style = "class:wechat.input"
    workspace_session.app.layout.current_window.height = Dimension.exact(1)
    workspace_session.app.layout.current_window.dont_extend_height = to_filter(True)
    workspace_session.app.ttimeoutlen = 0.05
    workspace_container = workspace_session.app.layout.container.children[0]
    workspace_container.alternative_content.style = "bg:default"
    for content in (
        workspace_container.content,
        workspace_container.alternative_content.content,
    ):
        content.children[0:0] = [
            Window(
                height=Dimension.exact(1),
                style="bg:default",
                dont_extend_height=True,
            ),
            Window(
                FormattedTextControl(
                    FormattedText(
                        [
                            ("class:wechat.marker", "● "),
                            ("class:wechat.title", "Connect WeChat\n"),
                            ("class:wechat.label", "  Workspace  "),
                            ("class:wechat.path", f"{default}\n"),
                            (
                                "class:wechat.hint",
                                "  Enter connect · Esc cancel · Type another path",
                            ),
                        ]
                    )
                ),
                height=Dimension.exact(3),
                style="class:wechat.input",
                dont_extend_height=True,
            ),
        ]
    result = str(await workspace_session.prompt_async())
    if result == "\x1b":
        raise _WechatSetupCanceled
    return result


def _is_blank_interactive_submission(value: str) -> bool:
    return not value.strip()


async def _read_plan_review_input(session: Any) -> str:
    """Read a plan decision without mutating the reusable main prompt session."""

    plain_prompt = "Review plan · Enter run · Esc cancel · Type to revise  › "
    if session is None:
        return await asyncio.to_thread(input, plain_prompt)

    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import DummyCompleter
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Dimension

    bindings = KeyBindings()

    @bindings.add("escape")
    @bindings.add("c-c")
    def cancel_review(event: Any) -> None:
        event.app.exit(result="\x1b")

    prompt = FormattedText(
        [
            ("class:plan-review.input", "\n"),
            ("class:plan-review.marker", "● "),
            ("class:plan-review.title", "Review plan"),
            ("class:plan-review.key", "  Enter"),
            ("class:plan-review.hint", " run  ·  "),
            ("class:plan-review.key", "Esc"),
            ("class:plan-review.hint", " cancel  ·  Type to revise"),
            ("class:plan-review.prompt", "  › "),
        ]
    )
    review_session: Any = PromptSession(
        message=prompt,
        key_bindings=bindings,
        completer=DummyCompleter(),
        complete_while_typing=False,
        reserve_space_for_menu=0,
        history=InMemoryHistory(),
        bottom_toolbar="",
        input=session.app.input,
        output=session.app.output,
        style=session.style,
        color_depth=session.color_depth,
    )
    review_session.app.layout.current_window.style = "class:plan-review.input"
    review_session.app.layout.current_window.height = Dimension.exact(2)
    review_container = review_session.app.layout.container.children[0]
    review_container.alternative_content.style = "class:plan-review.input"
    return str(await review_session.prompt_async())
