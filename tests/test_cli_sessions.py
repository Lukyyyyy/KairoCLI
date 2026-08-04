import asyncio
import importlib
import io
import multiprocessing
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

import kairocli.cli.history as input_history_module
from kairocli.agent import Agent
from kairocli.cli import (
    _append_input_history,
    _await_shutdown,
    _clear_input_history,
    _completion_word,
    _console,
    _erase_with_default_background,
    _handle_session_command,
    _highlight_input_line,
    _is_sensitive_history_input,
    _load_input_history,
    _local_path_completion_candidates,
    _print_answer_prefix,
    _print_command_output,
    _prompt_session,
    _read_input,
    _resume_session_hint,
    _session_startup_notice,
    _slash_completion_candidates,
    _StreamingAnswerDisplay,
    _terminal_background_block,
    _welcome_lines,
    _working_status_text,
    build_parser,
)
from kairocli.config import AppConfig
from kairocli.llm import LlmClient
from kairocli.models import LlmResponse, Message
from kairocli.paths import KairoPaths
from kairocli.plan import ExecutionPlan, PlanTask
from kairocli.sessions import SessionStore
from kairocli.thought_display import ThoughtDisplay
from kairocli.tools import ToolRegistry

cli_module = importlib.import_module("kairocli.cli.interactive")


class SessionClient(LlmClient):
    provider = "glm"
    model = "session-model"

    async def complete(
        self, messages: list[Message], tools: list[dict[str, object]] | None = None
    ) -> LlmResponse:
        return LlmResponse(content="ok")


class RecordingConsole:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def print(self, message: str) -> None:
        self.messages.append(message)


async def test_shutdown_defers_cancellation_until_cleanup_finishes() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    cleaned = asyncio.Event()

    async def cleanup() -> list[str]:
        started.set()
        await release.wait()
        cleaned.set()
        return ["finished"]

    shutdown_task = asyncio.create_task(cleanup())
    waiter = asyncio.create_task(_await_shutdown(shutdown_task))
    await started.wait()
    waiter.cancel()
    await asyncio.sleep(0)

    assert not waiter.done()
    assert not cleaned.is_set()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert cleaned.is_set()
    assert shutdown_task.result() == ["finished"]


async def test_interactive_startup_failure_closes_every_started_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    config = AppConfig.load(paths)
    agent = Agent(SessionClient(), ToolRegistry(workspace), "system")
    closed: list[str] = []

    class DisabledSnapshotService:
        class Config:
            enabled = False

        config = Config()

        async def close(self) -> None:
            return None

    agent.tools.snapshot_service = DisabledSnapshotService()  # type: ignore[assignment]
    original_tools_close = agent.tools.close

    async def close_tools() -> None:
        closed.append("tools")
        await original_tools_close()

    agent.tools.close = close_tools  # type: ignore[method-assign]

    class FakeMcpManager:
        def __init__(self, *_args: object) -> None:
            return None

        async def start_all(self) -> None:
            return None

        def resource_index(self) -> str:
            return ""

        async def close(self) -> None:
            closed.append("mcp")

    class FakeTaskManager:
        def __init__(self, *_args: object) -> None:
            return None

        def start(self) -> None:
            return None

        async def close(self) -> None:
            closed.append("task")

    monkeypatch.setattr(cli_module, "make_agent", lambda *_args, **_kwargs: agent)
    monkeypatch.setattr(cli_module, "McpServerManager", FakeMcpManager)
    monkeypatch.setattr(cli_module, "DurableTaskManager", FakeTaskManager)
    monkeypatch.setattr(
        cli_module,
        "_prompt_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("prompt failed")),
    )

    with pytest.raises(RuntimeError, match="prompt failed"):
        await cli_module.interactive(paths, config)

    assert set(closed) == {"mcp", "task", "tools"}


def _append_history_worker(path: str, prefix: str, count: int) -> None:
    history = Path(path)
    for index in range(count):
        if not _append_input_history(history, f"{prefix}-{index:03d}"):
            raise RuntimeError("history append failed")


async def test_session_command_switches_without_losing_histories(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    store = SessionStore(paths.session_database)
    first = store.create(workspace, "glm", "session-model")
    agent = Agent(SessionClient(), ToolRegistry(workspace), "system")
    agent.history = [Message("user", "first conversation")]
    console = RecordingConsole()

    second_id = await _handle_session_command("new", store, first.meta.id, paths, agent, console)
    assert second_id != first.meta.id
    assert agent.history == []
    agent.history = [Message("user", "second conversation")]

    resumed_id = await _handle_session_command(
        f"resume {first.meta.id}", store, second_id, paths, agent, console
    )

    assert resumed_id == first.meta.id
    assert [message.content for message in agent.history] == ["first conversation"]
    second = store.load(second_id, workspace)
    assert second is not None
    assert [message.content for message in second.messages] == ["second conversation"]


async def test_session_list_supports_default_filter_and_all_flag(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    store = SessionStore(paths.session_database)
    current = store.create(workspace, "glm", "session-model")
    other = store.create(workspace, "glm", "session-model")
    agent = Agent(SessionClient(), ToolRegistry(workspace), "system")
    console = RecordingConsole()

    await _handle_session_command("list", store, current.meta.id, paths, agent, console)
    assert current.meta.id in console.messages[-1]
    assert other.meta.id not in console.messages[-1]
    assert "/session list --all" in console.messages[-1]

    await _handle_session_command("list --all", store, current.meta.id, paths, agent, console)
    assert current.meta.id in console.messages[-1]
    assert other.meta.id in console.messages[-1]

    await _handle_session_command("list --unknown", store, current.meta.id, paths, agent, console)
    assert console.messages[-1] == "Usage: /session list [--all]"


async def test_session_delete_supports_empty_cleanup_and_multiple_ids(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    store = SessionStore(paths.session_database)
    current = store.create(workspace, "glm", "session-model")
    first_empty = store.create(workspace, "glm", "session-model")
    second_empty = store.create(workspace, "glm", "session-model")
    first_saved = store.create(workspace, "glm", "session-model")
    second_saved = store.create(workspace, "glm", "session-model")
    store.save(
        first_saved.meta.id,
        workspace,
        "glm",
        "session-model",
        [Message("user", "first")],
    )
    store.save(
        second_saved.meta.id,
        workspace,
        "glm",
        "session-model",
        [Message("user", "second")],
    )
    agent = Agent(SessionClient(), ToolRegistry(workspace), "system")
    console = RecordingConsole()

    await _handle_session_command("delete --empty", store, current.meta.id, paths, agent, console)
    assert console.messages[-1] == "Deleted 2 empty sessions."
    assert store.load(current.meta.id, workspace) is not None
    assert store.load(first_empty.meta.id, workspace) is None
    assert store.load(second_empty.meta.id, workspace) is None

    await _handle_session_command(
        f"delete {first_saved.meta.id} {second_saved.meta.id}",
        store,
        current.meta.id,
        paths,
        agent,
        console,
    )
    assert console.messages[-1] == "Deleted 2 sessions."
    assert store.load(first_saved.meta.id, workspace) is None
    assert store.load(second_saved.meta.id, workspace) is None


async def test_session_batch_delete_rejects_active_and_invalid_ids_without_deleting(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    store = SessionStore(paths.session_database)
    current = store.create(workspace, "glm", "session-model")
    saved = store.create(workspace, "glm", "session-model")
    agent = Agent(SessionClient(), ToolRegistry(workspace), "system")
    console = RecordingConsole()

    await _handle_session_command(
        f"delete {saved.meta.id} {current.meta.id}",
        store,
        current.meta.id,
        paths,
        agent,
        console,
    )
    assert console.messages[-1].startswith("Cannot delete the active session")
    assert store.load(saved.meta.id, workspace) is not None

    await _handle_session_command(
        f"delete {saved.meta.id} invalid",
        store,
        current.meta.id,
        paths,
        agent,
        console,
    )
    assert console.messages[-1] == "Invalid session ID"
    assert store.load(saved.meta.id, workspace) is not None


def test_resume_flags_are_mutually_exclusive() -> None:
    parser = build_parser()
    assert parser.parse_args(["--continue"]).continue_session is True
    assert parser.parse_args(["--resume", "session_123456789abc"]).resume == (
        "session_123456789abc"
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["--continue", "--resume", "session_123456789abc"])


@pytest.mark.parametrize(
    "value",
    [
        "GLM_API_KEY=secret",
        "Authorization: Bearer secret",
        "/config provider glm --api-key secret",
        "/config provider glm --api_key secret",
        "curl https://user:password@example.test/private",
        "https://example.test/?access_token=secret",
        "sk-proj-abcdefghijklmnopqrstuvwx",
        "github_pat_abcdefghijklmnopqrstuvwxyz",
        "sk_live_abcdefghijklmnop",
        "hf_abcdefghijklmnopqrstuvwxyz",
        "eyJabcdefghijk.abcdefghijklmnop.signature123",
        "-----BEGIN PRIVATE KEY-----",
        "AKIAABCDEFGHIJKLMNOP",
        "@image:data:image/png;base64," + "A" * 260,
    ],
)
def test_sensitive_input_is_excluded_from_history(value: str) -> None:
    assert _is_sensitive_history_input(value) is True
    assert _is_sensitive_history_input("read src/tokenizer.py") is False


def test_prompt_history_does_not_persist_secrets(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    session = _prompt_session(paths, lambda: "idle")
    assert session is not None
    session.history.append_string("normal request")
    session.history.append_string("API_KEY=should-not-persist")

    text = paths.history_file.read_text(encoding="utf-8")
    assert "normal request" in text
    assert "should-not-persist" not in text
    if os.name == "posix":
        assert stat.S_IMODE(paths.history_file.stat().st_mode) == 0o600
        assert stat.S_IMODE(paths.history_file.parent.stat().st_mode) == 0o700


def test_prompt_uses_gray_block_composer(tmp_path: Path) -> None:
    workspace = tmp_path / "KairoCLI"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")

    session = _prompt_session(paths, lambda: "ReAct · idle")

    assert session is not None
    message = session.message
    assert "›" in str(message)
    assert "─" not in str(message)
    assert any(fragment[1] == "\n" for fragment in message)
    assert session.reserve_space_for_menu == 0
    assert session.rprompt is None
    assert session.bottom_toolbar is None
    continuation = session._get_continuation(2, 1, 0)
    assert "".join(fragment[1] for fragment in continuation) == "  "
    assert session.app.color_depth.value == "DEPTH_24_BIT"
    assert session.app.output.responds_to_cpr is False
    prompt_container = session.app.layout.container.children[0]
    assert prompt_container.alternative_content.style == "class:composer.input"
    composer_height = session.app.layout.current_window.height
    assert callable(composer_height)
    assert composer_height().min == 2
    assert composer_height().max == 2
    session.default_buffer._set_text("one\ntwo\nthree\nfour\nfive")
    assert composer_height().min == 6
    assert composer_height().max == 6
    session.default_buffer._set_text("one")
    assert composer_height().min == 2
    assert composer_height().max == 2
    session.default_buffer._set_text("\n".join(str(index) for index in range(10)))
    assert composer_height().min == 8
    assert composer_height().max == 8
    session.default_buffer._set_text("")
    assert session.app.layout.current_window.style == "class:composer.input"
    menu_space = prompt_container.alternative_content.content.children[-1]
    assert menu_space.filter() is False
    status = session.app.layout.container.children[-1]
    assert status.filter() is True
    assert status.content.height.min == 1
    assert status.content.height.max == 1
    assert status.content.style == "class:bottom-toolbar"
    assert status.content.content.text() == "  ReAct · idle "

    session.default_buffer._set_text("/")
    session.default_buffer._set_cursor_position(1)
    assert menu_space.filter() is True
    menu_height = menu_space.content.height()
    assert menu_height.min == 8
    assert menu_height.max == 8
    assert menu_space.content.style == "bg:default"
    assert status.filter() is False
    rules = str(session.style.style_rules)
    assert "composer.input" in rules and "bg:#f1f1f1" in rules
    assert "bottom-toolbar" in rules and "bg:default" in rules
    assert "plan-review.input" in rules and "bg:#eaf5f8" in rules
    assert "wechat.input" in rules and "bg:#f1f1f1" in rules


def test_prompt_bottom_status_updates_when_plan_mode_is_armed(tmp_path: Path) -> None:
    workspace = tmp_path / "KairoCLI"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    mode = ["agent"]
    session = _prompt_session(
        paths,
        lambda: cli_module._mode_status_line(mode[0], "deepseek/model"),
    )

    assert session is not None
    status = session.app.layout.container.children[-1]
    assert status.content.content.text() == "  ReAct · deepseek/model "

    mode[0] = "plan"
    assert status.content.content.text() == "  Plan · deepseek/model "


def test_multiline_composer_supports_keyboard_and_mouse_scrolling(tmp_path: Path) -> None:
    workspace = tmp_path / "KairoCLI"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    session = _prompt_session(paths, lambda: "idle")
    buffer = session.default_buffer

    buffer._set_text("\n".join(str(index) for index in range(10)))
    buffer._set_cursor_position(len(buffer.text))
    assert session.mouse_support() is True

    def active_binding(key: str):
        return next(
            binding
            for binding in session.key_bindings.get_bindings_for_keys((key,))
            if binding.filter()
        )

    event = SimpleNamespace(current_buffer=buffer)
    active_binding("up").handler(event)
    assert buffer.document.cursor_position_row == 8
    active_binding("pageup").handler(event)
    assert buffer.document.cursor_position_row == 1
    active_binding("pagedown").handler(event)
    assert buffer.document.cursor_position_row == 8
    active_binding("down").handler(event)
    assert buffer.document.cursor_position_row == 9

    buffer._set_text("one line")
    buffer._set_cursor_position(len(buffer.text))
    assert session.mouse_support() is False


def test_prompt_ctrl_o_toggles_the_latest_thought(tmp_path: Path) -> None:
    workspace = tmp_path / "KairoCLI"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    thought = ThoughtDisplay(lambda: 1.0)
    thought.start()
    thought.add("tool detail")
    thought.finish("formal answer")
    session = _prompt_session(paths, lambda: "idle", thought_display=thought)

    class App:
        invalidations = 0

        def invalidate(self) -> None:
            self.invalidations += 1

    class Event:
        app = App()

    binding = session.key_bindings.get_bindings_for_keys(("c-o",))[0]
    binding.handler(Event())

    assert thought.expanded is True
    thought_panel = session.app.layout.container.children[0].alternative_content.content.children[0]
    assert thought_panel.filter() is True
    assert thought_panel.content.style == "bg:default"
    expanded_text = "".join(fragment[1] for fragment in thought_panel.content.content.text())
    assert expanded_text.startswith("\n  Thought for 1s (ctrl+o to collapse)\n")
    assert "Thought for 1s (ctrl+o to collapse)\n\n  tool detail\n" in expanded_text
    assert expanded_text.index("Thought for") < expanded_text.index("tool detail")
    assert expanded_text.index("tool detail") < expanded_text.index("formal answer")

    binding.handler(Event())

    assert thought.expanded is False
    assert thought_panel.filter() is True
    collapsed_text = "".join(fragment[1] for fragment in thought_panel.content.content.text())
    assert "tool detail" not in collapsed_text
    assert Event.app.invalidations == 2


def test_prompt_does_not_repeat_an_answer_already_streamed(tmp_path: Path) -> None:
    workspace = tmp_path / "KairoCLI"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    thought = ThoughtDisplay(lambda: 1.0)
    thought.start()
    thought.add("tool detail")
    thought.finish("streamed answer", answer_streamed=True)
    session = _prompt_session(paths, lambda: "idle", thought_display=thought)

    thought_panel = session.app.layout.container.children[0].alternative_content.content.children[0]
    rendered = "".join(fragment[1] for fragment in thought_panel.content.content.text())

    assert "Thought for 1s" not in rendered
    assert "tool detail" not in rendered
    assert "streamed answer" not in rendered

    assert thought.toggle() is True
    expanded = "".join(fragment[1] for fragment in thought_panel.content.content.text())
    assert "Thought for 1s (ctrl+o to collapse)" in expanded
    assert "tool detail" in expanded
    assert expanded.index("Thought for 1s") < expanded.index("tool detail")
    assert "streamed answer" not in expanded


async def test_prompt_ctrl_o_inline_round_trip_restores_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prompt_toolkit.output
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    workspace = tmp_path / "KairoCLI"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    thought = ThoughtDisplay(lambda: 1.0)
    thought.start("hello")
    thought.add("reasoning")
    thought.finish("answer")
    output = DummyOutput()
    monkeypatch.setattr(prompt_toolkit.output, "create_output", lambda: output)

    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=output):
            session = _prompt_session(paths, lambda: "idle", thought_display=thought)
            read_task = asyncio.create_task(session.prompt_async())
            await asyncio.sleep(0.02)
            pipe_input.send_bytes(b"\x0f")
            for _ in range(20):
                if thought.expanded:
                    break
                await asyncio.sleep(0.01)
            assert thought.expanded is True
            pipe_input.send_bytes(b"\x0f")
            for _ in range(20):
                if not thought.expanded:
                    break
                await asyncio.sleep(0.01)
            assert thought.expanded is False
            pipe_input.send_text("restored\r")
            assert await asyncio.wait_for(read_task, 1.0) == "restored"


def test_prompt_slash_menu_shows_commands_with_short_descriptions(tmp_path: Path) -> None:
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document

    workspace = tmp_path / "KairoCLI"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    session = _prompt_session(paths, lambda: "idle")

    assert session is not None
    native_completions = list(session.completer.get_completions(Document("/"), CompleteEvent()))
    assert native_completions == []

    session.default_buffer._set_text("/")
    session.default_buffer._set_cursor_position(1)
    prompt_container = session.app.layout.container.children[0]
    menu_space = prompt_container.alternative_content.content.children[-1]
    menu_fragments = menu_space.content.content.text()
    menu_text = "".join(fragment[1] for fragment in menu_fragments)
    assert "/help" in menu_text
    assert "command reference" in menu_text
    assert "/model" in menu_text
    assert "active model" in menu_text
    assert session.default_buffer.text == "/"

    session.default_buffer._set_text("/we")
    session.default_buffer._set_cursor_position(len("/we"))
    menu_fragments = menu_space.content.content.text()
    menu_text = "".join(fragment[1] for fragment in menu_fragments)
    assert "/wechat" in menu_text
    assert "/wechat setup" not in menu_text

    session.default_buffer._set_text("/wechat ")
    session.default_buffer._set_cursor_position(len("/wechat "))
    menu_fragments = menu_space.content.content.text()
    menu_text = "".join(fragment[1] for fragment in menu_fragments)
    assert "/wechat setup" in menu_text
    assert "connect a WeChat account" in menu_text
    assert "/wechat status" in menu_text
    assert "connection and channel status" in menu_text


async def test_prompt_mention_menu_shows_paths_without_descriptions_and_inserts_in_place(
    tmp_path: Path,
) -> None:
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document

    workspace = tmp_path / "KairoCLI"
    workspace.mkdir()
    (workspace / "README.md").write_text("demo", encoding="utf-8")
    (workspace / "docs").mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    session = _prompt_session(paths, lambda: "idle")

    assert session is not None
    native_completions = list(session.completer.get_completions(Document("@"), CompleteEvent()))
    assert native_completions == []

    session.default_buffer.document = Document("@", cursor_position=1)
    prompt_container = session.app.layout.container.children[0]
    menu_space = prompt_container.alternative_content.content.children[-1]
    menu_text = "".join(fragment[1] for fragment in menu_space.content.content.text())
    assert "@README.md" in menu_text
    assert "@docs/" in menu_text
    assert "command reference" not in menu_text
    assert menu_space.content.height().min == 2
    assert menu_space.content.height().max == 2

    value = "review @READ after"
    cursor = len("review @READ")
    session.default_buffer.document = Document(value, cursor_position=cursor)
    tab_binding = session.key_bindings.get_bindings_for_keys(("c-i",))[0]
    tab_binding.handler(SimpleNamespace(current_buffer=session.default_buffer))
    await asyncio.sleep(0)

    assert session.default_buffer.text == "review @README.md after"
    assert session.default_buffer.document.cursor_position == len("review @README.md")


def test_terminal_background_block_fills_every_terminal_row() -> None:
    assert _terminal_background_block("Session demo", columns=16) == "Session demo    "
    assert _terminal_background_block("first\nsecond", columns=8) == ("first   \nsecond  ")
    assert _terminal_background_block("", columns=4) == "    "


def test_console_preserves_the_terminals_configured_background() -> None:
    assert _console().style == "on default"


def test_answer_prefix_is_a_gray_solid_circle() -> None:
    class Console:
        _kairo_rich = True

        def __init__(self) -> None:
            self.calls: list[tuple[object, dict[str, object]]] = []

        def print(self, value: object, **kwargs: object) -> None:
            self.calls.append((value, kwargs))

    console = Console()

    _print_answer_prefix(console)

    assert console.calls == [
        (
            "● ",
            {
                "style": "#888888",
                "markup": False,
                "highlight": False,
                "end": "",
            },
        )
    ]


def test_command_output_matches_answer_spacing_and_indentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Console:
        _kairo_rich = True

        def __init__(self) -> None:
            self.calls: list[tuple[object, dict[str, object]]] = []

        def print(self, value: object, **kwargs: object) -> None:
            self.calls.append((value, kwargs))

    streamed: list[str] = []
    monkeypatch.setattr(cli_module, "_write_stream", streamed.append)
    console = Console()

    _print_command_output(console, "Session demo\nNew session")

    assert streamed == ["\n", "\n"]
    assert console.calls == [
        (
            "● ",
            {
                "style": "#888888",
                "markup": False,
                "highlight": False,
                "end": "",
            },
        ),
        (
            "Session demo\n  New session",
            {"markup": False, "highlight": False},
        ),
    ]


def test_plain_streaming_answer_writes_deltas_and_sanitizes_across_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamed: list[str] = []
    monkeypatch.setattr(cli_module, "_write_stream", streamed.append)
    display = _StreamingAnswerDisplay(RecordingConsole(), "plain", 80)

    display.append("hel")
    display.append("lo\x1b")
    display.append("[31m!\nnext")
    display.finish_block()

    assert streamed == ["\n", "● ", "hel", "lo", "!\n  next", "\n"]
    assert display.has_streamed_content is True


def test_plain_streaming_answer_places_thought_before_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streamed: list[str] = []
    monkeypatch.setattr(cli_module, "_write_stream", streamed.append)
    display = _StreamingAnswerDisplay(RecordingConsole(), "plain", 80, lambda: "Thought for 2s")

    display.start_turn()
    display.append("answer")
    display.finish_block()

    assert display.console.messages == ["  Thought for 2s"]
    assert streamed == ["\n", "\n", "● ", "answer", "\n"]


def test_rich_streaming_appends_completed_lines_without_live_redraws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rich.live

    class RichConsole:
        _kairo_rich = True

        def __init__(self) -> None:
            self.prints: list[tuple[str, dict[str, object]]] = []

        def print(self, value: str, **kwargs: object) -> None:
            self.prints.append((value, kwargs))

    def unexpected_live(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("streaming answers must not use Rich Live")

    monkeypatch.setattr(rich.live, "Live", unexpected_live)
    streamed: list[str] = []
    monkeypatch.setattr(cli_module, "_write_stream", streamed.append)
    console = RichConsole()
    display = _StreamingAnswerDisplay(console, "inline", 80, lambda: "Thought for 1s")

    display.append("first")
    display.append(" line\nsecond")
    display.finish_block()

    assert console.prints == [
        (
            "  Thought for 1s",
            {"style": "#888888", "markup": False, "highlight": False},
        ),
        (
            "● ",
            {
                "style": "#888888",
                "markup": False,
                "highlight": False,
                "end": "",
            },
        ),
    ]
    assert streamed == [
        "\n",
        "\n",
        "first line\n",
        "  second\n",
        "\n",
    ]
    assert "".join(streamed).count("first line") == 1
    assert "".join(streamed).count("second") == 1


def test_terminal_erase_uses_configured_default_background() -> None:
    class Output:
        def __init__(self) -> None:
            self.writes: list[str] = []

        def write_raw(self, value: str) -> None:
            self.writes.append(value)

    output = Output()

    _erase_with_default_background(output, "\x1b[J")

    assert output.writes == ["\x1b[49m\x1b[J\x1b[0m"]
    assert "48;2;255;255;255" not in output.writes[0]


def test_working_status_uses_elapsed_seconds_and_escape_hint() -> None:
    assert _working_status_text(0.9) == "● Working (0s · esc to interrupt)"
    assert _working_status_text(1.9) == "● Working (1s · esc to interrupt)"


def test_mode_status_line_reflects_pending_execution_mode() -> None:
    assert cli_module._mode_status_line("agent", "deepseek/model") == "ReAct · deepseek/model"
    assert cli_module._mode_status_line("plan", "deepseek/model") == "Plan · deepseek/model"
    assert cli_module._mode_status_line("team", "deepseek/model") == "Team · deepseek/model"


@pytest.mark.parametrize("value", ["", " ", "\t\n", "\u3000\u00a0"])
def test_interactive_blank_submission_is_silently_ignored(value: str) -> None:
    assert cli_module._is_blank_interactive_submission(value) is True


@pytest.mark.parametrize("value", ["task", "/plan", "  task  "])
def test_interactive_nonblank_submission_is_not_ignored(value: str) -> None:
    assert cli_module._is_blank_interactive_submission(value) is False


def test_plan_summary_hides_internal_ids_and_empty_dependencies() -> None:
    plan = ExecutionPlan(
        [
            PlanTask("fetch-weather", "Fetch Shenzhen weather"),
            PlanTask("present", "Present the result", {"fetch-weather"}),
        ]
    )

    summary = cli_module._plan_summary(plan)

    assert summary == (
        "Plan · 2 steps\n  1  Fetch Shenzhen weather\n  2  Present the result\n     ↳ after 1"
    )
    assert "fetch-weather" not in summary
    assert "depends: none" not in summary


def test_approval_request_renders_readable_tool_arguments_and_choices() -> None:
    console = RecordingConsole()

    cli_module._print_approval_request(
        console,
        "install_skill",
        {"source": "pdf"},
        has_server_scope=False,
    )

    rendered = console.messages[0]
    assert "Permission required" in rendered
    assert "install_skill" in rendered
    assert '\n  "source": "pdf"\n' in rendered
    assert "y allow once" in rendered
    assert "m modify" in rendered
    assert "v always this server" not in rendered


async def test_interrupt_reuses_prompt_input_and_triggers_only_once() -> None:
    from prompt_toolkit.input.defaults import create_pipe_input

    calls: list[str] = []
    with create_pipe_input() as backend:
        interrupt = cli_module._EscapeInterrupt(lambda: calls.append("cancel"))
        interrupt.bind_input(backend)
        interrupt.start()
        backend.send_text("\x1b\x03")
        for _ in range(20):
            if calls:
                break
            await asyncio.sleep(0.01)
        interrupt.stop()

    assert calls == ["cancel"]


async def test_interrupt_does_not_use_prompt_toolkit_attach_registry() -> None:
    from prompt_toolkit.input.defaults import create_pipe_input

    with create_pipe_input() as backend:
        backend.attach = lambda _callback: (_ for _ in ()).throw(  # type: ignore[method-assign]
            AssertionError("interrupt listener must not mutate prompt_toolkit input ownership")
        )
        interrupt = cli_module._EscapeInterrupt(lambda: None)
        interrupt.bind_input(backend)
        interrupt.start()
        assert interrupt.active is True
        interrupt.stop()
        assert interrupt.active is False


async def test_interrupt_releases_input_across_repeated_prompt_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prompt_toolkit.output
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    workspace = tmp_path / "KairoCLI"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    output = DummyOutput()
    monkeypatch.setattr(prompt_toolkit.output, "create_output", lambda: output)

    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=output):
            session = _prompt_session(paths, lambda: "idle")
            interrupt = cli_module._EscapeInterrupt(lambda: None)
            interrupt.bind_input(session.app.input)
            for index in range(10):
                interrupt.start()
                interrupt.stop()
                read_task = asyncio.create_task(session.prompt_async())
                await asyncio.sleep(0.01)
                expected = f"round-{index}"
                pipe_input.send_text(expected + "\r")
                assert await asyncio.wait_for(read_task, 0.5) == expected


async def test_approval_prompt_takes_input_after_interrupt_listener_releases_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prompt_toolkit.output
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    workspace = tmp_path / "KairoCLI"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    output = DummyOutput()
    monkeypatch.setattr(prompt_toolkit.output, "create_output", lambda: output)

    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=output):
            session = _prompt_session(paths, lambda: "idle")
            interrupt = cli_module._EscapeInterrupt(lambda: None)
            interrupt.bind_input(session.app.input)
            interrupt.start()
            interrupt.stop()

            approval_task = asyncio.create_task(cli_module._read_approval_input(session))
            await asyncio.sleep(0.02)
            pipe_input.send_text("y\r")

            assert await asyncio.wait_for(approval_task, 0.5) == "y"

            read_task = asyncio.create_task(session.prompt_async())
            await asyncio.sleep(0.02)
            pipe_input.send_text("next task\r")
            assert await asyncio.wait_for(read_task, 0.5) == "next task"


async def test_wechat_workspace_uses_an_isolated_composer_style_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class MainSession:
        def __init__(self) -> None:
            self.style = object()
            self.color_depth = object()
            self.app = SimpleNamespace(input=object(), output=object())

    created: list[object] = []

    class IsolatedWorkspaceSession:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            current_window = SimpleNamespace(style="", height=object())
            content = SimpleNamespace(children=[])
            alternative_content = SimpleNamespace(style="", content=SimpleNamespace(children=[]))
            prompt_container = SimpleNamespace(
                content=content,
                alternative_content=alternative_content,
            )
            layout = SimpleNamespace(
                current_window=current_window,
                container=SimpleNamespace(children=[prompt_container]),
            )
            self.app = SimpleNamespace(layout=layout)
            created.append(self)

        async def prompt_async(self) -> str:
            return ""

    import prompt_toolkit

    monkeypatch.setattr(prompt_toolkit, "PromptSession", IsolatedWorkspaceSession)
    session = MainSession()

    assert await cli_module._read_wechat_workspace_input(session, tmp_path) == ""
    assert len(created) == 1
    workspace_session = created[0]
    prompt_text = "".join(fragment[1] for fragment in workspace_session.kwargs["message"])
    assert prompt_text == "  › "
    assert workspace_session.app.layout.current_window.style == "class:wechat.input"
    assert workspace_session.app.layout.current_window.height.min == 1
    assert workspace_session.app.layout.current_window.height.max == 1
    assert workspace_session.app.layout.current_window.dont_extend_height() is True
    assert (
        workspace_session.app.layout.container.children[0].alternative_content.style == "bg:default"
    )
    prompt_container = workspace_session.app.layout.container.children[0]
    for content in (prompt_container.content, prompt_container.alternative_content.content):
        assert len(content.children) == 2
        spacer = content.children[0]
        assert spacer.style == "bg:default"
        assert spacer.height.min == 1
        assert spacer.height.max == 1
        header = content.children[1]
        assert header.style == "class:wechat.input"
        assert header.height.min == 3
        assert header.height.max == 3
        header_text = "".join(fragment[1] for fragment in header.content.text)
        assert header_text == (
            "● Connect WeChat\n"
            f"  Workspace  {tmp_path}\n"
            "  Enter connect · Esc cancel · Type another path"
        )
    assert workspace_session.kwargs["input"] is session.app.input
    assert workspace_session.kwargs["output"] is session.app.output
    assert workspace_session.kwargs["style"] is session.style
    assert workspace_session.kwargs["key_bindings"] is not None


async def test_wechat_workspace_cancel_keeps_main_prompt_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prompt_toolkit.output
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    workspace = tmp_path / "KairoCLI"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    output = DummyOutput()
    monkeypatch.setattr(prompt_toolkit.output, "create_output", lambda: output)

    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=output):
            session = _prompt_session(paths, lambda: "idle")
            read_task = asyncio.create_task(
                cli_module._read_wechat_workspace_input(session, workspace)
            )
            await asyncio.sleep(0.02)
            pipe_input.send_bytes(b"\x03")
            with pytest.raises(cli_module._WechatSetupCanceled):
                await asyncio.wait_for(read_task, 0.5)

            next_input = asyncio.create_task(session.prompt_async())
            await asyncio.sleep(0.02)
            pipe_input.send_text("still running\r")
            assert await asyncio.wait_for(next_input, 0.5) == "still running"


@pytest.mark.parametrize("key", ["\x1b", "\x03"])
async def test_interrupt_reads_escape_and_ctrl_c_from_prompt_toolkit_input(key: str) -> None:
    from prompt_toolkit.input.defaults import create_pipe_input

    canceled = asyncio.Event()
    with create_pipe_input() as backend:
        interrupt = cli_module._EscapeInterrupt(canceled.set)
        interrupt.bind_input(backend)
        interrupt.start()
        backend.send_text(key)

        await asyncio.wait_for(canceled.wait(), 0.5)
        interrupt.stop()


async def test_plan_review_uses_an_isolated_prompt_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Buffer:
        history = object()

    class MainSession:
        def __init__(self) -> None:
            self.message = "main prompt"
            self.key_bindings = object()
            self.completer = object()
            self.bottom_toolbar = object()
            self.default_buffer = Buffer()
            self.style = object()
            self.color_depth = object()
            current_window = SimpleNamespace(style="class:composer.input", height=object())
            alternative_content = SimpleNamespace(style="class:composer.input")
            prompt_container = SimpleNamespace(alternative_content=alternative_content)
            container = SimpleNamespace(children=[prompt_container])
            layout = SimpleNamespace(current_window=current_window, container=container)
            self.app = SimpleNamespace(
                layout=layout,
                input=object(),
                output=object(),
            )

        async def prompt_async(self, **_kwargs: object) -> str:
            raise AssertionError("the main prompt session must not be run for plan review")

    created: list[object] = []

    class IsolatedReviewSession:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            current_window = SimpleNamespace(style="", height=object())
            alternative_content = SimpleNamespace(style="")
            prompt_container = SimpleNamespace(alternative_content=alternative_content)
            container = SimpleNamespace(children=[prompt_container])
            layout = SimpleNamespace(current_window=current_window, container=container)
            self.app = SimpleNamespace(layout=layout)
            created.append(self)

        async def prompt_async(self) -> str:
            self.prompt_height = self.app.layout.current_window.height
            self.prompt_alternative_style = self.app.layout.container.children[
                0
            ].alternative_content.style
            return "/cancel"

    import prompt_toolkit

    monkeypatch.setattr(prompt_toolkit, "PromptSession", IsolatedReviewSession)
    session = MainSession()
    original_state = (
        session.message,
        session.key_bindings,
        session.completer,
        session.bottom_toolbar,
        session.default_buffer.history,
        session.app.layout.current_window.style,
        session.app.layout.current_window.height,
        session.app.layout.container.children[0].alternative_content.style,
    )

    answer = await cli_module._read_plan_review_input(session)

    assert answer == "/cancel"
    assert len(created) == 1
    review_session = created[0]
    prompt_text = "".join(fragment[1] for fragment in review_session.kwargs["message"])
    assert prompt_text == ("\n● Review plan  Enter run  ·  Esc cancel  ·  Type to revise  › ")
    assert review_session.prompt_height.min == 2
    assert review_session.prompt_height.max == 2
    assert review_session.prompt_alternative_style == "class:plan-review.input"
    assert review_session.kwargs["key_bindings"] is not None
    assert review_session.kwargs["completer"] is not original_state[2]
    assert review_session.kwargs["bottom_toolbar"] == ""
    assert review_session.kwargs["input"] is session.app.input
    assert review_session.kwargs["output"] is session.app.output
    assert (
        session.message,
        session.key_bindings,
        session.completer,
        session.bottom_toolbar,
        session.default_buffer.history,
        session.app.layout.current_window.style,
        session.app.layout.current_window.height,
        session.app.layout.container.children[0].alternative_content.style,
    ) == original_state


async def test_prompt_accepts_input_after_plan_review_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prompt_toolkit.output
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    workspace = tmp_path / "KairoCLI"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    output = DummyOutput()
    monkeypatch.setattr(prompt_toolkit.output, "create_output", lambda: output)

    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=output):
            session = _prompt_session(paths, lambda: "idle")
            interrupt = cli_module._EscapeInterrupt(lambda: None)
            interrupt.bind_input(session.app.input)
            interrupt.start()
            interrupt.stop()
            review_task = asyncio.create_task(cli_module._read_plan_review_input(session))
            await asyncio.sleep(0.02)
            pipe_input.send_text("\r")
            assert await asyncio.wait_for(review_task, 0.5) == ""

            interrupt.start()
            interrupt.stop()
            read_task = asyncio.create_task(session.prompt_async())
            await asyncio.sleep(0.02)
            pipe_input.send_text("still editable\r")
            assert await asyncio.wait_for(read_task, 0.5) == "still editable"


async def test_prompt_remains_responsive_after_opening_slash_menu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prompt_toolkit.output
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    workspace = tmp_path / "KairoCLI"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    output = DummyOutput()
    monkeypatch.setattr(prompt_toolkit.output, "create_output", lambda: output)

    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=output):
            session = _prompt_session(paths, lambda: "idle")
            read_task = asyncio.create_task(session.prompt_async())
            await asyncio.sleep(0.02)
            pipe_input.send_text("/")
            await asyncio.sleep(0.02)
            pipe_input.send_text("help\r")
            assert await asyncio.wait_for(read_task, 0.5) == "/help"


async def test_prompt_ignores_blank_enter_without_finishing_input_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prompt_toolkit.output
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    workspace = tmp_path / "KairoCLI"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    output = DummyOutput()
    monkeypatch.setattr(prompt_toolkit.output, "create_output", lambda: output)

    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=output):
            session = _prompt_session(paths, lambda: "idle")
            read_task = asyncio.create_task(session.prompt_async())
            await asyncio.sleep(0.02)
            pipe_input.send_text("\r")
            await asyncio.sleep(0.02)
            assert read_task.done() is False
            assert session.default_buffer.text == ""

            pipe_input.send_text(" \t\r")
            await asyncio.sleep(0.02)
            assert read_task.done() is False
            assert session.default_buffer.text == " "

            pipe_input.send_bytes(b"\x15")
            pipe_input.send_text("actual task\r")
            assert await asyncio.wait_for(read_task, 0.5) == "actual task"


async def test_inline_prompt_preserves_shift_enter_and_pasted_newlines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prompt_toolkit.output
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    workspace = tmp_path / "KairoCLI"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    output = DummyOutput()
    monkeypatch.setattr(prompt_toolkit.output, "create_output", lambda: output)

    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=output):
            session = _prompt_session(paths, lambda: "idle")
            read_task = asyncio.create_task(session.prompt_async())
            await asyncio.sleep(0.02)
            pipe_input.send_text("first")
            pipe_input.send_bytes(b"\x1b\r")
            pipe_input.send_text("second")
            await asyncio.sleep(0.02)
            assert session.default_buffer.text == "first\nsecond"
            pipe_input.send_text("\r")
            assert await asyncio.wait_for(read_task, 0.5) == "first\nsecond"

            read_task = asyncio.create_task(session.prompt_async())
            await asyncio.sleep(0.02)
            pipe_input.send_text("xterm")
            pipe_input.send_bytes(b"\x1b[27;2;13~")
            pipe_input.send_text("csi-u")
            pipe_input.send_bytes(b"\x1b[13;2u")
            pipe_input.send_text("done")
            await asyncio.sleep(0.02)
            assert session.default_buffer.text == "xterm\ncsi-u\ndone"
            pipe_input.send_text("\r")
            assert await asyncio.wait_for(read_task, 0.5) == "xterm\ncsi-u\ndone"

            read_task = asyncio.create_task(session.prompt_async())
            await asyncio.sleep(0.02)
            pipe_input.send_bytes(b"\x1b[200~pasted first\npasted second\x1b[201~")
            await asyncio.sleep(0.02)
            assert session.default_buffer.text == "pasted first\npasted second"
            pipe_input.send_text("\r")
            assert await asyncio.wait_for(read_task, 0.5) == "pasted first\npasted second"


async def test_inline_input_enables_and_restores_modified_key_reporting() -> None:
    writes: list[str] = []
    flushes = 0

    class Output:
        def write_raw(self, value: str) -> None:
            writes.append(value)

        def flush(self) -> None:
            nonlocal flushes
            flushes += 1

    class Session:
        app = SimpleNamespace(output=Output())

        async def prompt_async(self) -> str:
            assert writes == ["\x1b[>4;1m\x1b[>1u"]
            return "task"

    assert await _read_input(Session()) == "task"
    assert writes == ["\x1b[>4;1m\x1b[>1u", "\x1b[<u\x1b[>4m"]
    assert flushes == 2


async def test_ctrl_c_remains_responsive_with_slash_menu_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import prompt_toolkit.output
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    workspace = tmp_path / "KairoCLI"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    output = DummyOutput()
    monkeypatch.setattr(prompt_toolkit.output, "create_output", lambda: output)

    class PromptInterrupted(Exception):
        pass

    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=output):
            session = _prompt_session(paths, lambda: "idle")
            session.interrupt_exception = PromptInterrupted
            read_task = asyncio.create_task(session.prompt_async())
            await asyncio.sleep(0.02)
            pipe_input.send_text("/")
            await asyncio.sleep(0.02)
            pipe_input.send_bytes(b"\x03")
            with pytest.raises(PromptInterrupted):
                await asyncio.wait_for(read_task, 0.5)


def test_session_startup_notice_hides_session_id() -> None:
    assert _session_startup_notice(0) is None
    assert _session_startup_notice(1) == "Restored 1 message"
    assert _session_startup_notice(3) == "Restored 3 messages"


def test_resume_session_hint_uses_the_installed_cli_command() -> None:
    assert _resume_session_hint("session_123456789abc") == (
        "Resume this session with:\nkairocli --resume session_123456789abc"
    )


def test_working_indicator_has_black_marker_and_vertical_margins() -> None:
    lifecycle = cli_module._TurnLifecycleState()
    lifecycle.start()
    indicator = cli_module._WorkingIndicator(RecordingConsole(), lifecycle)

    rendered = indicator._render()

    assert rendered.plain.startswith("\n● Working (")
    assert rendered.plain.endswith("\n")
    marker = rendered.spans[0]
    assert (marker.start, marker.end, marker.style) == (1, 3, "#111111")


def test_working_indicator_uses_continuous_turn_lifecycle_time() -> None:
    now = [10.0]
    lifecycle = cli_module._TurnLifecycleState(lambda: now[0])
    lifecycle.start()
    first = cli_module._WorkingIndicator(RecordingConsole(), lifecycle)

    now[0] = 14.9
    assert first._text() == "● Working (4s · esc to interrupt)"
    first.stop()

    now[0] = 20.2
    restarted = cli_module._WorkingIndicator(RecordingConsole(), lifecycle)
    assert restarted._text() == "● Working (10s · esc to interrupt)"

    lifecycle.finish()
    assert lifecycle.running is False
    assert lifecycle.elapsed() == 0.0

    now[0] = 30.0
    lifecycle.start()
    now[0] = 31.9
    next_turn = cli_module._WorkingIndicator(RecordingConsole(), lifecycle)
    assert next_turn._text() == "● Working (1s · esc to interrupt)"


def test_turn_lifecycle_excludes_paused_plan_review_time() -> None:
    now = [10.0]
    lifecycle = cli_module._TurnLifecycleState(lambda: now[0])
    lifecycle.start()
    now[0] = 14.0
    lifecycle.pause()

    now[0] = 100.0
    assert lifecycle.elapsed() == 4.0

    lifecycle.resume()
    now[0] = 103.0
    assert lifecycle.elapsed() == 7.0

    lifecycle.finish()
    assert lifecycle.elapsed() == 0.0


def test_answer_block_leaves_a_terminal_background_row_below(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = io.StringIO()
    monkeypatch.setattr(cli_module.sys, "stdout", output)

    cli_module._end_answer_block()

    assert output.getvalue() == "\n"


def test_welcome_card_uses_display_brand_and_preserves_directory_name() -> None:
    workspace = Path("/tmp/KairoCLI")

    lines = _welcome_lines("deepseek", "v4-pro", workspace, columns=120)
    rendered = "\n".join(lines)

    assert "Kairo CLI 0.1.0" in rendered
    assert "deepseek/v4-pro · API usage" in rendered
    assert "Tips for getting started" in rendered
    assert "Run /init to create KAIRO.md" in rendered
    assert "█▄▀ ▄▀▄ ▀█▀ █▀▄ ▄▀▄" in rendered
    assert "▀ ▀ ▀ ▀ ▀▀▀ ▀ ▀  ▀" in rendered
    divider_column = lines[2].find("│", 1)
    assert lines[1][divider_column] == "╷"
    assert lines[-2][divider_column] == "╵"
    assert lines[0][divider_column] == "─"
    assert lines[-1][divider_column] == "─"
    assert any(
        line[divider_column] == "│"
        and line[divider_column + 1 :].startswith(" ─")
        and line.endswith("─ │")
        for line in lines
    )
    assert rendered.count("KairoCLI") == 1
    assert {len(line) for line in lines} == {120}


def test_welcome_card_stacks_at_narrow_terminal_width() -> None:
    lines = _welcome_lines("deepseek", "v4-pro", Path("/tmp/work"), columns=60)
    rendered = "\n".join(lines)

    assert "Welcome back!" in rendered
    assert "Tips for getting started" in rendered
    assert sum(line.startswith("├") and line.endswith("┤") for line in lines) == 1
    assert any(line.startswith("│ ─") and line.endswith("─ │") for line in lines)
    assert {len(line) for line in lines} == {60}


def test_welcome_card_keeps_side_by_side_layout_at_typical_terminal_width() -> None:
    lines = _welcome_lines("deepseek", "v4-pro", Path("/tmp/work"), columns=80)

    assert not any(line.startswith("├") for line in lines)
    divider_column = lines[2].find("│", 1)
    assert lines[1][divider_column] == "╷"
    assert lines[-2][divider_column] == "╵"
    assert lines[0][divider_column] == "─"
    assert lines[-1][divider_column] == "─"
    assert len(lines) < 16
    assert {len(line) for line in lines} == {80}


def test_prompt_history_rejects_symlinked_user_container(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    outside = tmp_path / "outside"
    history_dir = outside / "history"
    history_dir.mkdir(parents=True)
    sentinel = history_dir / "input.history"
    sentinel.write_text("outside-secret", encoding="utf-8")
    try:
        (home / ".kairocli").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")
    paths = KairoPaths.discover(workspace, home)

    session = _prompt_session(paths, lambda: "idle")
    assert session is not None
    session.history.append_string("must stay in memory")

    assert sentinel.read_text(encoding="utf-8") == "outside-secret"
    assert _clear_input_history(paths.history_file) is False
    assert sentinel.read_text(encoding="utf-8") == "outside-secret"


def test_prompt_history_tail_read_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    history = tmp_path / "input.history"
    history.write_bytes(b"old" * 100 + b"\n# recent\n+latest entry\n")
    reads: list[int] = []
    original_read = os.read

    def recording_read(descriptor: int, maximum: int) -> bytes:
        reads.append(maximum)
        return original_read(descriptor, maximum)

    monkeypatch.setattr(input_history_module, "MAX_INPUT_HISTORY_READ_BYTES", 64)
    monkeypatch.setattr(input_history_module.os, "read", recording_read)

    assert _load_input_history(history) == ["latest entry"]
    assert reads and max(reads) <= 64


def test_prompt_history_skips_corrupt_and_legacy_sensitive_entries(
    tmp_path: Path,
) -> None:
    history = tmp_path / "input.history"
    history.write_bytes(
        b"\n# first\n+valid old\n"
        b"\n# secret\n+API_KEY=legacy-secret\n"
        b"\n# corrupt\n+\xff\xfe\n"
        b"\n# latest\n+valid new\n"
    )

    assert _load_input_history(history) == ["valid new", "valid old"]


def test_prompt_history_retention_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    history = tmp_path / "input.history"
    monkeypatch.setattr(input_history_module, "MAX_INPUT_HISTORY_FILE_BYTES", 192)

    for index in range(20):
        assert _append_input_history(history, f"entry-{index:02d}-" + "x" * 24)

    assert history.stat().st_size <= 192
    loaded = _load_input_history(history)
    assert loaded
    assert loaded[0] == "entry-19-" + "x" * 24
    assert all("entry-00-" not in value for value in loaded)


def test_prompt_history_appends_are_serialized_across_processes(tmp_path: Path) -> None:
    history = tmp_path / "history" / "input.history"
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_append_history_worker,
            args=(str(history), f"worker-{worker}", 30),
        )
        for worker in range(3)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0

    loaded = _load_input_history(history)
    assert len(loaded) == 90
    assert len(set(loaded)) == 90
    assert {f"worker-{worker}-{index:03d}" for worker in range(3) for index in range(30)} == set(
        loaded
    )


def test_prompt_history_refuses_symlinked_lock_file(tmp_path: Path) -> None:
    history = tmp_path / "history" / "input.history"
    history.parent.mkdir()
    outside = tmp_path / "outside.lock"
    outside.write_text("sentinel", encoding="utf-8")
    try:
        (history.parent / ".input-history.lock").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")

    assert _append_input_history(history, "must not persist") is False
    assert _clear_input_history(history) is False
    assert not history.exists()
    assert outside.read_text(encoding="utf-8") == "sentinel"


def test_clear_input_history_removes_only_regular_history_file(tmp_path: Path) -> None:
    history = tmp_path / "history" / "input.history"
    history.parent.mkdir()
    history.write_text("entry", encoding="utf-8")

    assert _clear_input_history(history) is True
    assert not history.exists()

    outside = tmp_path / "outside.history"
    outside.write_text("sentinel", encoding="utf-8")
    try:
        history.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")
    assert _clear_input_history(history) is False
    assert history.is_symlink()
    assert outside.read_text(encoding="utf-8") == "sentinel"


def test_history_excludes_empty_and_oversized_lines() -> None:
    assert _is_sensitive_history_input("   ") is True
    assert _is_sensitive_history_input("x" * 8_001) is True
    assert _is_sensitive_history_input(("word " * 1_600).strip()) is False


def test_local_path_completion_is_workspace_fenced_and_bounded(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    (tmp_path / "outside.txt").write_text("secret", encoding="utf-8")
    (workspace / "hello world.txt").write_text("hello", encoding="utf-8")
    (workspace / "[literal].txt").write_text("literal", encoding="utf-8")
    for index in range(60):
        (workspace / f"file-{index:02}.txt").touch()
    try:
        (workspace / "escape").symlink_to(tmp_path / "outside.txt")
        (workspace / "broken").symlink_to(tmp_path / "missing.txt")
    except OSError:
        pass

    assert _local_path_completion_candidates(workspace, "@../") == []
    assert _local_path_completion_candidates(workspace, "@escape") == []
    assert _local_path_completion_candidates(workspace, "@[") == ["@[literal].txt"]
    assert _local_path_completion_candidates(workspace, "@hello") == ["@<hello world.txt>"]
    assert _local_path_completion_candidates(workspace, "@image:hello") == [
        "@image:<hello world.txt>"
    ]
    assert len(_local_path_completion_candidates(workspace, "@file-")) == 50


def test_completion_understands_open_angles_and_slash_subcommands() -> None:
    assert _completion_word("read @<hello wor") == "@<hello wor"
    assert _completion_word("read @image:<hello wor") == "@image:<hello wor"
    assert _slash_completion_candidates("/mcp r", ["/mcp"]) == [
        "/mcp resources",
        "/mcp restart",
    ]
    assert _slash_completion_candidates("/se", ["/session", "/search"]) == [
        "/session",
        "/search",
    ]
    assert _slash_completion_candidates("/history c", ["/history clear"]) == ["/history clear"]
    assert _slash_completion_candidates("/skill i", ["/skill"]) == ["/skill install"]
    assert _slash_completion_candidates("/wechat ", ["/wechat"]) == [
        "/wechat setup",
        "/wechat start",
        "/wechat status",
        "/wechat stop",
    ]
    assert _slash_completion_candidates("/hitl ", ["/hitl"]) == [
        "/hitl on",
        "/hitl off",
    ]
    assert _slash_completion_candidates("/config ", ["/config"]) == ["/config provider"]
    assert _slash_completion_candidates("/trace reasoning ", ["/trace"]) == [
        "/trace reasoning off",
        "/trace reasoning on",
    ]
    assert _slash_completion_candidates("/todo clear ", ["/todo"]) == [
        "/todo clear completed",
        "/todo clear all",
    ]


def test_slash_completion_uses_live_provider_mcp_and_skill_names() -> None:
    options = {
        "providers": ("step", "freellmapi"),
        "mcp_servers": ("chrome", "fs"),
        "skills": ("web-access", "review"),
    }
    assert _slash_completion_candidates("/model st", ["/model"], **options) == ["/model step"]
    assert _slash_completion_candidates("/config provider fr", ["/config"], **options) == [
        "/config provider freellmapi"
    ]
    assert _slash_completion_candidates("/mcp logs ch", ["/mcp"], **options) == ["/mcp logs chrome"]
    assert _slash_completion_candidates("/skill show web", ["/skill"], **options) == [
        "/skill show web-access"
    ]


def test_input_highlighting_warns_without_changing_text() -> None:
    text = 'run sudo rm -rf / with API_KEY and @image:<shot.png then "open'
    highlighted = _highlight_input_line(text)

    assert "".join(part for _, part in highlighted) == text
    styles = {style for style, _ in highlighted}
    assert "fg:#ff5f5f bold underline" in styles
    assert "fg:#ffd75f bold" in styles
    assert "fg:#d787ff bold" in styles
    assert "fg:#ffd75f underline" in styles

    slash = _highlight_input_line("/mcp restart")
    assert slash[0] == ("fg:#111111", "/mcp")
