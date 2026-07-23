import asyncio
import json
import threading
from pathlib import Path
from typing import Any

import pytest
from textual.app import App
from textual.widgets import Button, DirectoryTree, Input, RichLog

import kairocli.tui as tui_module
from kairocli.agent import Agent
from kairocli.config import AppConfig
from kairocli.llm import LlmClient
from kairocli.mcp import McpServerConfig, McpServerManager
from kairocli.memory import MemoryStore
from kairocli.models import LlmResponse, Message
from kairocli.paths import KairoPaths
from kairocli.policy import ApprovalDecision
from kairocli.skills import SkillRegistry
from kairocli.tasks import DurableTaskManager, DurableTaskStore
from kairocli.tools import ToolRegistry
from kairocli.tui import (
    _modified_approval_result,
    _redact_tui_transcript_input,
    _safe_tui_error,
    _tui_config_summary,
    _tui_run_mode,
    _tui_submission_action,
    run_tui,
)


def test_tui_error_text_is_total_redacted_and_unicode_safe() -> None:
    class UnprintableTuiError(RuntimeError):
        def __str__(self) -> str:
            raise KeyboardInterrupt

    assert _safe_tui_error(UnprintableTuiError()) == ("UnprintableTuiError message unavailable")
    assert _safe_tui_error(RuntimeError("token=secret-value\ud800")) == "token=***"


async def test_tui_unmount_bounds_uncooperative_worker_wait(
    tmp_path: Path, monkeypatch: Any
) -> None:
    agent = Agent(BlockingClient(), ToolRegistry(tmp_path), "system")
    captured: list[App[Any]] = []
    monkeypatch.setattr(App, "run", lambda self: captured.append(self))
    monkeypatch.setattr(tui_module, "TUI_WORKER_SHUTDOWN_GRACE_SECONDS", 0.01)
    run_tui(agent)
    release = asyncio.Event()

    class UncooperativeWorker:
        is_running = True
        cancel_count = 0

        def cancel(self) -> None:
            self.cancel_count += 1

        async def wait(self) -> None:
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()

    worker = UncooperativeWorker()
    captured[0]._turn_worker = worker  # type: ignore[attr-defined,assignment]

    await asyncio.wait_for(captured[0].on_unmount(), 0.2)  # type: ignore[attr-defined]

    assert worker.cancel_count == 1
    assert tui_module._DETACHED_TUI_WAITS
    release.set()
    for _ in range(20):
        if not tui_module._DETACHED_TUI_WAITS:
            break
        await asyncio.sleep(0.01)
    assert tui_module._DETACHED_TUI_WAITS == set()


async def test_tui_unmount_finishes_session_save_before_propagating_cancel(
    tmp_path: Path, monkeypatch: Any
) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    store = tui_module.SessionStore(tmp_path / "sessions.db")
    state = store.create(workspace, "test", "blocking")
    agent = Agent(BlockingClient(), ToolRegistry(workspace), "system")
    agent.history = [Message("user", "last turn")]
    captured: list[App[Any]] = []
    monkeypatch.setattr(App, "run", lambda self: captured.append(self))
    original_save = store.save_snapshot
    save_started = threading.Event()
    release_save = threading.Event()
    tools_closed = asyncio.Event()
    original_close = agent.tools.close

    def delayed_save(*args: Any, **kwargs: Any) -> None:
        save_started.set()
        assert release_save.wait(2)
        original_save(*args, **kwargs)

    async def close_tools() -> None:
        await original_close()
        tools_closed.set()

    monkeypatch.setattr(store, "save_snapshot", delayed_save)
    monkeypatch.setattr(agent.tools, "close", close_tools)
    run_tui(agent, store, state.meta.id, workspace)
    closing = asyncio.create_task(captured[0].on_unmount())  # type: ignore[attr-defined]
    assert await asyncio.to_thread(save_started.wait, 2)
    closing.cancel()
    await asyncio.sleep(0)

    assert not closing.done()
    release_save.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert tools_closed.is_set()
    persisted = store.load(state.meta.id, workspace)
    assert persisted is not None
    assert [message.content for message in persisted.messages] == ["last turn"]


def test_tui_submission_requires_explicit_cancellation() -> None:
    assert _tui_submission_action("new work", running=True) == "busy"
    assert _tui_submission_action("/plan new work", running=True) == "busy"
    assert _tui_submission_action("/cancel", running=True) == "cancel"
    assert _tui_submission_action("cancel", running=False) == "idle-cancel"
    assert _tui_submission_action("/exit", running=False) == "exit"
    assert _tui_submission_action("  ", running=False) == "ignore"


def test_tui_run_mode_preserves_task_text() -> None:
    assert _tui_run_mode("normal task") == ("react", "normal task")
    assert _tui_run_mode("/plan Refactor This Class") == (
        "plan",
        "Refactor This Class",
    )
    assert _tui_run_mode("/TEAM verify tests") == ("team", "verify tests")
    assert _tui_run_mode("/plan") == ("plan", "")


def test_tui_modified_approval_requires_json_object() -> None:
    result = _modified_approval_result('{"path":"safe.txt","content":"ok"}')
    assert result.decision == ApprovalDecision.MODIFIED
    assert result.modified_arguments == {"path": "safe.txt", "content": "ok"}
    deeply_nested = '{"path":' + "[" * 33 + "0" + "]" * 33 + "}"
    for invalid in (
        "not-json",
        "[]",
        '"text"',
        '{"path":"first","path":"second"}',
        '{"timeout":NaN}',
        deeply_nested,
    ):
        try:
            _modified_approval_result(invalid)
        except ValueError:
            pass
        else:  # pragma: no cover - makes failure intent explicit
            raise AssertionError(f"Expected invalid replacement: {invalid}")


def test_tui_transcript_redacts_config_api_key() -> None:
    rendered = _redact_tui_transcript_input("/config provider glm api-key top-secret-value")
    assert "top-secret-value" not in rendered
    assert rendered.endswith("api-key ***")


def test_tui_config_summary_omits_api_keys(tmp_path: Path) -> None:
    paths = KairoPaths.discover(tmp_path, tmp_path / "home")
    config = AppConfig.load(paths)
    config.providers["glm"].api_key = "top-secret"
    summary = _tui_config_summary(config)
    assert "Active provider:" in summary
    assert "glm" in summary
    assert "top-secret" not in summary


async def test_tui_has_toggleable_workspace_tree(tmp_path: Path, monkeypatch: Any) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    (workspace / "src").mkdir()
    (workspace / "src" / "main.py").write_text("print('ok')", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = workspace / "outside-link"
    linked.symlink_to(outside, target_is_directory=True)
    agent = Agent(ImmediateClient(), ToolRegistry(workspace), "system")
    captured: list[App[Any]] = []
    monkeypatch.setattr(App, "run", lambda self: captured.append(self))
    run_tui(agent, workspace=workspace)

    async with captured[0].run_test() as pilot:
        tree = captured[0].query_one(DirectoryTree)
        assert tree.path == workspace
        assert linked not in list(tree.filter_paths([linked]))
        monkeypatch.setattr(tui_module, "MAX_TUI_TREE_ENTRIES", 1)
        assert len(list(tree.filter_paths([workspace / "src", workspace / "other"]))) == 1
        await pilot.press("ctrl+b")
        assert tree.has_class("hidden")
        await pilot.press("ctrl+b")
        assert not tree.has_class("hidden")


class BlockingClient(LlmClient):
    provider = "test"
    model = "blocking"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return LlmResponse(content="done")


class ImmediateClient(LlmClient):
    provider = "test"
    model = "immediate"

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        return LlmResponse(content="background done")


class UnsafeDisplayClient(LlmClient):
    provider = "test"
    model = "unsafe-display"

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        return LlmResponse(content="[red]literal[/red]\x1b]52;c;clipboard-secret\x07")


async def test_tui_treats_model_markup_as_text_and_removes_terminal_controls(
    tmp_path: Path, monkeypatch: Any
) -> None:
    agent = Agent(UnsafeDisplayClient(), ToolRegistry(tmp_path), "system")
    captured: list[App[Any]] = []
    monkeypatch.setattr(App, "run", lambda self: captured.append(self))
    run_tui(agent)

    async with captured[0].run_test() as pilot:
        input_widget = captured[0].query_one(Input)
        input_widget.focus()
        input_widget.value = "render safely"
        await pilot.press("enter")
        for _ in range(50):
            worker = captured[0]._turn_worker  # type: ignore[attr-defined]
            if worker is not None and not worker.is_running:
                break
            await asyncio.sleep(0.01)
        transcript = "\n".join(line.text for line in captured[0].query_one(RichLog).lines)
        assert "[red]literal[/red]" in transcript
        assert "clipboard-secret" not in transcript
        assert "\x1b" not in transcript


async def test_tui_busy_input_does_not_implicitly_cancel_active_turn(
    tmp_path: Path, monkeypatch: Any
) -> None:
    client = BlockingClient()
    agent = Agent(client, ToolRegistry(tmp_path), "system")
    captured: list[App[Any]] = []
    monkeypatch.setattr(App, "run", lambda self: captured.append(self))
    run_tui(agent)
    app = captured[0]

    async with app.run_test() as pilot:
        input_widget = app.query_one(Input)
        input_widget.focus()
        await pilot.pause()
        input_widget.value = "first task"
        await pilot.press("enter")
        await asyncio.wait_for(client.started.wait(), 1)

        input_widget.value = "second task"
        await pilot.press("enter")
        await pilot.pause()
        assert client.calls == 1
        assert not agent.cancel_event.is_set()

        input_widget.value = "/cancel"
        await pilot.press("enter")
        await pilot.pause()
        assert agent.cancel_event.is_set()


async def test_tui_starts_mcp_and_expands_local_and_resource_mentions(
    tmp_path: Path, monkeypatch: Any
) -> None:
    (tmp_path / "note.txt").write_text("trusted source body", encoding="utf-8")

    class RecordingClient(LlmClient):
        provider = "test"
        model = "recording"

        def __init__(self) -> None:
            self.called = asyncio.Event()
            self.prompt = ""

        async def complete(
            self, messages: list[Message], tools: list[dict[str, Any]] | None = None
        ) -> LlmResponse:
            self.prompt = str(messages[-1].content)
            self.called.set()
            return LlmResponse(content="done")

    class FakeMcpManager:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.closed = False

        async def start_all(self) -> None:
            self.started.set()

        def resource_index(self) -> str:
            return "- demo: demo://x"

        def status(self) -> dict[str, str]:
            return {"demo": "RUNNING"}

        async def expand_resource_mentions(self, value: str) -> str:
            return value.replace("@demo:demo://x", "<resource>remote body</resource>")

        async def close(self) -> None:
            self.closed = True

    client = RecordingClient()
    manager = FakeMcpManager()
    agent = Agent(client, ToolRegistry(tmp_path), "system")
    captured: list[App[Any]] = []
    monkeypatch.setattr(App, "run", lambda self: captured.append(self))
    run_tui(agent, workspace=tmp_path, mcp_manager=manager)  # type: ignore[arg-type]

    async with captured[0].run_test() as pilot:
        await asyncio.wait_for(manager.started.wait(), 1)
        input_widget = captured[0].query_one(Input)
        input_widget.focus()
        input_widget.value = "read @note.txt and @demo:demo://x"
        await pilot.press("enter")
        await asyncio.wait_for(client.called.wait(), 1)
        assert "trusted source body" in client.prompt
        assert "<resource>remote body</resource>" in client.prompt
    assert manager.closed


async def test_tui_modified_approval_is_rechecked_by_tool_policy(
    tmp_path: Path, monkeypatch: Any
) -> None:
    agent = Agent(BlockingClient(), ToolRegistry(tmp_path), "system")
    captured: list[App[Any]] = []
    monkeypatch.setattr(App, "run", lambda self: captured.append(self))
    run_tui(agent)
    app = captured[0]

    async with app.run_test() as pilot:
        await pilot.pause()
        execution = app.run_worker(
            agent.tools.execute("write_file", {"path": "original.txt", "content": "original"}),
            name="approval-test",
        )
        for _ in range(20):
            await pilot.pause()
            try:
                editor = app.screen.query_one("#approval-arguments", Input)
                break
            except Exception:
                continue
        else:
            raise AssertionError(
                f"Approval modal did not open: {execution.state} {execution.error!r}"
            )
        editor.value = "not-json"
        await pilot.click("#approval-modify")
        await pilot.pause()
        assert "Invalid JSON" in str(app.screen.query_one("#approval-error").render())
        editor.value = '{"path":"../escape.txt","content":"changed"}'
        await pilot.pause()
        app.screen.query_one("#approval-modify", Button).press()
        await pilot.pause()
        result = json.loads(await asyncio.wait_for(execution.wait(), 1))
        assert result["policy_denied"] is True
        assert not (tmp_path.parent / "escape.txt").exists()


async def test_tui_approval_supports_tool_and_mcp_server_scopes(
    tmp_path: Path, monkeypatch: Any
) -> None:
    agent = Agent(BlockingClient(), ToolRegistry(tmp_path), "system")
    captured: list[App[Any]] = []
    monkeypatch.setattr(App, "run", lambda self: captured.append(self))
    run_tui(agent)
    app = captured[0]

    async with app.run_test() as pilot:
        await pilot.pause()
        approve_tool = app.run_worker(
            app.approve_tool("write_file", {"path": "x", "content": "y"}),  # type: ignore[attr-defined]
            name="approve-tool-scope",
        )
        for _ in range(20):
            await pilot.pause()
            if app.screen.query("#approval-all"):
                break
        await pilot.click("#approval-all")
        tool_result = await approve_tool.wait()
        assert tool_result.decision == ApprovalDecision.APPROVED_ALL

        approve_server = app.run_worker(
            app.approve_tool("mcp__demo__write", {"value": 1}),  # type: ignore[attr-defined]
            name="approve-server-scope",
        )
        for _ in range(20):
            await pilot.pause()
            if app.screen.query("#approval-server"):
                break
        await pilot.click("#approval-server")
        server_result = await approve_server.wait()
        assert server_result.decision == ApprovalDecision.APPROVED_ALL_BY_SERVER


async def test_tui_task_command_runs_and_persists_background_work(
    tmp_path: Path, monkeypatch: Any
) -> None:
    main_agent = Agent(BlockingClient(), ToolRegistry(tmp_path), "system")
    store = DurableTaskStore(tmp_path / "tui-tasks.db")
    manager = DurableTaskManager(
        store,
        lambda: Agent(ImmediateClient(), ToolRegistry(tmp_path), "system"),
        workers=1,
    )
    captured: list[App[Any]] = []
    monkeypatch.setattr(App, "run", lambda self: captured.append(self))
    run_tui(main_agent, task_store=store, task_manager=manager)

    async with captured[0].run_test() as pilot:
        input_widget = captured[0].query_one(Input)
        input_widget.focus()
        input_widget.value = "/task add background work"
        await pilot.press("enter")
        for _ in range(100):
            tasks = store.list(1)
            if tasks and tasks[0].status == "completed":
                break
            await asyncio.sleep(0.02)
        assert tasks[0].prompt == "background work"
        assert tasks[0].output == "background done"


async def test_tui_index_and_search_are_management_commands(
    tmp_path: Path, monkeypatch: Any
) -> None:
    (tmp_path / "service.py").write_text(
        "def unique_payment_handler():\n    return 'paid'\n", encoding="utf-8"
    )
    client = BlockingClient()
    agent = Agent(client, ToolRegistry(tmp_path), "system")
    captured: list[App[Any]] = []
    monkeypatch.setattr(App, "run", lambda self: captured.append(self))
    run_tui(agent)

    async with captured[0].run_test() as pilot:
        input_widget = captured[0].query_one(Input)
        input_widget.focus()
        input_widget.value = "/index"
        await pilot.press("enter")
        for _ in range(100):
            if "service.py" in agent.tools.code_index.store.paths():
                break
            await asyncio.sleep(0.02)
        for _ in range(50):
            worker = captured[0]._turn_worker  # type: ignore[attr-defined]
            if worker is not None and not worker.is_running:
                break
            await asyncio.sleep(0.01)
        input_widget.value = "/search unique_payment_handler"
        await pilot.press("enter")
        for _ in range(50):
            worker = captured[0]._turn_worker  # type: ignore[attr-defined]
            if worker is not None and not worker.is_running:
                break
            await asyncio.sleep(0.01)
        transcript = "\n".join(line.text for line in captured[0].query_one(RichLog).lines)
        assert "service.py" in transcript
        assert "unique_payment_handler" in transcript
        assert client.calls == 0


async def test_tui_mcp_management_uses_shared_commands(tmp_path: Path, monkeypatch: Any) -> None:
    client = BlockingClient()
    registry = ToolRegistry(tmp_path)
    agent = Agent(client, registry, "system")
    paths = KairoPaths.discover(tmp_path, tmp_path / "home")
    manager = McpServerManager(paths, registry)
    manager.configs = {"demo": McpServerConfig(command="unused")}

    class FakeMcpClient:
        resources = [{"uri": "demo://one", "name": "one"}]
        prompts = [{"name": "review"}]
        stderr_log = ["ready"]
        tools: list[dict[str, object]] = []

        async def close(self) -> None:
            return None

    manager.clients["demo"] = FakeMcpClient()  # type: ignore[assignment]

    async def keep_existing_clients() -> None:
        return None

    monkeypatch.setattr(manager, "start_all", keep_existing_clients)
    captured: list[App[Any]] = []
    monkeypatch.setattr(App, "run", lambda self: captured.append(self))
    run_tui(agent, mcp_manager=manager)

    async with captured[0].run_test() as pilot:
        input_widget = captured[0].query_one(Input)
        input_widget.focus()
        input_widget.value = "/mcp resources demo"
        await pilot.press("enter")
        for _ in range(50):
            worker = captured[0]._turn_worker  # type: ignore[attr-defined]
            if worker is not None and not worker.is_running:
                break
            await asyncio.sleep(0.01)
        transcript = "\n".join(line.text for line in captured[0].query_one(RichLog).lines)
        assert "demo://one" in transcript
        assert client.calls == 0


async def test_tui_plan_requires_explicit_review(tmp_path: Path, monkeypatch: Any) -> None:
    agent = Agent(ImmediateClient(), ToolRegistry(tmp_path), "system")
    captured: list[App[Any]] = []
    monkeypatch.setattr(App, "run", lambda self: captured.append(self))
    run_tui(agent)

    async with captured[0].run_test() as pilot:
        input_widget = captured[0].query_one(Input)
        input_widget.focus()
        input_widget.value = "/plan inspect behavior"
        await pilot.press("enter")
        for _ in range(50):
            await pilot.pause()
            if captured[0].screen.query("#plan-cancel"):
                break
        assert captured[0].screen.query("#plan-execute")
        assert captured[0].screen.query("#plan-replan")
        await pilot.click("#plan-cancel")
        for _ in range(50):
            worker = captured[0]._turn_worker  # type: ignore[attr-defined]
            if worker is not None and not worker.is_running:
                break
            await asyncio.sleep(0.01)
        transcript = "\n".join(line.text for line in captured[0].query_one(RichLog).lines)
        assert "Plan canceled." in transcript


async def test_tui_team_command_reaches_orchestrator(tmp_path: Path, monkeypatch: Any) -> None:
    agent = Agent(ImmediateClient(), ToolRegistry(tmp_path), "system")
    captured: list[App[Any]] = []
    monkeypatch.setattr(App, "run", lambda self: captured.append(self))
    run_tui(agent)

    async with captured[0].run_test() as pilot:
        input_widget = captured[0].query_one(Input)
        input_widget.focus()
        input_widget.value = "/team inspect behavior"
        await pilot.press("enter")
        for _ in range(100):
            worker = captured[0]._turn_worker  # type: ignore[attr-defined]
            if worker is not None and not worker.is_running:
                break
            await asyncio.sleep(0.01)
        transcript = "\n".join(line.text for line in captured[0].query_one(RichLog).lines)
        assert "background done" in transcript
        assert "Unknown or unavailable" not in transcript


async def test_tui_memory_skill_and_unknown_slash_stay_local(
    tmp_path: Path, monkeypatch: Any
) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    paths.workspace.mkdir()
    skill_file = paths.project_dir / "skills" / "demo" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(
        "---\nname: demo\ndescription: local demo\n---\nUse locally.",
        encoding="utf-8",
    )
    skills = SkillRegistry(paths)
    skills.reload()
    client = BlockingClient()
    agent = Agent(
        client,
        ToolRegistry(paths.workspace),
        "system",
        memory_store=MemoryStore(paths),
    )
    captured: list[App[Any]] = []
    monkeypatch.setattr(App, "run", lambda self: captured.append(self))
    run_tui(agent, workspace=paths.workspace, skill_registry=skills)

    async with captured[0].run_test() as pilot:
        input_widget = captured[0].query_one(Input)
        input_widget.focus()
        for command in (
            "/init",
            "/save managed locally",
            "/memory search managed",
            "/skill off demo",
            "/unknown-command secret-local-intent",
        ):
            input_widget.value = command
            await pilot.press("enter")
            for _ in range(50):
                worker = captured[0]._turn_worker  # type: ignore[attr-defined]
                if worker is not None and not worker.is_running:
                    break
                await asyncio.sleep(0.01)
        transcript = "\n".join(line.text for line in captured[0].query_one(RichLog).lines)
        assert "managed locally" in transcript
        assert "Unknown or unavailable TUI command" in transcript
        assert not skills.skills["demo"].enabled
        assert (paths.workspace / "KAIRO.md").is_file()
        assert client.calls == 0


async def test_tui_model_and_config_are_local_and_secret_safe(
    tmp_path: Path, monkeypatch: Any
) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    paths.workspace.mkdir()
    config = AppConfig.load(paths)
    client = BlockingClient()
    agent = Agent(
        client,
        ToolRegistry(paths.workspace),
        "system",
        memory_store=MemoryStore(paths),
    )
    captured: list[App[Any]] = []
    monkeypatch.setattr(App, "run", lambda self: captured.append(self))
    run_tui(agent, workspace=paths.workspace, app_config=config)

    async with captured[0].run_test() as pilot:
        input_widget = captured[0].query_one(Input)
        input_widget.focus()
        for command in (
            "/config provider glm api-key top-secret-value",
            "/model moonshot",
        ):
            input_widget.value = command
            await pilot.press("enter")
            for _ in range(50):
                worker = captured[0]._turn_worker  # type: ignore[attr-defined]
                if worker is not None and not worker.is_running:
                    break
                await asyncio.sleep(0.01)
        transcript = "\n".join(line.text for line in captured[0].query_one(RichLog).lines)
        assert "top-secret-value" not in transcript
        assert config.default_provider == "kimi"
        assert client.calls == 0
