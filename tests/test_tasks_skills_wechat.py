import asyncio
import importlib
import io
import json
import os
import sqlite3
from contextlib import closing
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import kairocli.tasks as tasks_module
from kairocli.agent import Agent, AgentCanceled
from kairocli.channels.store import ChannelStore
from kairocli.cli import _handle_task, _InteractiveWechatRuntime, handle_wechat
from kairocli.cli.wechat_ui import QR_SCAN_PROMPT, workspace_prompt
from kairocli.llm import LlmClient
from kairocli.models import LlmResponse, Message
from kairocli.paths import KairoPaths
from kairocli.skills import SkillRegistry
from kairocli.tasks import (
    MAX_TASK_OUTPUT_BYTES,
    MAX_TASK_PROMPT_BYTES,
    DurableTaskManager,
    DurableTaskStore,
)
from kairocli.tools import ToolRegistry
from kairocli.web_auth import WebUserStore
from kairocli.wechat import (
    IlinkClient,
    WechatAccount,
    WechatAccountStore,
    WechatChannel,
    WechatMediaItem,
    WechatMessage,
    WechatPolicy,
    daemon_command,
    daemon_paths,
    format_wechat_text,
    split_message,
)

wechat_module = importlib.import_module("kairocli.channels.wechat.daemon")
wechat_accounts_module = importlib.import_module("kairocli.channels.wechat.accounts")
cli_interactive_module = importlib.import_module("kairocli.cli.interactive")


def test_wechat_setup_prompts_make_each_next_action_explicit(tmp_path: Path) -> None:
    prompt = workspace_prompt(tmp_path)
    assert prompt.startswith("\nConnect WeChat")
    assert f"Workspace: {tmp_path}" in prompt
    assert "Press Enter to connect, enter another directory, or Ctrl+C to cancel" in prompt
    assert QR_SCAN_PROMPT.startswith("Next: Scan this QR code")


def test_durable_task_lifecycle(tmp_path: Path) -> None:
    store = DurableTaskStore(tmp_path / "tasks.db")
    task = store.add("do work")
    claimed = store.claim()
    assert claimed and claimed.id == task.id and claimed.status == "running"
    assert claimed.started_at
    store.update(task.id, "completed", "ok")
    completed = store.get(task.id)
    assert completed and completed.output == "ok" and completed.finished_at


def test_durable_task_validates_prompt_and_migrates_legacy_database(tmp_path: Path) -> None:
    database = tmp_path / "legacy.db"
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(
            "CREATE TABLE tasks (id TEXT PRIMARY KEY, prompt TEXT NOT NULL, "
            "status TEXT NOT NULL, output TEXT NOT NULL DEFAULT '', "
            "error TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, "
            "updated_at TEXT NOT NULL)"
        )
    store = DurableTaskStore(database)
    with pytest.raises(ValueError, match="empty"):
        store.add("  \n")
    task = store.add("  durable work  ")
    assert task.id.startswith("task_")
    assert task.prompt == "durable work"
    assert task.duration_ms == 0


def test_durable_task_store_bounds_private_data_and_retention(tmp_path: Path) -> None:
    database = tmp_path / "private" / "tasks.db"
    store = DurableTaskStore(database, max_terminal_tasks=2)
    if os.name == "posix":
        assert database.stat().st_mode & 0o777 == 0o600
        assert database.parent.stat().st_mode & 0o777 == 0o700
    with pytest.raises(ValueError, match="exceeds"):
        store.add("x" * (MAX_TASK_PROMPT_BYTES + 1))

    ids: list[str] = []
    for index in range(3):
        task = store.add(f"work {index}")
        ids.append(task.id)
        assert store.claim()
        store.update(
            task.id,
            "completed",
            "界" * MAX_TASK_OUTPUT_BYTES,
            "token=should-not-survive",
        )
    assert store.get(ids[0]) is None
    retained = store.get(ids[-1])
    assert retained
    assert len(retained.output.encode("utf-8")) <= MAX_TASK_OUTPUT_BYTES
    assert retained.output.endswith("[task output truncated]")


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm"])
def test_durable_task_store_rechecks_database_symlinks(tmp_path: Path, suffix: str) -> None:
    database = tmp_path / "tasks.db"
    store = DurableTaskStore(database)
    store.add("safe work")
    attacked = Path(str(database) + suffix)
    target = tmp_path / f"outside{suffix or '-db'}"
    if suffix:
        target.touch()
        attacked.unlink(missing_ok=True)
    else:
        database.replace(target)
    attacked.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        store.list()


def test_durable_task_store_rejects_parent_symlink(tmp_path: Path) -> None:
    target = tmp_path / "outside"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        DurableTaskStore(linked / "tasks.db")


def test_durable_task_store_rejects_high_level_state_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    paths.home.mkdir(parents=True)
    outside = tmp_path / "outside-task-state"
    outside.mkdir()
    paths.user_dir.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        DurableTaskStore(paths.task_database)

    assert list(outside.iterdir()) == []


def test_durable_task_store_terminal_state_cannot_be_overwritten(tmp_path: Path) -> None:
    store = DurableTaskStore(tmp_path / "safe.db")

    store = DurableTaskStore(tmp_path / "safe.db")
    task = store.add("work")
    assert store.claim()
    assert store.cancel(task.id)
    assert not store.update(task.id, "completed", "late answer")
    canceled = store.get(task.id)
    assert canceled and canceled.status == "canceled" and not canceled.output
    assert store.get("../../outside") is None


def test_durable_task_store_closes_connections_and_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connections: list[sqlite3.Connection] = []
    closed: set[int] = set()
    original_connect = sqlite3.connect

    class TrackingConnection(sqlite3.Connection):
        def close(self) -> None:
            closed.add(id(self))
            super().close()

    def tracking_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        kwargs["factory"] = TrackingConnection
        connection = original_connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(tasks_module.sqlite3, "connect", tracking_connect)
    store = DurableTaskStore(tmp_path / "closed.db")
    task = store.add("rollback work")
    assert store.claim()

    def fail_prune(_connection: sqlite3.Connection) -> None:
        raise RuntimeError("prune failed")

    monkeypatch.setattr(store, "_prune_terminal", fail_prune)
    with pytest.raises(RuntimeError, match="prune failed"):
        store.update(task.id, "completed", "must roll back")

    assert connections and closed == {id(connection) for connection in connections}
    with closing(original_connect(store.database)) as connection, connection:
        row = connection.execute(
            "SELECT status, output FROM tasks WHERE id=?", (task.id,)
        ).fetchone()
    assert row == ("running", "")


def test_task_command_outputs_reusable_id_and_structured_log(tmp_path: Path) -> None:
    store = DurableTaskStore(tmp_path / "commands.db")
    manager = DurableTaskManager(store, lambda: None, workers=1)

    class Console:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def print(self, message: str) -> None:
            self.messages.append(message)

    console = Console()
    _handle_task("add inspect the queue", store, manager, console)
    task = store.list(1)[0]
    assert task.id in console.messages[-1]
    _handle_task("list 1", store, manager, console)
    assert task.id in console.messages[-1]
    _handle_task(f"log {task.id}", store, manager, console)
    assert f"Background task {task.id}" in console.messages[-1]
    assert "Status: enqueued" in console.messages[-1]
    assert "Task:\ninspect the queue" in console.messages[-1]
    _handle_task(f"cancel {task.id}", store, manager, console)
    assert console.messages[-1] == f"Cancellation requested: {task.id}"


def test_skill_discovery_and_state(tmp_path: Path) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    skill_file = paths.project_dir / "skills" / "demo" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(
        "---\nname: demo\ndescription: Demo skill\n---\nInstructions", encoding="utf-8"
    )
    registry = SkillRegistry(paths)
    registry.reload()
    assert "demo" in registry.skills
    registry.set_enabled("demo", False)
    assert "demo" not in registry.index()
    assert "web-access" in registry.index()


def test_wechat_noninteractive_policy() -> None:
    policy = WechatPolicy()
    assert policy.allow_tool("read_file", {"path": "README.md"})
    assert policy.allow_tool("query_code_graph", {"symbol": "UserService"})
    assert not policy.allow_tool("execute_command", {"command": "git status"})
    assert policy.allow_tool("write_file", {"path": "x"})
    assert policy.allow_tool("create_project", {"path": "x"})
    assert not policy.allow_tool("revert_turn", {"steps": 1})
    configured = WechatPolicy(command_allowlist=("git status",), mcp_allowlist=("chrome-devtools",))
    assert configured.allow_tool("execute_command", {"command": "git status"})
    assert not configured.allow_tool("execute_command", {"command": "git status && rm -rf src"})
    assert configured.allow_tool("mcp__chrome-devtools__take_snapshot", {})


async def test_interactive_wechat_runtime_routes_setup_and_start(tmp_path: Path) -> None:
    class Console:
        def print(self, _value: object) -> None:
            pass

    runtime = _InteractiveWechatRuntime(
        KairoPaths.discover(tmp_path, tmp_path / "home"),
        object(),  # type: ignore[arg-type]
        Console(),
    )
    events: list[str] = []

    async def setup() -> None:
        events.append("setup")

    async def start() -> str:
        events.append("start")
        return "started"

    async def close() -> None:
        events.append("close")

    runtime._setup = setup  # type: ignore[method-assign]
    runtime._start = start  # type: ignore[method-assign]
    runtime.close = close  # type: ignore[method-assign]

    assert await runtime.command("setup") == "started"
    assert events == ["close", "setup", "start"]
    help_text = await runtime.command("")
    assert "WeChat · not connected" in help_text
    assert "/wechat setup" in help_text
    assert "type `/wechat `" in help_text
    assert events == ["close", "setup", "start"]
    assert "Unknown WeChat command" in await runtime.command("unknown")


async def test_interactive_wechat_setup_can_be_canceled(tmp_path: Path) -> None:
    class Console:
        def print(self, _value: object) -> None:
            pass

    runtime = _InteractiveWechatRuntime(
        KairoPaths.discover(tmp_path, tmp_path / "home"),
        object(),  # type: ignore[arg-type]
        Console(),
    )
    started = False

    async def canceled_setup() -> None:
        raise cli_interactive_module._WechatSetupCanceled

    async def start() -> str:
        nonlocal started
        started = True
        return "started"

    runtime._setup = canceled_setup  # type: ignore[method-assign]
    runtime._start = start  # type: ignore[method-assign]
    assert await runtime.command("setup") == "WeChat setup canceled."
    assert started is False


async def test_interactive_wechat_runtime_cleans_up_after_channel_exit(
    tmp_path: Path,
) -> None:
    class Console:
        def print(self, _value: object) -> None:
            pass

    events: list[str] = []

    class Component:
        def __init__(self, label: str) -> None:
            self.label = label

        async def close(self) -> None:
            events.append(self.label)

    runtime = _InteractiveWechatRuntime(
        KairoPaths.discover(tmp_path, tmp_path / "home"),
        object(),  # type: ignore[arg-type]
        Console(),
    )
    channel = SimpleNamespace(running=True)
    runtime.channel = channel
    runtime.manager = Component("manager")  # type: ignore[assignment]
    runtime.agent = SimpleNamespace(tools=Component("tools"))  # type: ignore[assignment]
    task = asyncio.create_task(asyncio.sleep(0))
    runtime.channel_task = task
    task.add_done_callback(runtime._on_channel_done)

    await task
    for _ in range(20):
        if runtime.cleanup_task is None and runtime.manager is None:
            break
        await asyncio.sleep(0)

    assert events == ["manager", "tools"]
    assert not channel.running
    assert runtime.channel_task is None
    assert runtime.manager is None
    assert runtime.agent is None


def test_wechat_message_parsing() -> None:
    message = IlinkClient._parse_message(
        {
            "message_id": "m1",
            "from_user_id": "u1",
            "context_token": "ctx",
            "item_list": [
                {"text_item": {"text": "hello"}},
                {
                    "file_item": {
                        "file_name": "report.txt",
                        "mime_type": "application/pdf",
                        "media": {
                            "encrypt_query_param": "file-query",
                            "aes_key": "file-secret",
                        },
                    }
                },
                {
                    "image_item": {
                        "mime_type": "image/jpeg",
                        "aeskey": "image-secret",
                        "cdn_media": {"encrypt_query_param": "image-query"},
                    }
                },
            ],
        }
    )
    assert message.message_id == "m1"
    assert message.text == "hello\n[User sent a file: report.txt]"
    assert message.media_items == (
        WechatMediaItem("file", "report.txt", "application/pdf", "file-query", "file-secret"),
        WechatMediaItem("image", "", "image/jpeg", "image-query", "image-secret"),
    )
    assert message.media_items[1].is_image


def test_wechat_message_parser_does_not_coerce_identity_or_metadata_types() -> None:
    message = IlinkClient._parse_message(
        {
            "message_id": True,
            "from_user_id": {"spoof": "bound-user"},
            "context_token": "bad\ud800",
            "item_list": [
                {"text_item": {"text": {"not": "text"}}},
                {
                    "file_item": {
                        "file_name": ["not", "a", "name"],
                        "mime_type": {"not": "mime"},
                        "media": {
                            "encrypt_query_param": ["not", "query"],
                            "aes_key": {"not": "key"},
                        },
                    }
                },
            ],
        }
    )

    assert message.message_id == ""
    assert message.from_user_id == ""
    assert message.context_token == ""
    assert message.text == "[User sent a file: unknown]"
    assert message.media_items == (WechatMediaItem("file", "unknown"),)


def test_wechat_message_chunking() -> None:
    chunks = split_message("a" * 25, max_chars=10)
    assert chunks == ["a" * 10, "a" * 10, "a" * 5]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("#标题\n** 加粗 **", "**标题**\n**加粗**"),
        ("前![alt](https://example.com/a.png)后", "前后"),
        ("##### 小标题\n#一级", "小标题\n**一级**"),
        ("*中文强调* **加粗** `code` *ascii*", "中文强调 **加粗** `code` *ascii*"),
        ("\x1b[1m\x1b[32m■\x1b[0m 你好", "你好"),
    ],
)
def test_wechat_markdown_formatter(raw: str, expected: str) -> None:
    assert format_wechat_text(raw) == expected


def test_wechat_formatter_handles_tables_and_code_fences() -> None:
    table = "| API | 用途 |\n| --- | --- |\n| getupdates | 长轮询 |\n| sendmessage | 发送回复 |"
    assert format_wechat_text(table) == ("- **getupdates**：长轮询\n- **sendmessage**：发送回复")
    collapsed = (
        "|---|---|| AppID / AppSecret |每个微信应用的身份凭证 || "
        "Access Token |调用服务端 API 的短期凭证 |"
    )
    assert format_wechat_text(collapsed) == (
        "- **AppID / AppSecret**：每个微信应用的身份凭证\n"
        "- **Access Token**：调用服务端 API 的短期凭证"
    )
    prose = "```\n用户发消息 → 微信服务器 → 处理并返回 XML\n```"
    assert format_wechat_text(prose) == ("用户发消息\n→ 微信服务器\n→ 处理并返回 XML")
    code = '```python\nprint("hi")\n```'
    assert format_wechat_text(code) == code


async def test_wechat_channel_queues_work_and_stop_bypasses_active_turn() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send_text(self, account: object, user: str, context: str, text: str) -> None:
            self.sent.append(text)

        async def send_typing(self, account: object, user: str, context: str, status: int) -> None:
            return None

    class FakeAgent:
        def __init__(self) -> None:
            self.prompts: list[str] = []
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.canceled = False

        async def run(self, prompt: str) -> str:
            self.canceled = False
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                self.started.set()
                await self.release.wait()
                if self.canceled:
                    raise AgentCanceled("canceled")
            return f"answer: {prompt}"

        def cancel(self) -> None:
            self.canceled = True
            self.release.set()

        def clear(self) -> None:
            return None

        async def compact(self) -> bool:
            return False

    client = FakeClient()
    agent = FakeAgent()
    account = WechatAccount("token", "bot", "https://example.test", "user", ".")
    channel = WechatChannel(client, object(), account, agent)  # type: ignore[arg-type]
    first = WechatMessage("1", "user", "ctx", "first")
    second = WechatMessage("2", "user", "ctx", "second")

    await channel.handle(first)
    await agent.started.wait()
    await channel.handle(second)
    assert len(channel.queue) == 1
    await channel.handle(WechatMessage("3", "user", "ctx", "/stop"))
    assert channel.active_task is not None
    await channel.active_task
    await channel._reap_active()
    assert channel.active_task is not None
    await channel.active_task
    await channel._reap_active()

    assert agent.prompts == ["first", "second"]
    assert any("Cancellation requested" in text for text in client.sent)
    assert any("Task canceled" in text for text in client.sent)
    assert any("answer: second" in text for text in client.sent)


async def test_wechat_workspace_command_switches_agent_and_thread() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send_text(self, *args: object) -> None:
            self.sent.append(str(args[-1]))

    class FakeTools:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    class FakeAgent:
        def __init__(self) -> None:
            self.tools = FakeTools()

        def cancel(self) -> None:
            return None

    old_agent = FakeAgent()
    new_agent = FakeAgent()

    async def switch(argument: str) -> tuple[Any, str, str, str]:
        assert argument == "2"
        return new_agent, "thread_two", "/work/two", "已切换到工作区：two"

    client = FakeClient()
    channel = WechatChannel(
        client,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        WechatAccount("token", "bot", "https://example.test", "user", "/work/one"),
        old_agent,  # type: ignore[arg-type]
        workspace_switcher=switch,
    )

    assert await channel.handle(WechatMessage("1", "user", "ctx", "/workspace 2"))
    assert channel.active_task is not None
    await channel.active_task
    await channel._reap_active()

    assert channel.agent is new_agent
    assert channel.runtime_thread_id == "thread_two"
    assert channel.account.workspace == "/work/two"
    assert old_agent.tools.closed is True
    assert client.sent == ["已切换到工作区：two"]


async def test_wechat_media_only_message_passes_metadata_notice_without_keys() -> None:
    class FakeClient:
        sent: list[str] = []

        async def send_text(self, *args: object) -> None:
            self.sent.append(str(args[-1]))

        async def send_typing(self, *args: object) -> None:
            return None

    class FakeAgent:
        prompt = ""

        async def run(self, prompt: str) -> str:
            self.prompt = prompt
            return "done"

        def cancel(self) -> None:
            return None

    client = FakeClient()
    agent = FakeAgent()
    account = WechatAccount("token", "bot", "https://example.test", "user", ".")
    channel = WechatChannel(client, object(), account, agent)  # type: ignore[arg-type]
    media = WechatMediaItem("image", "", "image/png", "query", "top-secret-key")

    await channel.handle(WechatMessage("media-1", "user", "ctx", "", (media,)))
    assert channel.active_task is not None
    await channel.active_task
    await channel._reap_active()
    assert "Media message: 1 item" in agent.prompt
    assert "metadata only" in agent.prompt
    assert "top-secret-key" not in agent.prompt
    assert client.sent == ["done"]


def test_wechat_account_store_is_private_atomic_and_validated(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    store = WechatAccountStore(paths)
    account = WechatAccount(
        "secret-token",
        "bot",
        "https://ilinkai.weixin.qq.com",
        "bound-user",
        str(workspace),
    )

    store.save(account)

    assert store.load() == account
    assert not list(store.file.parent.glob(".account-*.tmp"))
    if os.name != "nt":
        assert store.file.stat().st_mode & 0o777 == 0o600
        assert store.file.parent.stat().st_mode & 0o777 == 0o700

    invalid = WechatAccount("token", "bot", "http://example.test", "user", str(workspace))
    with pytest.raises(ValueError, match="HTTPS"):
        store.save(invalid)

    normalized = WechatAccount("token", "bot", "HTTPS://EXAMPLE.TEST/root/", "user", str(workspace))
    store.save(normalized)
    assert normalized.base_url == "https://example.test/root"
    assert store.load() == normalized


async def test_legacy_wechat_binding_migrates_to_web_and_leaves_backup(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    legacy = WechatAccountStore(paths)
    legacy.save(
        WechatAccount(
            "secret-token",
            "bot",
            "https://ilinkai.weixin.qq.com",
            "bound-user",
            str(workspace),
        )
    )
    users = WebUserStore(paths.user_dir / "web" / "users.db")
    user = users.create_user("alice@example.com", "password-123")
    users.add_workspace(user.id, str(workspace))

    assert await handle_wechat(paths, SimpleNamespace(), "migrate-web", None, user.id) == 0

    assert not legacy.file.exists()
    assert legacy.file.with_name("account.migrated.json").is_file()
    channels = ChannelStore(paths.user_dir / "web" / "users.db")
    binding = channels.binding_for_user(user.id, "wechat")
    assert binding is not None and binding.enabled is False
    assert channels.wechat_credentials(binding.id).token == "secret-token"


@pytest.mark.parametrize("kind", ["duplicate", "nonfinite", "overdeep"])
def test_wechat_account_store_rejects_ambiguous_or_pathological_json(
    tmp_path: Path, kind: str
) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    store = WechatAccountStore(paths)
    store.file.parent.mkdir(parents=True)
    account = WechatAccount("secret-token", "bot", "https://example.test", "user", str(workspace))
    canonical = json.dumps(asdict(account))
    if kind == "duplicate":
        payload = canonical[:-1] + ', "token": "replacement"}'
    elif kind == "nonfinite":
        payload = canonical[:-1] + ', "unknown": NaN}'
    else:
        payload = canonical[:-1] + ', "unknown": ' + "[" * 20 + "0" + "]" * 20 + "}"
    store.file.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="Cannot read WeChat account: ValueError"):
        store.load()


def test_wechat_sync_update_cannot_overwrite_rebound_account(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    store = WechatAccountStore(paths)
    original = WechatAccount(
        "old-token", "old-bot", "https://example.test", "old-user", str(workspace)
    )
    store.save(original)
    stale = store.load()
    assert stale is not None
    rebound = WechatAccount(
        "new-token", "new-bot", "https://example.test", "new-user", str(workspace)
    )
    store.save(rebound)

    assert not store.update_sync_buf(stale, "stale-cursor")
    assert store.load() == rebound
    assert store.update_sync_buf(rebound, "fresh-cursor")
    assert store.load() == replace(rebound, sync_buf="fresh-cursor")


def test_wechat_account_lock_rejects_symlink_without_external_write(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    store = WechatAccountStore(paths)
    store.root.mkdir(parents=True)
    outside = tmp_path / "outside-account-lock"
    outside.write_text("sentinel", encoding="utf-8")
    (store.root / ".account.lock").symlink_to(outside)
    account = WechatAccount("token", "bot", "https://example.test", "user", str(workspace))

    with pytest.raises(ValueError, match="symlink"):
        store.save(account)

    assert outside.read_text(encoding="utf-8") == "sentinel"
    assert not store.file.exists()


@pytest.mark.parametrize(
    "base_url",
    [
        "https://user:secret@example.test",
        "https://example.test?token=secret",
        "https://example.test/#fragment",
        "https://example.test/ bad",
        "https:\\example.test",
        "https://example.test:99999",
    ],
)
def test_wechat_account_store_rejects_ambiguous_base_urls(tmp_path: Path, base_url: str) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    account = WechatAccount("token", "bot", base_url, "user", str(workspace))

    with pytest.raises(ValueError, match="WeChat base URL"):
        WechatAccountStore(paths).save(account)


def test_wechat_account_store_rejects_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    store = WechatAccountStore(paths)
    store.file.parent.mkdir(parents=True)
    target = tmp_path / "outside-account.json"
    target.write_text("{}", encoding="utf-8")
    store.file.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        store.load()


def test_wechat_account_read_bounds_growth_after_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    store = WechatAccountStore(paths)
    store.file.parent.mkdir(parents=True)
    store.file.write_text("{}", encoding="utf-8")
    requested: list[int] = []

    class GrowingReader(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            requested.append(size)
            return super().read(size)

    def growing_open(
        _path: Path, mode: str = "r", *args: object, **kwargs: object
    ) -> GrowingReader:
        assert mode == "rb"
        return GrowingReader(b"x" * 9)

    monkeypatch.setattr(wechat_accounts_module, "MAX_WECHAT_ACCOUNT_BYTES", 8)
    monkeypatch.setattr(Path, "open", growing_open)

    with pytest.raises(ValueError, match="128 KiB"):
        store.load()
    assert requested == [9]


def test_wechat_account_store_rejects_parent_symlink_without_external_writes(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    paths.user_dir.mkdir(parents=True)
    outside = tmp_path / "outside-wechat"
    outside.mkdir()
    (paths.user_dir / "wechat").symlink_to(outside, target_is_directory=True)
    store = WechatAccountStore(paths)
    account = WechatAccount("secret-token", "bot", "https://example.test", "user", str(workspace))

    for operation in (store.load, lambda: store.save(account), store.clear, store.media_dir):
        with pytest.raises(ValueError, match="symlink"):
            operation()

    assert list(outside.iterdir()) == []


def test_wechat_media_directory_is_private_and_rejects_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    store = WechatAccountStore(paths)
    directory = store.media_dir()
    if os.name != "nt":
        assert directory.stat().st_mode & 0o777 == 0o700

    external = tmp_path / "external-media"
    external.mkdir()
    directory.rmdir()
    directory.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        store.media_dir()


async def test_wechat_transport_is_bounded_and_rejects_redirects() -> None:
    class AllowPolicy:
        async def check(self, url: str) -> None:
            return None

        async def acquire(self) -> None:
            return None

    async def ok_handler(request: httpx.Request) -> httpx.Response:
        assert request.content == b'{"value":"ok"}'
        return httpx.Response(200, json={"ok": True})

    client = IlinkClient(
        network_policy=AllowPolicy(),  # type: ignore[arg-type]
        transport=httpx.MockTransport(ok_handler),
    )
    assert await client._request("POST", "test", {"value": "ok"}) == {"ok": True}

    async def large_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * (2 * 1024 * 1024 + 1))

    client.transport = httpx.MockTransport(large_handler)
    with pytest.raises(RuntimeError, match="2 MiB"):
        await client._request("GET", "test")

    async def redirect_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://example.test/other"})

    client.transport = httpx.MockTransport(redirect_handler)
    with pytest.raises(RuntimeError, match="redirect"):
        await client._request("GET", "test")

    with pytest.raises(ValueError, match="1 MiB"):
        await client._request("POST", "test", {"value": "x" * (1024 * 1024)})


@pytest.mark.parametrize(
    "response_body",
    [
        b'{"ok":true,"ok":false}',
        b'{"score":NaN}',
        b'{"nested":' + b"[" * 40 + b"0" + b"]" * 40 + b"}",
        b'{"text":"\xff"}',
        b'{"items":[' + b"0," * 200_000 + b"0]}",
    ],
)
async def test_wechat_transport_rejects_ambiguous_or_unbounded_json(
    response_body: bytes,
) -> None:
    class AllowPolicy:
        async def check(self, url: str) -> None:
            return None

        async def acquire(self) -> None:
            return None

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=response_body)

    client = IlinkClient(
        network_policy=AllowPolicy(),  # type: ignore[arg-type]
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(RuntimeError, match="invalid or unsafe JSON"):
        await client._request("GET", "test")


async def test_wechat_invalid_request_fails_before_network_policy() -> None:
    class CountingPolicy:
        checks = 0
        acquires = 0

        async def check(self, url: str) -> None:
            self.checks += 1

        async def acquire(self) -> None:
            self.acquires += 1

    policy = CountingPolicy()
    client = IlinkClient(network_policy=policy)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="finite standard JSON"):
        await client._request("POST", "test", {"score": float("nan")})

    assert policy.checks == 0
    assert policy.acquires == 0


async def test_wechat_update_fields_use_exact_types(tmp_path: Path) -> None:
    class AllowPolicy:
        async def check(self, url: str) -> None:
            return None

        async def acquire(self) -> None:
            return None

    responses = [
        {"ret": True, "longpolling_timeout_ms": 1.5, "get_updates_buf": "next"},
        {"ret": 0, "get_updates_buf": {"not": "a cursor"}},
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=responses.pop(0))

    account = WechatAccount(
        "token", "account", "https://ilinkai.weixin.qq.com", "user", str(tmp_path)
    )
    client = IlinkClient(
        network_policy=AllowPolicy(),  # type: ignore[arg-type]
        transport=httpx.MockTransport(handler),
    )

    update = await client.get_updates(account, 12_345)
    assert update.code == -1
    assert update.timeout_ms == 12_345
    assert update.sync_buf == "next"
    with pytest.raises(RuntimeError, match="sync buffer is invalid"):
        await client.get_updates(account)


async def test_wechat_confirmed_login_rejects_coerced_credentials() -> None:
    class AllowPolicy:
        async def check(self, url: str) -> None:
            return None

        async def acquire(self) -> None:
            return None

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "confirmed",
                "bot_token": 12345,
                "ilink_bot_id": "bot",
                "baseurl": "https://ilinkai.weixin.qq.com",
                "ilink_user_id": "user",
            },
        )

    client = IlinkClient(
        network_policy=AllowPolicy(),  # type: ignore[arg-type]
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        await client.poll_qr_status("qr-id")


def test_wechat_daemon_logs_are_private_rotated_and_tailed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    WechatAccountStore(paths).save(
        WechatAccount(
            "token",
            "account",
            "https://ilinkai.weixin.qq.com",
            "user",
            str(workspace),
        )
    )
    pid_file, stdout_file, stderr_file = daemon_paths(paths)
    stdout_file.parent.mkdir(parents=True)
    stdout_file.write_bytes(b"old" + b"x" * (5 * 1024 * 1024))
    captured: list[object] = []

    class FakeProcess:
        pid = 987_654

    def fake_popen(*args: object, **kwargs: object) -> FakeProcess:
        captured.extend([kwargs["stdout"], kwargs["stderr"]])
        assert not kwargs["stdout"].closed
        assert not kwargs["stderr"].closed
        return FakeProcess()

    monkeypatch.setattr(wechat_module.subprocess, "Popen", fake_popen)
    result = daemon_command(paths, "start")

    assert "987654" in result
    assert all(handle.closed for handle in captured)  # type: ignore[union-attr]
    assert stdout_file.with_name("stdout.log.1").is_file()
    assert pid_file.read_text(encoding="utf-8") == "987654"
    if os.name != "nt":
        assert pid_file.stat().st_mode & 0o777 == 0o600
        assert stdout_file.stat().st_mode & 0o777 == 0o600

    stdout_file.write_text("\n".join(f"line-{index}" for index in range(500)), encoding="utf-8")
    tail = daemon_command(paths, "logs")
    assert "line-499" in tail
    assert "line-0\n" not in tail
    assert len(tail.splitlines()) == 100
    pid_file.unlink()
    stderr_file.unlink(missing_ok=True)


def test_wechat_daemon_refuses_to_spawn_without_bound_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    spawned = False

    def fake_popen(*_args: object, **_kwargs: object) -> None:
        nonlocal spawned
        spawned = True

    monkeypatch.setattr(wechat_module.subprocess, "Popen", fake_popen)
    with pytest.raises(RuntimeError, match="wechat setup"):
        daemon_command(paths, "start")

    assert spawned is False
    pid_file, stdout_file, stderr_file = daemon_paths(paths)
    assert not pid_file.exists()
    assert not stdout_file.exists()
    assert not stderr_file.exists()


def test_wechat_daemon_rejects_pid_symlink_before_start(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    pid_file, _, _ = daemon_paths(paths)
    pid_file.parent.mkdir(parents=True)
    target = tmp_path / "outside.pid"
    target.write_text("1", encoding="utf-8")
    pid_file.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        daemon_command(paths, "start")


def test_wechat_daemon_pid_read_bounds_growth_before_process_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid_file = tmp_path / "daemon.pid"
    pid_file.write_text("1", encoding="utf-8")
    requested: list[int] = []
    process_probed = False

    class GrowingReader(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            requested.append(size)
            return super().read(size)

    def growing_open(
        _path: Path, mode: str = "r", *args: object, **kwargs: object
    ) -> GrowingReader:
        assert mode == "rb"
        return GrowingReader(b"1" * 65)

    def unexpected_kill(_pid: int, _signal: int) -> None:
        nonlocal process_probed
        process_probed = True

    monkeypatch.setattr(Path, "open", growing_open)
    monkeypatch.setattr(wechat_module.os, "kill", unexpected_kill)

    assert wechat_module._read_live_pid(pid_file) is None
    assert requested == [65]
    assert process_probed is False


def test_wechat_daemon_rejects_symlinked_log_container(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    pid_file, stdout_file, _ = daemon_paths(paths)
    pid_file.parent.mkdir(parents=True)
    outside = tmp_path / "outside-logs"
    outside.mkdir()
    stdout_file.parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        daemon_command(paths, "logs")

    assert list(outside.iterdir()) == []


class TaskClient(LlmClient):
    provider = "task"
    model = "task"

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        return LlmResponse(content="finished")


async def test_durable_task_worker(tmp_path: Path) -> None:
    store = DurableTaskStore(tmp_path / "worker.db")
    task = store.add("background work")
    manager = DurableTaskManager(
        store, lambda: Agent(TaskClient(), ToolRegistry(tmp_path), "system"), workers=1
    )
    manager.start()
    for _ in range(30):
        current = store.get(task.id)
        if current and current.status == "completed":
            break
        await asyncio.sleep(0.02)
    await manager.close()
    current = store.get(task.id)
    assert current and current.status == "completed" and current.output == "finished"


async def test_durable_task_manager_requeues_interrupted_running_work(tmp_path: Path) -> None:
    store = DurableTaskStore(tmp_path / "requeue.db")
    task = store.add("resume later")
    manager = DurableTaskManager(
        store, lambda: Agent(TaskClient(), ToolRegistry(tmp_path), "system"), workers=1
    )
    assert store.claim(manager._owner_token, manager._owner_pid)  # noqa: SLF001
    await manager.close()
    current = store.get(task.id)
    assert current and current.status == "enqueued" and current.started_at == ""


def test_durable_task_live_owner_is_not_requeued_by_second_store(tmp_path: Path) -> None:
    database = tmp_path / "shared.db"
    first = DurableTaskStore(database)
    task = first.add("run exactly once")
    assert first.claim("live-owner", os.getpid())

    second = DurableTaskStore(database)

    current = second.get(task.id)
    assert current and current.status == "running"
    assert second.claim("second-owner", os.getpid()) is None


def test_durable_task_recovers_dead_owner_and_rejects_late_result(tmp_path: Path) -> None:
    database = tmp_path / "recovery.db"
    store = DurableTaskStore(database)
    task = store.add("recover after crash")
    assert store.claim("dead-owner", 99_999_999)

    recovered = DurableTaskStore(database)
    assert recovered.get(task.id).status == "enqueued"  # type: ignore[union-attr]
    assert recovered.claim("new-owner", os.getpid())
    assert not recovered.update(task.id, "completed", "stale", owner_token="dead-owner")
    assert recovered.update(task.id, "completed", "fresh", owner_token="new-owner")
    final = recovered.get(task.id)
    assert final and final.status == "completed" and final.output == "fresh"


def test_durable_task_claim_recovers_owner_that_died_after_store_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "live-recovery.db"
    first = DurableTaskStore(database)
    task = first.add("recover without restart")
    assert first.claim("first-owner", os.getpid())
    waiting_manager_store = DurableTaskStore(database)

    monkeypatch.setattr(tasks_module, "_process_is_alive", lambda _pid: False)
    reclaimed = waiting_manager_store.claim("replacement-owner", os.getpid())

    assert reclaimed and reclaimed.id == task.id and reclaimed.status == "running"


async def test_durable_task_close_bounds_uncooperative_worker_and_ignores_late_result(
    tmp_path: Path,
) -> None:
    store = DurableTaskStore(tmp_path / "stubborn-worker.db")
    task = store.add("resume safely")
    started = asyncio.Event()
    release = asyncio.Event()

    class Tools:
        async def close(self) -> None:
            return None

    class StubbornAgent:
        tools = Tools()

        async def run(self, prompt: str) -> str:
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            return "late success"

        def cancel(self) -> None:
            return None

    manager = DurableTaskManager(store, StubbornAgent, workers=1)
    manager.start()
    await asyncio.wait_for(started.wait(), 1)

    await asyncio.wait_for(manager.close(), 0.5)

    current = store.get(task.id)
    assert current and current.status == "enqueued"
    assert tasks_module._DETACHED_MANAGER_TASKS
    release.set()
    for _ in range(50):
        if not tasks_module._DETACHED_MANAGER_TASKS:
            break
        await asyncio.sleep(0.01)
    assert not tasks_module._DETACHED_MANAGER_TASKS
    current = store.get(task.id)
    assert current and current.status == "enqueued" and current.output == ""


async def test_durable_task_close_requeues_before_propagating_cancellation(
    tmp_path: Path,
) -> None:
    store = DurableTaskStore(tmp_path / "canceled-close.db")
    task = store.add("must not remain running")
    started = asyncio.Event()
    release = asyncio.Event()

    class Tools:
        async def close(self) -> None:
            return None

    class StubbornAgent:
        tools = Tools()

        async def run(self, prompt: str) -> str:
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            return "late"

        def cancel(self) -> None:
            return None

    manager = DurableTaskManager(store, StubbornAgent, workers=1)
    manager.start()
    await asyncio.wait_for(started.wait(), 1)
    closing = asyncio.create_task(manager.close())
    await asyncio.sleep(0)
    closing.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(closing, 0.5)
    current = store.get(task.id)
    assert current and current.status == "enqueued" and current.started_at == ""
    assert manager._close_task is not None  # noqa: SLF001
    assert manager._close_task.done()  # noqa: SLF001

    release.set()
    for _ in range(50):
        if not tasks_module._DETACHED_MANAGER_TASKS:
            break
        await asyncio.sleep(0.01)
    assert not tasks_module._DETACHED_MANAGER_TASKS


async def test_durable_task_restart_fences_detached_worker_generation(
    tmp_path: Path,
) -> None:
    store = DurableTaskStore(tmp_path / "restart-generation.db")
    task = store.add("restart safely")
    started = asyncio.Event()
    release = asyncio.Event()
    factory_calls = 0

    class Tools:
        async def close(self) -> None:
            return None

    class StubbornAgent:
        tools = Tools()

        async def run(self, prompt: str) -> str:
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            return "stale success"

        def cancel(self) -> None:
            return None

    class FreshAgent:
        tools = Tools()

        async def run(self, prompt: str) -> str:
            return "fresh success"

        def cancel(self) -> None:
            return None

    def factory() -> StubbornAgent | FreshAgent:
        nonlocal factory_calls
        factory_calls += 1
        return StubbornAgent() if factory_calls == 1 else FreshAgent()

    manager = DurableTaskManager(store, factory, workers=1)
    manager.start()
    first_owner = manager._owner_token  # noqa: SLF001
    await asyncio.wait_for(started.wait(), 1)
    await asyncio.wait_for(manager.close(), 0.5)

    manager.start()
    assert manager._owner_token != first_owner  # noqa: SLF001
    for _ in range(100):
        current = store.get(task.id)
        if current and current.status == "completed":
            break
        await asyncio.sleep(0.01)

    current = store.get(task.id)
    assert current and current.status == "completed" and current.output == "fresh success"
    release.set()
    for _ in range(100):
        if not tasks_module._DETACHED_MANAGER_TASKS:
            break
        await asyncio.sleep(0.01)
    await manager.close()

    assert not tasks_module._DETACHED_MANAGER_TASKS
    current = store.get(task.id)
    assert current and current.status == "completed" and current.output == "fresh success"


async def test_durable_task_worker_survives_factory_and_cleanup_failures(
    tmp_path: Path,
) -> None:
    store = DurableTaskStore(tmp_path / "resilient.db")
    first = store.add("factory failure")
    second = store.add("cleanup failure")
    third = store.add("still runs")
    calls = 0

    class BrokenTools:
        async def close(self) -> None:
            raise RuntimeError("cleanup secret=hidden")

    class LocalAgent:
        tools = BrokenTools()

        async def run(self, prompt: str) -> str:
            suffix = "\ud800" if prompt == "cleanup failure" else ""
            return f"done: {prompt}{suffix}"

        def cancel(self) -> None:
            return None

    class UnprintableError(RuntimeError):
        def __str__(self) -> str:
            raise KeyboardInterrupt

    def factory() -> LocalAgent:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise UnprintableError()
        return LocalAgent()

    manager = DurableTaskManager(store, factory, workers=1)
    manager.start()
    for _ in range(100):
        current = store.get(third.id)
        if current and current.status == "completed":
            break
        await asyncio.sleep(0.02)
    await manager.close()

    failed = store.get(first.id)
    assert failed and failed.status == "failed"
    assert failed.error == "UnprintableError message unavailable"
    second_result = store.get(second.id)
    assert second_result and second_result.status == "completed"
    assert second_result.output == "done: cleanup failure?"
    assert store.get(third.id).status == "completed"  # type: ignore[union-attr]


async def test_wechat_error_reply_survives_unprintable_exception() -> None:
    class FakeClient:
        sent: list[str] = []

        async def send_text(self, *args: object) -> None:
            self.sent.append(str(args[-1]))

        async def send_typing(self, *args: object) -> None:
            return None

    class UnprintableError(RuntimeError):
        def __str__(self) -> str:
            raise KeyboardInterrupt

    class FailingAgent:
        async def run(self, prompt: str) -> str:
            raise UnprintableError()

    client = FakeClient()
    channel = WechatChannel(
        client,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        WechatAccount("token", "bot", "https://example.test", "user", "."),
        FailingAgent(),  # type: ignore[arg-type]
    )
    await channel._process(WechatMessage("id", "user", "context", "hello"))

    assert any("UnprintableError message unavailable" in message for message in client.sent)


async def test_durable_task_terminal_callback_is_isolated(tmp_path: Path) -> None:
    store = DurableTaskStore(tmp_path / "callback.db")
    completed: list[str] = []

    async def callback(task: Any) -> None:
        completed.append(f"{task.id}:{task.status}")
        raise RuntimeError("notification UI disappeared")

    task = store.add("notify me")
    manager = DurableTaskManager(
        store,
        lambda: Agent(TaskClient(), ToolRegistry(tmp_path), "system"),
        workers=1,
        terminal_callback=callback,
    )
    manager.start()
    for _ in range(50):
        if completed:
            break
        await asyncio.sleep(0.02)
    await manager.close()
    assert completed == [f"{task.id}:completed"]
    assert store.get(task.id).status == "completed"  # type: ignore[union-attr]


async def test_durable_task_notification_consumes_store_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = DurableTaskStore(tmp_path / "callback-failure.db")
    manager = DurableTaskManager(
        store,
        lambda: Agent(TaskClient(), ToolRegistry(tmp_path), "system"),
        terminal_callback=lambda _task: None,
    )
    loop = asyncio.get_running_loop()
    unhandled: list[dict[str, object]] = []
    previous_handler = loop.get_exception_handler()

    def fail_get(_task_id: str) -> None:
        raise OSError("task store unavailable")

    monkeypatch.setattr(store, "get", fail_get)
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    try:
        manager._schedule_terminal_notification("task_000000000000")
        for _ in range(20):
            if not manager._notification_tasks:
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert manager._notification_tasks == set()
    assert unhandled == []
