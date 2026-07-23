import io
import json
import multiprocessing
import os
from pathlib import Path
from typing import Any

import pytest

import kairocli.memory as memory_module
from kairocli.agent import Agent
from kairocli.cli import make_agent
from kairocli.config import AppConfig
from kairocli.llm import LlmClient
from kairocli.memory import (
    MAX_MEMORY_ENTRIES,
    MAX_MEMORY_FACT_CHARS,
    MemoryStore,
    browser_login_fact,
    handle_memory_command,
    handle_save_command,
    tokenize_memory_query,
)
from kairocli.models import LlmResponse, Message
from kairocli.paths import KairoPaths
from kairocli.tools import ToolRegistry


def _concurrent_memory_writer(
    home: str, workspace: str, prefix: str, start: Any
) -> None:
    memory = MemoryStore(KairoPaths.discover(Path(workspace), Path(home)))
    start.wait(10)
    for number in range(20):
        memory.save(f"{prefix} fact {number}")


def test_memory_scopes_and_crud(tmp_path: Path) -> None:
    home = tmp_path / "home"
    one = MemoryStore(KairoPaths.discover(tmp_path / "one", home))
    two = MemoryStore(KairoPaths.discover(tmp_path / "two", home))
    local = one.save("project uses asyncio")
    global_entry = one.save("prefer concise output", "global")
    assert [entry.id for entry in two.list_entries()] == [global_entry.id]
    assert one.search("uses asyncio")[0].id == local.id
    assert one.delete(local.id)


def test_memory_tokenizer_and_ranked_chinese_search(tmp_path: Path) -> None:
    memory = MemoryStore(KairoPaths.discover(tmp_path / "project", tmp_path / "home"))
    relevant = memory.save("用户偏好使用 Java 开发")
    memory.save("项目使用 PostgreSQL 数据库")
    assert "偏好" in tokenize_memory_query("偏好设置")
    assert memory.search("偏好设置")[0].id == relevant.id
    assert memory.search("Java 偏好")[0].id == relevant.id


def test_memory_deduplicates_normalized_content(tmp_path: Path) -> None:
    memory = MemoryStore(KairoPaths.discover(tmp_path / "project", tmp_path / "home"))
    first = memory.save("project uses   asyncio")
    duplicate = memory.save("PROJECT USES ASYNCIO", "global")
    assert duplicate.id == first.id
    assert len(memory.list_entries()) == 1


def test_memory_commands_share_save_search_and_delete(tmp_path: Path) -> None:
    memory = MemoryStore(KairoPaths.discover(tmp_path / "project", tmp_path / "home"))
    saved = handle_save_command("--global prefer deterministic tests", memory)
    entry = memory.list_entries()[0]
    assert entry.id in saved and entry.scope == "global"
    assert "prefer deterministic tests" in handle_memory_command("search deterministic", memory)
    assert handle_memory_command(f"delete {entry.id}", memory) == "Deleted."
    assert handle_memory_command("list", memory) == "No visible memories."


def test_memory_context_respects_project_scope_and_token_budget(tmp_path: Path) -> None:
    home = tmp_path / "home"
    current = MemoryStore(KairoPaths.discover(tmp_path / "current", home))
    other = MemoryStore(KairoPaths.discover(tmp_path / "other", home))
    current.save("当前项目使用 Python 3.14")
    current.save("默认使用中文回答", "global")
    other.save("其他项目使用 Java 17")
    context = current.context_for_query("项目 使用 Python Java", max_tokens=100)
    assert "Python 3.14" in context
    assert "Java 17" not in context
    assert current.context_for_query("Python", max_tokens=1) == ""


def test_invalid_memory_file_degrades_to_empty(tmp_path: Path) -> None:
    paths = KairoPaths.discover(tmp_path / "project", tmp_path / "home")
    paths.memory_file.parent.mkdir(parents=True)
    paths.memory_file.write_text("not json", encoding="utf-8")
    assert MemoryStore(paths).list_entries() == []


@pytest.mark.parametrize(
    "payload",
    [
        '[{"id":"000000000001","fact":"first","fact":"second",'
        '"scope":"global","project":null,"created_at":"2026-01-01"}]',
        '[{"id":"000000000001","fact":"valid","scope":"global",'
        '"project":null,"created_at":"2026-01-01","unknown":NaN}]',
        '[{"id":"000000000001","fact":"valid","scope":"global",'
        '"project":null,"created_at":"2026-01-01","unknown":'
        + "[" * 20
        + "0"
        + "]" * 20
        + "}]",
    ],
)
def test_memory_strict_json_failures_degrade_and_recover(
    tmp_path: Path, payload: str
) -> None:
    paths = KairoPaths.discover(tmp_path / "project", tmp_path / "home")
    paths.memory_file.parent.mkdir(parents=True)
    paths.memory_file.write_text(payload, encoding="utf-8")
    memory = MemoryStore(paths)

    assert memory.list_entries() == []
    saved = memory.save("recovered memory")

    assert memory.list_entries() == [saved]


def test_memory_lock_rejects_non_regular_file(tmp_path: Path) -> None:
    paths = KairoPaths.discover(tmp_path / "project", tmp_path / "home")
    lock = paths.memory_file.parent / ".long_term_memory.lock"
    lock.parent.mkdir(parents=True)
    if os.name == "posix":
        os.mkfifo(lock)
        with pytest.raises(ValueError, match="not a regular file"):
            MemoryStore(paths).save("blocked")
    else:  # pragma: no cover - exercised on Windows CI
        lock.mkdir()
        with pytest.raises(OSError):
            MemoryStore(paths).save("blocked")

    assert not paths.memory_file.exists()


def test_memory_read_is_bounded_when_file_grows_after_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = KairoPaths.discover(tmp_path / "project", tmp_path / "home")
    paths.memory_file.parent.mkdir(parents=True)
    paths.memory_file.write_text("[]", encoding="utf-8")
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

    monkeypatch.setattr(memory_module, "MAX_MEMORY_FILE_BYTES", 8)
    monkeypatch.setattr(Path, "open", growing_open)

    assert MemoryStore(paths).list_entries() == []
    assert requested == [9]


def test_memory_rejects_symlinked_user_container_without_external_access(
    tmp_path: Path,
) -> None:
    paths = KairoPaths.discover(tmp_path / "project", tmp_path / "home")
    paths.home.mkdir(parents=True)
    outside = tmp_path / "outside-memory"
    memory_dir = outside / "memory"
    memory_dir.mkdir(parents=True)
    external = memory_dir / "long_term_memory.json"
    external.write_text("[]", encoding="utf-8")
    paths.user_dir.symlink_to(outside, target_is_directory=True)
    store = MemoryStore(paths)

    assert store.list_entries() == []
    with pytest.raises(ValueError, match="symlink component"):
        store.save("must remain local")

    assert external.read_text(encoding="utf-8") == "[]"
    assert list(memory_dir.iterdir()) == [external]


def test_memory_storage_is_private_bounded_and_strict(tmp_path: Path) -> None:
    paths = KairoPaths.discover(tmp_path / "project", tmp_path / "home")
    memory = MemoryStore(paths)
    with pytest.raises(ValueError, match="cannot exceed"):
        memory.save("x" * (MAX_MEMORY_FACT_CHARS + 1))
    saved = memory.save("private fact")
    if os.name == "posix":
        assert paths.memory_file.parent.stat().st_mode & 0o777 == 0o700
        assert paths.memory_file.stat().st_mode & 0o777 == 0o600

    raw = [
        {
            "id": f"{number:012x}",
            "fact": f"fact {number}",
            "scope": "global",
            "project": None,
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        for number in range(MAX_MEMORY_ENTRIES + 2)
    ]
    raw.extend(
        [
            {"id": "../escape", "fact": "bad", "scope": "global"},
            {"id": "f" * 12, "fact": ["not text"], "scope": "global"},
        ]
    )
    paths.memory_file.write_text(json.dumps(raw), encoding="utf-8")
    loaded = memory.list_entries()
    assert len(loaded) == MAX_MEMORY_ENTRIES
    assert loaded[0].id == "000000000002"
    assert saved.id not in {entry.id for entry in loaded}


def test_memory_storage_refuses_symlink_write(tmp_path: Path) -> None:
    paths = KairoPaths.discover(tmp_path / "project", tmp_path / "home")
    paths.memory_file.parent.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("[]", encoding="utf-8")
    paths.memory_file.symlink_to(outside)
    memory = MemoryStore(paths)
    assert memory.list_entries() == []
    with pytest.raises(ValueError, match="symlink"):
        memory.save("must stay local")
    assert outside.read_text(encoding="utf-8") == "[]"


def test_memory_lock_refuses_symlink_and_prevents_multiprocess_lost_updates(
    tmp_path: Path,
) -> None:
    paths = KairoPaths.discover(tmp_path / "project", tmp_path / "home")
    paths.workspace.mkdir()
    paths.memory_file.parent.mkdir(parents=True)
    outside = tmp_path / "outside.lock"
    outside.write_text("outside", encoding="utf-8")
    lock = paths.memory_file.parent / ".long_term_memory.lock"
    lock.symlink_to(outside)
    with pytest.raises(ValueError, match="lock.*symlink"):
        MemoryStore(paths).save("blocked")
    assert outside.read_text(encoding="utf-8") == "outside"
    lock.unlink()

    context = multiprocessing.get_context("spawn")
    start = context.Event()
    processes = [
        context.Process(
            target=_concurrent_memory_writer,
            args=(str(paths.home), str(paths.workspace), f"writer-{number}", start),
        )
        for number in range(4)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(20)
        assert process.exitcode == 0
    entries = MemoryStore(paths).list_entries()
    assert len(entries) == 80
    assert len({entry.fact for entry in entries}) == 80
    if os.name == "posix":
        assert lock.stat().st_mode & 0o777 == 0o600


def test_browser_login_fact_requires_explicit_remember_intent() -> None:
    assert (
        browser_login_fact(
            "你可以直接复用我已经登录的 Chrome，记一下",
            ["请打开 https://www.yuque.com/example/doc"],
        )
        == "访问 yuque.com（语雀）时优先复用用户已登录的 Chrome 登录态。"
    )
    assert browser_login_fact("直接复用已登录的 Chrome", []) is None
    assert (
        browser_login_fact("以后需要登录态时直接用已登录 Chrome，记住", [])
        == "用户明确允许在需要登录态的网站访问中复用已登录 Chrome。"
    )


class MemoryClient(LlmClient):
    provider = "fake"
    model = "fake"

    def __init__(self) -> None:
        self.system_prompts: list[str] = []

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        self.system_prompts.append(str(messages[0].content))
        return LlmResponse(content="done")


async def test_agent_injects_only_query_relevant_long_term_memory(tmp_path: Path) -> None:
    paths = KairoPaths.discover(tmp_path / "project", tmp_path / "home")
    memory = MemoryStore(paths)
    memory.save("项目使用 asyncio 处理并发")
    client = MemoryClient()
    agent = Agent(client, ToolRegistry(paths.workspace), "base", memory_store=memory)
    await agent.run("asyncio 怎么使用")
    await agent.run("今天天气如何")
    assert "asyncio 处理并发" in client.system_prompts[0]
    assert "asyncio 处理并发" not in client.system_prompts[1]


async def test_agent_stores_explicit_browser_login_hint(tmp_path: Path) -> None:
    paths = KairoPaths.discover(tmp_path / "project", tmp_path / "home")
    memory = MemoryStore(paths)
    agent = Agent(MemoryClient(), ToolRegistry(paths.workspace), "base", memory_store=memory)
    agent.history.append(Message("user", "打开 https://www.yuque.com/example/doc"))
    await agent.run("可以复用我已登录的 Chrome，记一下")
    assert "yuque.com" in memory.search("登录 Chrome")[0].fact
    other = MemoryStore(KairoPaths.discover(tmp_path / "other", paths.home))
    assert "yuque.com" in other.search("登录 Chrome")[0].fact


async def test_make_agent_registers_save_memory_tool(tmp_path: Path) -> None:
    paths = KairoPaths.discover(tmp_path / "project", tmp_path / "home")
    config = AppConfig.load(paths)
    agent = make_agent(paths, config)
    result = await agent.tools.execute(
        "save_memory", {"fact": "项目使用 Ruff", "scope": "project"}
    )
    assert "Saved long-term memory" in result
    assert MemoryStore(paths).search("Ruff")
