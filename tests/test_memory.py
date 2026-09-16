import io
import json
import multiprocessing
import os
from pathlib import Path
from typing import Any

import pytest

import kairocli.cli.bootstrap as bootstrap_module
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
from kairocli.policy import ApprovalPolicy
from kairocli.tools import ToolRegistry


def _concurrent_memory_writer(home: str, workspace: str, prefix: str, start: Any) -> None:
    memory = MemoryStore(KairoPaths.discover(Path(workspace), Path(home)))
    start.wait(10)
    for number in range(20):
        memory.save(f"{prefix} fact {number}")


class SemanticEmbedding:
    signature = "semantic-test-v1"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.fail = False

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.fail:
            raise RuntimeError("embedding unavailable")
        return [
            [1.0, 0.0] if any(word in text for word in ("咖啡", "冰美式", "拿铁")) else [0.0, 1.0]
            for text in texts
        ]


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


def test_memory_versions_replace_same_key_and_project_overrides_global(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    memory = MemoryStore(KairoPaths.discover(project, home))
    old = memory.save("用户喜欢冰美式", "global", key="user.preference.drink")
    current = memory.save("用户现在喜欢拿铁", "global", key="user.preference.drink")
    local = memory.save("本项目使用茶", key="user.preference.drink")

    assert memory.list_entries() == [local]
    history = memory.list_entries(include_superseded=True)
    assert next(entry for entry in history if entry.id == old.id).superseded_by == current.id
    assert next(entry for entry in history if entry.id == current.id).status == "active"
    other = MemoryStore(KairoPaths.discover(tmp_path / "other", home))
    assert other.list_entries() == [current]


def test_memory_can_replace_unkeyed_legacy_fact(tmp_path: Path) -> None:
    memory = MemoryStore(KairoPaths.discover(tmp_path / "project", tmp_path / "home"))
    old = memory.save("用户喜欢冰美式", "global")
    new = memory.save(
        "用户现在喜欢拿铁",
        "global",
        key="user.preference.drink",
        replaces=[old.id],
    )
    assert memory.list_entries() == [new]
    assert memory.list_entries(include_superseded=True)[0].superseded_by == new.id


def test_project_memory_cannot_supersede_global_memory(tmp_path: Path) -> None:
    memory = MemoryStore(KairoPaths.discover(tmp_path / "project", tmp_path / "home"))
    global_entry = memory.save("用户喜欢冰美式", "global")
    with pytest.raises(ValueError, match="another scope or project"):
        memory.save(
            "本项目改喝茶",
            key="user.preference.drink",
            replaces=[global_entry.id],
        )
    assert memory.list_entries() == [global_entry]


async def test_hybrid_memory_search_uses_semantics_rrf_and_cache(tmp_path: Path) -> None:
    embedding = SemanticEmbedding()
    memory = MemoryStore(KairoPaths.discover(tmp_path / "project", tmp_path / "home"), embedding)
    coffee = memory.save("用户喜欢冰美式", "global", key="user.preference.drink")
    memory.save("项目使用 PostgreSQL", key="project.database")

    assert await memory.search_hybrid("平时点哪种咖啡") == [coffee]
    assert len(embedding.calls) == 1
    assert await memory.search_hybrid("平时点哪种咖啡") == [coffee]
    assert len(embedding.calls) == 2
    assert embedding.calls[1] == ["平时点哪种咖啡"]
    if os.name == "posix":
        assert memory.embedding_file.stat().st_mode & 0o777 == 0o600


async def test_save_immediately_indexes_memory_and_reuses_cached_vector(tmp_path: Path) -> None:
    embedding = SemanticEmbedding()
    memory = MemoryStore(KairoPaths.discover(tmp_path / "project", tmp_path / "home"), embedding)

    entry, indexed = await memory.save_with_embedding("用户喜欢冰美式", "global")
    duplicate, duplicate_indexed = await memory.save_with_embedding("用户喜欢冰美式", "global")

    assert indexed and duplicate_indexed
    assert duplicate.id == entry.id
    assert embedding.calls == [["用户喜欢冰美式"]]


async def test_save_keeps_memory_when_embedding_is_deferred(tmp_path: Path) -> None:
    embedding = SemanticEmbedding()
    embedding.fail = True
    memory = MemoryStore(KairoPaths.discover(tmp_path / "project", tmp_path / "home"), embedding)

    entry, indexed = await memory.save_with_embedding("用户喜欢冰美式", "global")

    assert not indexed
    assert memory.list_entries() == [entry]


async def test_hybrid_memory_search_falls_back_when_embedding_fails(tmp_path: Path) -> None:
    embedding = SemanticEmbedding()
    embedding.fail = True
    memory = MemoryStore(KairoPaths.discover(tmp_path / "project", tmp_path / "home"), embedding)
    expected = memory.save("用户喜欢冰美式", "global")
    assert await memory.search_hybrid("喜欢冰美式") == [expected]


async def test_hybrid_memory_search_refuses_symlinked_cache(tmp_path: Path) -> None:
    embedding = SemanticEmbedding()
    memory = MemoryStore(KairoPaths.discover(tmp_path / "project", tmp_path / "home"), embedding)
    expected = memory.save("用户喜欢冰美式", "global")
    outside = tmp_path / "outside.db"
    outside.write_text("untouched", encoding="utf-8")
    try:
        memory.embedding_file.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is not permitted")

    assert await memory.search_hybrid("喜欢冰美式") == [expected]
    assert outside.read_text(encoding="utf-8") == "untouched"


async def test_memory_commands_share_save_search_and_delete(tmp_path: Path) -> None:
    embedding = SemanticEmbedding()
    memory = MemoryStore(KairoPaths.discover(tmp_path / "project", tmp_path / "home"), embedding)
    saved = await handle_save_command("--global prefer deterministic tests", memory)
    project_entry = memory.save("project uses pytest")
    entry = memory.list_entries()[0]
    assert entry.id in saved and entry.scope == "global"
    listed = (await handle_memory_command("list", memory)).splitlines()
    assert entry.created_at[:19].replace("T", " ") in listed[0]
    assert listed[0].index(entry.fact) == listed[1].index(project_entry.fact)
    assert "prefer deterministic tests" in await handle_memory_command(
        "search deterministic", memory
    )
    assert await handle_memory_command(f"delete {entry.id}", memory) == "Deleted."
    assert await handle_memory_command(f"delete {project_entry.id}", memory) == "Deleted."
    assert await handle_memory_command("list", memory) == "No visible memories."
    with memory._embedding_connection() as connection:
        assert connection.execute("SELECT count(*) FROM memory_embeddings").fetchone()[0] == 0


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
        '"project":null,"created_at":"2026-01-01","unknown":' + "[" * 20 + "0" + "]" * 20 + "}]",
    ],
)
def test_memory_strict_json_failures_degrade_and_recover(tmp_path: Path, payload: str) -> None:
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


async def test_agent_retrieves_memory_for_provenance_followup(tmp_path: Path) -> None:
    paths = KairoPaths.discover(tmp_path / "project", tmp_path / "home")
    memory = MemoryStore(paths)
    memory.save("用户喜欢冰美式", "global", key="user.preference.drink")
    client = MemoryClient()
    agent = Agent(client, ToolRegistry(paths.workspace), "base", memory_store=memory)

    await agent.run("你知道我喜欢喝什么吗？")
    await agent.run("你怎么知道的？")

    assert "用户喜欢冰美式" in client.system_prompts[0]
    assert "用户喜欢冰美式" in client.system_prompts[1]


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
    schema = next(
        item for item in agent.tools.schemas() if item["function"]["name"] == "save_memory"
    )
    properties = schema["function"]["parameters"]["properties"]
    assert "Canonical current fact" in properties["fact"]["description"]
    assert "Stable semantic slot" in properties["key"]["description"]
    assert "stable key that names the semantic slot" in agent.base_system_prompt
    result = await agent.tools.execute("save_memory", {"fact": "项目使用 Ruff", "scope": "project"})
    assert "Saved long-term memory" in result
    assert MemoryStore(paths).search("Ruff")


async def test_save_memory_tool_immediately_indexes_with_configured_embedding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = KairoPaths.discover(tmp_path / "project", tmp_path / "home")
    embedding = SemanticEmbedding()
    monkeypatch.setenv("KAIROCLI_MEMORY_SEMANTIC_SEARCH", "true")
    monkeypatch.setattr(bootstrap_module, "embedding_client_from_environment", lambda: embedding)
    agent = make_agent(paths, AppConfig.load(paths))

    await agent.tools.execute("save_memory", {"fact": "用户喜欢冰美式", "scope": "global"})

    assert embedding.calls == [["用户喜欢冰美式"]]


async def test_agent_memory_tools_search_and_supersede_without_approval(tmp_path: Path) -> None:
    paths = KairoPaths.discover(tmp_path / "project", tmp_path / "home")
    config = AppConfig.load(paths)
    agent = make_agent(paths, config, approval_policy=ApprovalPolicy(enabled=True))
    first = await agent.tools.execute(
        "save_memory",
        {
            "fact": "用户喜欢冰美式",
            "scope": "global",
            "key": "user.preference.drink",
        },
    )
    assert "Saved long-term memory" in first
    found = json.loads(await agent.tools.execute("search_memory", {"query": "喜欢喝什么"}))
    old_id = found["results"][0]["id"]

    await agent.tools.execute(
        "save_memory",
        {
            "fact": "用户现在喜欢拿铁",
            "scope": "global",
            "key": "user.preference.drink",
            "replaces": [old_id],
        },
    )
    updated = json.loads(await agent.tools.execute("search_memory", {"query": "喜欢"}))
    assert [item["fact"] for item in updated["results"]] == ["用户现在喜欢拿铁"]

    direct = MemoryStore(paths)
    for number in range(12):
        direct.save(f"shared preference {number}", "global")
    default_results = json.loads(
        await agent.tools.execute("search_memory", {"query": "shared preference"})
    )
    assert len(default_results["results"]) == 10
