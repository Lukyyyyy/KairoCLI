import asyncio
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import kairocli.runtime_api as runtime_module
from kairocli.agent import Agent
from kairocli.brand import API_KEY_HEADER
from kairocli.llm import LlmClient
from kairocli.models import LlmResponse, Message
from kairocli.paths import KairoPaths
from kairocli.runtime_api import RuntimeThreadStore, create_app
from kairocli.tools import ToolRegistry


class RuntimeClient(LlmClient):
    provider = "test"
    model = "test"

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        return LlmResponse(content="runtime answer")


@pytest.mark.parametrize("api_key", [" ", "secret\n", "é", "x" * 1_025])
def test_runtime_rejects_unusable_api_keys_before_database_creation(
    tmp_path: Path, api_key: str
) -> None:
    database = tmp_path / "runtime.db"
    with pytest.raises(ValueError, match="visible ASCII"):
        create_app(lambda: None, api_key, database)

    assert not database.exists()


def test_runtime_auth_and_events(tmp_path: Path) -> None:
    database = tmp_path / "runtime.db"
    app = create_app(
        lambda: Agent(RuntimeClient(), ToolRegistry(tmp_path), "system"), "secret", database
    )
    with TestClient(app) as client:
        assert client.post("/v1/threads").status_code == 401
        headers = {API_KEY_HEADER: "secret"}
        thread = client.post("/v1/threads", headers=headers).json()["id"]
        response = client.post(
            f"/v1/threads/{thread}/turns", headers=headers, json={"input": "hello"}
        )
        assert response.status_code == 202
        assert thread.startswith("thread_")
        assert response.json()["id"].startswith("turn_")
        assert response.json()["status"] == "running"
        events = client.get(f"/v1/threads/{thread}/events", headers=headers)
        assert events.status_code == 200
        assert "id: " in events.text
        assert "thread.created" in events.text
        assert "turn.completed" in events.text
        event_ids = [
            int(line.removeprefix("id: "))
            for line in events.text.splitlines()
            if line.startswith("id: ")
        ]
        assert event_ids == sorted(event_ids)
        resumed = client.get(f"/v1/threads/{thread}/events?after={event_ids[-2]}", headers=headers)
        assert resumed.text.count("id: ") == 1
        assert f"id: {event_ids[-1]}" in resumed.text
        assert (
            client.get(
                f"/v1/threads/{thread}/events",
                headers={**headers, "Last-Event-ID": str(event_ids[-1])},
            ).text
            == ""
        )
        assert (
            client.get(
                f"/v1/threads/{thread}/events",
                headers={**headers, "Last-Event-ID": "bad"},
            ).status_code
            == 400
        )
    restarted = create_app(
        lambda: Agent(RuntimeClient(), ToolRegistry(tmp_path), "system"), "secret", database
    )
    with TestClient(restarted) as client:
        assert client.get(f"/v1/threads/{thread}/events", headers=headers).status_code == 200


def test_runtime_refreshes_inherited_owner_identity(tmp_path: Path) -> None:
    app = create_app(lambda: None, "secret", tmp_path / "fork-owner.db")
    state = app.state.runtime
    inherited_token = state.owner_token
    state.owner_pid = runtime_module.MAX_SQLITE_INTEGER
    state.active_agents["turn_inherited"] = object()
    state.active_turn_ids.add("turn_inherited")
    state.active_turn_tasks.add(object())  # type: ignore[arg-type]

    state.refresh_owner_identity()

    assert state.owner_pid == os.getpid()
    assert state.owner_token != inherited_token
    assert state.active_agents == {}
    assert state.active_turn_ids == set()
    assert state.active_turn_tasks == set()


async def test_runtime_shutdown_terminalizes_and_awaits_active_turns(
    tmp_path: Path,
) -> None:
    app = create_app(lambda: None, "secret", tmp_path / "shutdown.db")
    state = app.state.runtime
    thread_id = "thread_shutdown"
    state.store.create(thread_id)
    turn_id, _, _ = state.store.reserve_turn(
        thread_id,
        "slow",
        "shutdown-key",
        owner_token=state.owner_token,
        owner_pid=state.owner_pid,
    )
    canceled = asyncio.Event()

    class ActiveAgent:
        cancel_count = 0

        def cancel(self) -> None:
            self.cancel_count += 1

    active = ActiveAgent()

    async def running_turn() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            canceled.set()

    async with app.router.lifespan_context(app):
        task = asyncio.create_task(running_turn())
        state.active_agents[turn_id] = active  # type: ignore[assignment]
        state.active_turn_ids.add(turn_id)
        state.active_turn_tasks.add(task)
        await asyncio.sleep(0)

    assert active.cancel_count == 1
    assert canceled.is_set()
    assert task.cancelled()
    assert state.store.turn_status(thread_id, turn_id) == "canceled"
    assert state.active_agents == {}
    assert state.active_turn_ids == set()
    assert state.active_turn_tasks == set()


async def test_runtime_shutdown_terminalizes_reserved_turn_before_agent_starts(
    tmp_path: Path,
) -> None:
    app = create_app(lambda: None, "secret", tmp_path / "pending-shutdown.db")
    state = app.state.runtime
    thread_id = "thread_pending"
    state.store.create(thread_id)
    turn_id, _, _ = state.store.reserve_turn(
        thread_id,
        "accepted but not started",
        None,
        owner_token=state.owner_token,
        owner_pid=state.owner_pid,
    )

    async with app.router.lifespan_context(app):
        state.active_turn_ids.add(turn_id)

    assert state.store.turn_status(thread_id, turn_id) == "canceled"
    assert state.active_turn_ids == set()
    assert [event.type for event in state.store.events(thread_id)] == ["turn.canceled"]


async def test_runtime_shutdown_finishes_cleanup_before_propagating_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = create_app(lambda: None, "secret", tmp_path / "canceled-shutdown.db")
    state = app.state.runtime
    thread_id = "thread_cancelshutdown"
    state.store.create(thread_id)
    turn_id, _, _ = state.store.reserve_turn(
        thread_id,
        "stubborn",
        None,
        owner_token=state.owner_token,
        owner_pid=state.owner_pid,
    )
    cancellation_seen = asyncio.Event()
    release = asyncio.Event()

    async def stubborn_turn() -> None:
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release.wait()

    monkeypatch.setattr(runtime_module, "RUNTIME_SHUTDOWN_GRACE_SECONDS", 0.01)
    context = app.router.lifespan_context(app)
    await context.__aenter__()
    task = asyncio.create_task(stubborn_turn())
    state.active_turn_ids.add(turn_id)
    state.active_turn_tasks.add(task)
    closing = asyncio.create_task(context.__aexit__(None, None, None))
    await asyncio.wait_for(cancellation_seen.wait(), 1)
    closing.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(closing, 0.5)
    assert state.store.turn_status(thread_id, turn_id) == "canceled"
    assert state.active_turn_ids == set()
    assert state.active_turn_tasks == set()
    assert task in runtime_module._DETACHED_RUNTIME_TASKS

    release.set()
    for _ in range(50):
        if task not in runtime_module._DETACHED_RUNTIME_TASKS:
            break
        await asyncio.sleep(0.01)
    assert task not in runtime_module._DETACHED_RUNTIME_TASKS


def test_runtime_bearer_scheme_is_case_insensitive(tmp_path: Path) -> None:
    app = create_app(lambda: None, "secret", tmp_path / "bearer-case.db")
    with TestClient(app) as client:
        assert (
            client.post("/v1/threads", headers={"Authorization": "bearer secret"}).status_code
            == 200
        )
        assert (
            client.post("/v1/threads", headers={"Authorization": "BEARER   secret"}).status_code
            == 200
        )


def test_runtime_rejects_invalid_identifiers_before_store_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = create_app(lambda: None, "secret", tmp_path / "invalid-identifiers.db")
    headers = {API_KEY_HEADER: "secret"}

    def unexpected_exists(_thread_id: str) -> bool:
        raise AssertionError("invalid identifier reached the runtime store")

    monkeypatch.setattr(app.state.runtime.store, "exists", unexpected_exists)
    invalid_thread = "thread_" + "x" * 65
    with TestClient(app) as client:
        create = client.post(
            f"/v1/threads/{invalid_thread}/turns",
            headers=headers,
            json={"input": "hello"},
        )
        events = client.get(
            f"/v1/threads/{invalid_thread}/events",
            headers=headers,
        )

    assert create.status_code == 404
    assert create.json() == {"detail": "Thread not found"}
    assert events.status_code == 404
    assert events.json() == {"detail": "Thread not found"}


def test_runtime_rejects_invalid_turn_identifier_before_turn_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = create_app(lambda: None, "secret", tmp_path / "invalid-turn.db")
    state = app.state.runtime
    state.store.create("thread_valid")

    def unexpected_turn_status(_thread_id: str, _turn_id: str) -> str | None:
        raise AssertionError("invalid turn identifier reached the runtime store")

    monkeypatch.setattr(state.store, "turn_status", unexpected_turn_status)
    headers = {API_KEY_HEADER: "secret"}
    with TestClient(app) as client:
        response = client.post(
            "/v1/threads/thread_valid/turns/not-a-turn/cancel",
            headers=headers,
        )

    assert response.status_code == 404
    assert response.json() == {"detail": "Turn not found"}


@pytest.mark.parametrize(
    ("target", "headers"),
    [
        ("?after=9223372036854775808", {}),
        ("", {"Last-Event-ID": "9223372036854775808"}),
        ("", {"Last-Event-ID": "-1"}),
        ("", {"Last-Event-ID": "+1"}),
    ],
)
def test_runtime_rejects_event_cursors_outside_sqlite_integer_range(
    tmp_path: Path, target: str, headers: dict[str, str]
) -> None:
    app = create_app(lambda: None, "secret", tmp_path / "invalid-cursor.db")
    app.state.runtime.store.create("thread_cursor")
    request_headers = {API_KEY_HEADER: "secret", **headers}

    with TestClient(app) as client:
        response = client.get(f"/v1/threads/thread_cursor/events{target}", headers=request_headers)

    assert response.status_code in {400, 422}


def test_runtime_rejects_ambiguous_or_oversized_authorization(tmp_path: Path) -> None:
    app = create_app(lambda: None, "secret", tmp_path / "bearer-invalid.db")
    invalid = (
        "Basic secret",
        "Bearer\tsecret",
        "Bearer secret extra",
        "Bearer " + "x" * 1_025,
    )
    with TestClient(app) as client:
        for authorization in invalid:
            response = client.post("/v1/threads", headers={"Authorization": authorization})
            assert response.status_code == 401
            assert response.json() == {"detail": "Unauthorized"}


@pytest.mark.parametrize("invalid_input", [123, True, ["hello"], {"text": "hello"}])
def test_runtime_rejects_non_string_input(tmp_path: Path, invalid_input: Any) -> None:
    app = create_app(
        lambda: Agent(RuntimeClient(), ToolRegistry(tmp_path), "system"),
        "secret",
        tmp_path / "typed-input.db",
    )
    headers = {API_KEY_HEADER: "secret"}
    with TestClient(app) as client:
        thread_id = client.post("/v1/threads", headers=headers).json()["id"]
        response = client.post(
            f"/v1/threads/{thread_id}/turns",
            headers=headers,
            json={"input": invalid_input, "prompt": "must not be used"},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "input must be a string"


def test_runtime_turn_idempotency_prevents_duplicate_execution(tmp_path: Path) -> None:
    app = create_app(
        lambda: Agent(RuntimeClient(), ToolRegistry(tmp_path), "system"),
        "secret",
        tmp_path / "idempotency.db",
    )
    headers = {API_KEY_HEADER: "secret", "Idempotency-Key": "request-1"}
    with TestClient(app) as client:
        thread = client.post("/v1/threads", headers=headers).json()["id"]
        first = client.post(f"/v1/threads/{thread}/turns", headers=headers, json={"input": "hello"})
        second = client.post(
            f"/v1/threads/{thread}/turns", headers=headers, json={"input": "hello"}
        )
        assert first.status_code == second.status_code == 202
        assert first.json()["id"] == second.json()["id"]
        events = client.get(f"/v1/threads/{thread}/events", headers={API_KEY_HEADER: "secret"}).text
        assert events.count("event: turn.started") == 1
        assert events.count("event: turn.completed") == 1
        conflict = client.post(
            f"/v1/threads/{thread}/turns",
            headers=headers,
            json={"input": "different"},
        )
        assert conflict.status_code == 409


@pytest.mark.parametrize("key", ["contains space", "x" * 201])
def test_runtime_rejects_invalid_idempotency_keys(tmp_path: Path, key: str) -> None:
    app = create_app(
        lambda: Agent(RuntimeClient(), ToolRegistry(tmp_path), "system"),
        "secret",
        tmp_path / "idempotency-characters.db",
    )
    headers = {API_KEY_HEADER: "secret"}
    with TestClient(app) as client:
        thread = client.post("/v1/threads", headers=headers).json()["id"]
        response = client.post(
            f"/v1/threads/{thread}/turns",
            headers={**headers, "Idempotency-Key": key},
            json={"input": "hello"},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid Idempotency-Key"


def test_runtime_turn_cancel_is_idempotent_and_terminal_state_is_cas(
    tmp_path: Path,
) -> None:
    app = create_app(
        lambda: Agent(RuntimeClient(), ToolRegistry(tmp_path), "system"),
        "secret",
        tmp_path / "cancel.db",
    )
    headers = {API_KEY_HEADER: "secret"}
    with TestClient(app) as client:
        thread_id = client.post("/v1/threads", headers=headers).json()["id"]
        state = app.state.runtime
        turn_id, _, _ = state.store.reserve_turn(thread_id, "slow", "cancel-key")
        state.event(thread_id, "turn.started", {"turn_id": turn_id})
        active = Agent(RuntimeClient(), ToolRegistry(tmp_path), "system")
        state.active_agents[turn_id] = active

        first = client.post(f"/v1/threads/{thread_id}/turns/{turn_id}/cancel", headers=headers)
        second = client.post(f"/v1/threads/{thread_id}/turns/{turn_id}/cancel", headers=headers)

        assert first.json()["status"] == second.json()["status"] == "canceled"
        assert active.cancel_event.is_set()
        assert state.store.update_turn_status(turn_id, "completed", response="late") is False
        events = client.get(f"/v1/threads/{thread_id}/events", headers=headers).text
        assert events.count("event: turn.canceled") == 1
        assert "event: turn.completed" not in events


def test_runtime_store_rejects_parent_symlink_and_terminal_overwrite(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        RuntimeThreadStore(linked / "runtime.db")

    store = RuntimeThreadStore(tmp_path / "terminal.db")
    store.create("thread_terminal")
    turn_id, _, _ = store.reserve_turn("thread_terminal", "work", None)
    assert store.update_turn_status(turn_id, "failed", error="first") is True
    assert store.update_turn_status(turn_id, "completed", response="late") is False
    assert store.turn_status("thread_terminal", turn_id) == "failed"


def test_runtime_store_commits_turn_state_and_events_atomically(tmp_path: Path) -> None:
    database = tmp_path / "atomic-events.db"
    store = RuntimeThreadStore(database)
    store.create("thread_atomic")
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(
            "CREATE TRIGGER reject_started BEFORE INSERT ON events "
            "WHEN NEW.type='turn.started' BEGIN "
            "SELECT RAISE(ABORT, 'simulated event failure'); END"
        )

    with pytest.raises(sqlite3.DatabaseError, match="simulated event failure"):
        store.reserve_turn(
            "thread_atomic",
            "prompt",
            None,
            event_type="turn.started",
            event_data={"input": "prompt"},
        )

    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM turns").fetchone()[0] == 0
        connection.execute("DROP TRIGGER reject_started")
        connection.commit()

    turn_id, _, _ = store.reserve_turn(
        "thread_atomic",
        "prompt",
        None,
        event_type="turn.started",
        event_data={"input": "prompt"},
    )
    assert store.events("thread_atomic")[0].data == {
        "input": "prompt",
        "turn_id": turn_id,
    }
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(
            "CREATE TRIGGER reject_completed BEFORE INSERT ON events "
            "WHEN NEW.type='turn.completed' BEGIN "
            "SELECT RAISE(ABORT, 'simulated terminal event failure'); END"
        )

    with pytest.raises(sqlite3.DatabaseError, match="simulated terminal event failure"):
        store.update_turn_status(
            turn_id,
            "completed",
            response="answer",
            event_type="turn.completed",
            event_data={"turn_id": turn_id},
        )

    assert store.turn_status("thread_atomic", turn_id) == "running"
    assert [event.type for event in store.events("thread_atomic")] == ["turn.started"]


def test_runtime_store_commits_thread_and_created_event_atomically(tmp_path: Path) -> None:
    database = tmp_path / "atomic-thread.db"
    store = RuntimeThreadStore(database)
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(
            "CREATE TRIGGER reject_created BEFORE INSERT ON events "
            "WHEN NEW.type='thread.created' BEGIN "
            "SELECT RAISE(ABORT, 'simulated thread event failure'); END"
        )

    with pytest.raises(sqlite3.DatabaseError, match="simulated thread event failure"):
        store.create(
            "thread_atomic",
            event_type="thread.created",
            event_data={"thread_id": "thread_atomic"},
        )

    assert store.exists("thread_atomic") is False


def test_runtime_store_rejects_high_level_state_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    paths.home.mkdir(parents=True)
    outside = tmp_path / "outside-runtime-state"
    outside.mkdir()
    paths.user_dir.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        RuntimeThreadStore(paths.runtime_dir / "runtime.db")

    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm"])
def test_runtime_store_rechecks_database_symlinks_before_every_connection(
    tmp_path: Path, suffix: str
) -> None:
    database = tmp_path / "runtime.db"
    store = RuntimeThreadStore(database)
    store.create("thread_safe")
    attacked = Path(str(database) + suffix)
    target = tmp_path / f"real{suffix or '-db'}"
    if suffix:
        target.write_bytes(b"")
        attacked.unlink(missing_ok=True)
    else:
        database.replace(target)
    attacked.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        store.exists("thread_safe")


def test_runtime_tool_cleanup_failure_does_not_erase_completion(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path)

    async def fail_close() -> None:
        raise RuntimeError("cleanup failed")

    registry.close = fail_close  # type: ignore[method-assign]
    app = create_app(
        lambda: Agent(RuntimeClient(), registry, "system"),
        "secret",
        tmp_path / "cleanup.db",
    )
    headers = {API_KEY_HEADER: "secret"}
    with TestClient(app) as client:
        thread_id = client.post("/v1/threads", headers=headers).json()["id"]
        response = client.post(
            f"/v1/threads/{thread_id}/turns",
            headers=headers,
            json={"input": "hello"},
        )
        events = client.get(f"/v1/threads/{thread_id}/events", headers=headers).text
    assert response.status_code == 202
    assert "event: turn.completed" in events
    assert "event: turn.failed" not in events


def test_runtime_thread_restores_completed_conversation_context(tmp_path: Path) -> None:
    seen: list[list[Message]] = []

    class ContextClient(LlmClient):
        provider = "test"
        model = "context"

        async def complete(
            self,
            messages: list[Message],
            tools: list[dict[str, Any]] | None = None,
        ) -> LlmResponse:
            seen.append(list(messages))
            return LlmResponse(content=f"answer-{len(seen)}")

    app = create_app(
        lambda: Agent(ContextClient(), ToolRegistry(tmp_path), "system"),
        "secret",
        tmp_path / "context.db",
    )
    headers = {API_KEY_HEADER: "secret"}
    with TestClient(app) as client:
        thread = client.post("/v1/threads", headers=headers).json()["id"]
        assert (
            client.post(
                f"/v1/threads/{thread}/turns",
                headers=headers,
                json={"input": "first question"},
            ).status_code
            == 202
        )
        assert (
            client.post(
                f"/v1/threads/{thread}/turns",
                headers=headers,
                json={"input": "second question"},
            ).status_code
            == 202
        )

    assert len(seen) == 2
    second_contents = [str(message.content) for message in seen[1]]
    assert second_contents == [
        "system",
        "first question",
        "answer-1",
        "second question",
    ]


def test_runtime_store_serializes_turns_and_recovers_interrupted_work(
    tmp_path: Path,
) -> None:
    database = tmp_path / "private" / "runtime.db"
    store = RuntimeThreadStore(database)
    store.create("thread_test")
    turn_id, _, created = store.reserve_turn("thread_test", "first", "same")
    assert created is True
    assert store.reserve_turn("thread_test", "first", "same") == (
        turn_id,
        "running",
        False,
    )
    with pytest.raises(RuntimeError, match="running turn"):
        store.reserve_turn("thread_test", "second", None)

    live_peer = RuntimeThreadStore(database)
    with pytest.raises(RuntimeError, match="running turn"):
        live_peer.reserve_turn("thread_test", "second", None)

    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(
            "UPDATE turns SET owner_pid=? WHERE id=?",
            (runtime_module.MAX_SQLITE_INTEGER, turn_id),
        )
    recovered = RuntimeThreadStore(database)
    next_turn, status, next_created = recovered.reserve_turn("thread_test", "second", None)
    assert next_turn != turn_id
    assert status == "running"
    assert next_created is True
    if os.name != "nt":
        assert database.stat().st_mode & 0o777 == 0o600
        assert database.parent.stat().st_mode & 0o777 == 0o700


def test_runtime_store_recovers_owner_that_dies_after_peer_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "runtime-live-recovery.db"
    first = RuntimeThreadStore(database)
    first.create("thread_recovery")
    turn_id, _, _ = first.reserve_turn(
        "thread_recovery",
        "first",
        None,
        owner_token="first-runtime",
        owner_pid=os.getpid(),
    )
    waiting = RuntimeThreadStore(database)
    assert waiting.turn_status("thread_recovery", turn_id) == "running"

    monkeypatch.setattr(runtime_module, "_process_is_alive", lambda _pid: False)
    replacement_id, status, created = waiting.reserve_turn(
        "thread_recovery",
        "replacement",
        None,
        owner_token="replacement-runtime",
        owner_pid=os.getpid(),
    )

    assert replacement_id != turn_id
    assert status == "running" and created is True
    assert waiting.turn_status("thread_recovery", turn_id) == "failed"


def test_runtime_streaming_coalesces_small_deltas_without_duplication(tmp_path: Path) -> None:
    class StreamingClient(RuntimeClient):
        async def complete_streaming(
            self,
            messages: list[Message],
            tools: list[dict[str, Any]] | None = None,
            on_delta: Any = None,
        ) -> LlmResponse:
            on_delta("one")
            on_delta("two")
            return LlmResponse(content="onetwo", streamed=True)

    app = create_app(
        lambda: Agent(StreamingClient(), ToolRegistry(tmp_path), "system"),
        "secret",
        tmp_path / "stream.db",
    )
    headers = {API_KEY_HEADER: "secret"}
    with TestClient(app) as client:
        thread = client.post("/v1/threads", headers=headers).json()["id"]
        client.post(f"/v1/threads/{thread}/turns", headers=headers, json={"input": "hello"})
        events = client.get(f"/v1/threads/{thread}/events", headers=headers).text

    assert events.count("event: message.delta") == 1
    assert '"delta": "onetwo"' in events


def test_runtime_streaming_chunks_large_token_sequences(tmp_path: Path) -> None:
    class ManyDeltaClient(RuntimeClient):
        async def complete_streaming(
            self,
            messages: list[Message],
            tools: list[dict[str, Any]] | None = None,
            on_delta: Any = None,
        ) -> LlmResponse:
            for _ in range(5_000):
                on_delta("x")
            return LlmResponse(content="x" * 5_000, streamed=True)

    app = create_app(
        lambda: Agent(ManyDeltaClient(), ToolRegistry(tmp_path), "system"),
        "secret",
        tmp_path / "chunks.db",
    )
    headers = {API_KEY_HEADER: "secret"}
    with TestClient(app) as client:
        thread = client.post("/v1/threads", headers=headers).json()["id"]
        client.post(f"/v1/threads/{thread}/turns", headers=headers, json={"input": "hello"})
        body = client.get(f"/v1/threads/{thread}/events", headers=headers).text

    delta_lines = [
        line.removeprefix("data: ")
        for line in body.splitlines()
        if line.startswith("data: ") and '"delta"' in line
    ]
    assert len(delta_lines) == 2
    assert "".join(json.loads(line)["delta"] for line in delta_lines) == "x" * 5_000


def test_runtime_event_replay_is_byte_bounded_and_cursor_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_module, "MAX_RUNTIME_EVENT_RESPONSE_BYTES", 420)
    app = create_app(
        lambda: Agent(RuntimeClient(), ToolRegistry(tmp_path), "system"),
        "secret",
        tmp_path / "event-pages.db",
    )
    state = app.state.runtime
    thread_id = "thread_eventpages"
    state.store.create(thread_id)
    for index in range(7):
        state.store.append(thread_id, "test.event", {"index": index, "value": "x" * 80})

    headers = {API_KEY_HEADER: "secret"}
    cursor = 0
    seen: list[int] = []
    with TestClient(app) as client:
        for _ in range(10):
            response = client.get(
                f"/v1/threads/{thread_id}/events?after={cursor}&limit=10",
                headers=headers,
            )
            assert response.status_code == 200
            assert len(response.content) <= 420
            page_ids = [
                int(line.removeprefix("id: "))
                for line in response.text.splitlines()
                if line.startswith("id: ")
            ]
            seen.extend(page_ids)
            next_cursor = int(response.headers["X-Kairo-CLI-Next-Event-ID"])
            assert next_cursor >= cursor
            cursor = next_cursor
            if response.headers["X-Kairo-CLI-Has-More"] == "false":
                break

        oversized_limit = client.get(f"/v1/threads/{thread_id}/events?limit=1001", headers=headers)

    assert seen == list(range(1, 8))
    assert oversized_limit.status_code == 422


def test_runtime_event_has_more_requires_an_additional_event(tmp_path: Path) -> None:
    app = create_app(lambda: None, "secret", tmp_path / "event-has-more.db")
    state = app.state.runtime
    state.store.create("thread_exactpage")
    state.store.append("thread_exactpage", "test.event", {"index": 1})
    state.store.append("thread_exactpage", "test.event", {"index": 2})
    headers = {API_KEY_HEADER: "secret"}

    with TestClient(app) as client:
        exact = client.get("/v1/threads/thread_exactpage/events?limit=2", headers=headers)
        state.store.append("thread_exactpage", "test.event", {"index": 3})
        additional = client.get("/v1/threads/thread_exactpage/events?limit=2", headers=headers)

    assert exact.text.count("event: test.event") == 2
    assert exact.headers["X-Kairo-CLI-Has-More"] == "false"
    assert additional.text.count("event: test.event") == 2
    assert additional.headers["X-Kairo-CLI-Has-More"] == "true"


def test_runtime_event_replay_isolates_corrupt_rows_and_advances_cursor(
    tmp_path: Path,
) -> None:
    database = tmp_path / "corrupt-events.db"
    app = create_app(lambda: None, "secret", database)
    state = app.state.runtime
    thread_id = "thread_corrupt"
    state.store.create(thread_id)
    first_id = state.store.append(thread_id, "test.event", {"index": 1})
    duplicate_id = state.store.append(thread_id, "test.event", {"index": 2})
    unsafe_type_id = state.store.append(thread_id, "test.event", {"index": 3})
    last_id = state.store.append(thread_id, "test.event", {"index": 4})
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(
            "UPDATE events SET data=? WHERE sequence=?",
            ('{"index":2,"index":999}', duplicate_id),
        )
        connection.execute(
            "UPDATE events SET type=? WHERE sequence=?",
            ("test.event\nevent: injected", unsafe_type_id),
        )

    headers = {API_KEY_HEADER: "secret"}
    with TestClient(app) as client:
        response = client.get(
            f"/v1/threads/{thread_id}/events?after={first_id}&limit=3",
            headers=headers,
        )

    assert response.status_code == 200
    assert response.text.count("event: runtime.event.invalid") == 2
    assert "event: injected" not in response.text
    assert f"id: {last_id}" in response.text
    assert response.headers["X-Kairo-CLI-Next-Event-ID"] == str(last_id)
    assert response.headers["X-Kairo-CLI-Has-More"] == "false"


@pytest.mark.parametrize(
    ("payload", "error_name"),
    [
        ({"value": float("nan")}, "ValueError"),
        ({"value": {"nested": None}}, "ValueError"),
    ],
)
def test_runtime_event_write_replaces_non_finite_or_overdeep_json(
    tmp_path: Path, payload: dict[str, Any], error_name: str
) -> None:
    if isinstance(payload["value"], dict):
        current = payload["value"]
        for _ in range(runtime_module.MAX_RUNTIME_EVENT_JSON_DEPTH):
            child: dict[str, Any] = {}
            current["nested"] = child
            current = child
    store = RuntimeThreadStore(tmp_path / "strict-event-write.db")
    store.create("thread_strict")

    store.append("thread_strict", "test.event", payload)

    event = store.events("thread_strict")[-1]
    assert event.data == {"serialization_error": error_name}


def test_runtime_store_rejects_unsafe_event_type_atomically(tmp_path: Path) -> None:
    store = RuntimeThreadStore(tmp_path / "event-type.db")
    store.create("thread_type")

    with pytest.raises(ValueError, match="event type"):
        store.append("thread_type", "safe\nevent: injected", {"value": 1})

    assert store.events("thread_type") == []


def test_runtime_store_migrates_existing_turn_schema(tmp_path: Path) -> None:
    database = tmp_path / "legacy.db"
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("CREATE TABLE threads (id TEXT PRIMARY KEY, created_at TEXT NOT NULL)")
        connection.execute(
            "CREATE TABLE events (sequence INTEGER PRIMARY KEY AUTOINCREMENT, "
            "thread_id TEXT NOT NULL, type TEXT NOT NULL, data TEXT NOT NULL, "
            "timestamp TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE turns (id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, "
            "prompt TEXT NOT NULL, idempotency_key TEXT, status TEXT NOT NULL, "
            "created_at TEXT NOT NULL, updated_at TEXT NOT NULL, "
            "UNIQUE(thread_id, idempotency_key))"
        )

    store = RuntimeThreadStore(database)
    store.create("thread_legacy", workspace="/tmp/project")
    turn_id, _, _ = store.reserve_turn("thread_legacy", "hello", None)
    store.update_turn_status(turn_id, "completed", response="world")

    assert [message.content for message in store.completed_messages("thread_legacy")] == [
        "hello",
        "world",
    ]
    assert store.list_threads("")[0]["workspace"] == "/tmp/project"


def test_runtime_store_bounds_events_turns_and_threads(tmp_path: Path) -> None:
    store = RuntimeThreadStore(
        tmp_path / "bounded.db",
        max_events_per_thread=3,
        max_turns_per_thread=2,
        max_threads=2,
    )
    store.create("thread_one")
    for index in range(5):
        store.append("thread_one", "test", {"index": index})
    retained = store.events("thread_one")
    assert [event.data["index"] for event in retained] == [2, 3, 4]

    for index in range(4):
        turn_id, _, _ = store.reserve_turn("thread_one", f"q{index}", None)
        store.update_turn_status(turn_id, "completed", response=f"a{index}")
    assert [message.content for message in store.completed_messages("thread_one")] == [
        "q2",
        "a2",
        "q3",
        "a3",
    ]

    store.create("thread_two")
    store.create("thread_three")
    assert store.exists("thread_one") is False
    assert store.exists("thread_two") is True
    assert store.exists("thread_three") is True


def test_runtime_store_deletes_workspace_threads_but_rejects_running_turns(
    tmp_path: Path,
) -> None:
    store = RuntimeThreadStore(tmp_path / "workspace-delete.db")
    store.create("thread_first", owner_user_id="user", workspace="/project/a")
    store.create("thread_second", owner_user_id="user", workspace="/project/a")
    store.create("thread_other", owner_user_id="user", workspace="/project/b")
    completed_id, _, _ = store.reserve_turn("thread_first", "done", None)
    store.update_turn_status(completed_id, "completed", response="ok")
    running_id, _, _ = store.reserve_turn("thread_second", "running", None)

    with pytest.raises(RuntimeError, match="running turn"):
        store.delete_workspace_threads("user", ("/project/a",))

    assert store.exists("thread_first") is True
    store.update_turn_status(running_id, "canceled")
    assert store.delete_workspace_threads("user", ("/project/a",)) == 2
    assert store.exists("thread_first") is False
    assert store.exists("thread_second") is False
    assert store.exists("thread_other") is True


def test_runtime_event_size_and_errors_are_bounded_and_redacted(tmp_path: Path) -> None:
    database = tmp_path / "private.db"
    store = RuntimeThreadStore(database)
    store.create("thread_private")
    store.append("thread_private", "large", {"value": "\\" * 300_000})
    event = store.events("thread_private")[-1]
    assert event.data["partial"] is True
    with closing(sqlite3.connect(database)) as connection, connection:
        event_json = connection.execute(
            "SELECT data FROM events WHERE thread_id=? ORDER BY sequence DESC LIMIT 1",
            ("thread_private",),
        ).fetchone()[0]
    assert len(str(event_json).encode("utf-8")) <= 256 * 1024

    turn_id, _, _ = store.reserve_turn("thread_private", "prompt", None)
    store.update_turn_status(
        turn_id,
        "failed",
        error=(
            "Authorization: Bearer top-secret TOKEN=also-secret "
            "https://user:url-secret@example.test/?access_token=query-secret "
            "sk-proj-abcdefghijklmnopqrstuvwx"
        ),
    )
    with closing(sqlite3.connect(database)) as connection, connection:
        error = connection.execute("SELECT error FROM turns WHERE id=?", (turn_id,)).fetchone()[0]
    assert "top-secret" not in error
    assert "also-secret" not in error
    assert "url-secret" not in error
    assert "query-secret" not in error
    assert "sk-proj-abcdefghijklmnopqrstuvwx" not in error
    assert "***" in error


def test_runtime_response_truncation_includes_marker_within_byte_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_module, "MAX_RUNTIME_CONTEXT_BYTES", 64)
    database = tmp_path / "response.db"
    store = RuntimeThreadStore(database)
    store.create("thread_response")
    turn_id, _, _ = store.reserve_turn("thread_response", "prompt", None)

    assert store.update_turn_status(turn_id, "completed", response="你" * 100) is True
    with closing(sqlite3.connect(database)) as connection, connection:
        response = str(
            connection.execute("SELECT response FROM turns WHERE id=?", (turn_id,)).fetchone()[0]
        )

    assert response.endswith("[response truncated]")
    assert len(response.encode("utf-8")) <= 64


def test_runtime_terminal_state_repairs_surrogate_and_unprintable_error(
    tmp_path: Path,
) -> None:
    store = RuntimeThreadStore(tmp_path / "safe-terminal.db")
    store.create("thread_safe")
    completed_id, _, _ = store.reserve_turn("thread_safe", "prompt", None)
    assert store.update_turn_status(completed_id, "completed", response="answer\ud800")
    assert [message.content for message in store.completed_messages("thread_safe")] == [
        "prompt",
        "answer?",
    ]

    class UnprintableError(RuntimeError):
        def __str__(self) -> str:
            raise KeyboardInterrupt

    failed_id, _, _ = store.reserve_turn("thread_safe", "fail", None)
    assert store.update_turn_status(
        failed_id,
        "failed",
        error=UnprintableError(),  # type: ignore[arg-type]
    )
    with closing(sqlite3.connect(store.database)) as connection:
        stored_error = connection.execute(
            "SELECT error FROM turns WHERE id=?", (failed_id,)
        ).fetchone()[0]
    assert stored_error == "UnprintableError message unavailable"
