import asyncio
import json
import os
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import kairocli.tools.shell as shell_module
from kairocli.cancellation import AgentCanceled
from kairocli.cli import _handle_shell
from kairocli.commands import CommandType, parse_command
from kairocli.policy import ApprovalPolicy
from kairocli.shell import MAX_SHELL_SESSIONS, ShellSessionManager
from kairocli.tools import ToolRegistry

pytestmark = pytest.mark.skipif(os.name not in {"posix", "nt"}, reason="no shell")


class RecordingConsole:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def print(self, message: str) -> None:
        self.messages.append(message)


async def test_shell_close_attempts_every_process_after_one_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = ShellSessionManager(tmp_path)
    first = object()
    second = object()
    manager._sessions = {  # type: ignore[dict-item]
        "shell_first": SimpleNamespace(process=first),
        "shell_second": SimpleNamespace(process=second),
    }
    terminated: list[object] = []

    async def fake_terminate(process: object) -> None:
        terminated.append(process)
        if process is first:
            raise OSError("terminate failed")

    monkeypatch.setattr(shell_module, "_terminate_process_tree", fake_terminate)

    await manager.close()
    await manager.close()

    assert set(terminated) == {first, second}
    assert manager._sessions == {}


async def test_shell_manager_close_is_terminal(tmp_path: Path) -> None:
    manager = ShellSessionManager(tmp_path)
    await manager.close()

    assert await manager.list() == []
    with pytest.raises(RuntimeError, match="closed"):
        await manager.start()


async def test_shell_close_fences_process_creation_in_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = ShellSessionManager(tmp_path)
    creation_started = asyncio.Event()
    release_creation = asyncio.Event()
    process = object()
    terminated: list[object] = []

    async def delayed_create(*args: object, **kwargs: object) -> object:
        creation_started.set()
        await release_creation.wait()
        return process

    async def terminate(value: object) -> None:
        terminated.append(value)

    monkeypatch.setattr(shell_module.asyncio, "create_subprocess_exec", delayed_create)
    monkeypatch.setattr(shell_module, "_terminate_process_tree", terminate)
    starting = asyncio.create_task(manager.start())
    await creation_started.wait()
    closing = asyncio.create_task(manager.close())
    await asyncio.sleep(0)
    release_creation.set()

    with pytest.raises(RuntimeError, match="closed"):
        await starting
    await closing
    assert terminated == [process]
    assert manager._sessions == {}  # noqa: SLF001


async def test_persistent_shell_preserves_cwd_environment_and_exit_status(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    (workspace / "nested").mkdir()
    manager = ShellSessionManager(workspace)
    started = await manager.start()
    identifier = started["session_id"]

    changed = await manager.execute(
        identifier,
        "cd nested && DEMO_STATE=kept && export DEMO_STATE",
        5,
        None,
    )
    observed = await manager.execute(
        identifier,
        'printf "%s\\n" "$DEMO_STATE"; pwd; false',
        5,
        None,
    )

    assert changed["exit_code"] == 0
    assert "kept" in observed["stdout"]
    assert str(workspace / "nested") in observed["stdout"]
    assert observed["exit_code"] == 1
    assert await manager.stop(identifier) is True
    assert await manager.list() == []


async def test_shell_timeout_and_cancellation_close_complete_session(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    manager = ShellSessionManager(workspace)
    first = (await manager.start())["session_id"]

    timed_out = await manager.execute(first, "printf before-timeout; sleep 5", 0.1, None)

    assert timed_out["timed_out"] is True
    assert timed_out["session_closed"] is True
    assert "before-timeout" in timed_out["stdout"]
    assert timed_out["stdout_bytes"] >= len("before-timeout")
    assert await manager.list() == []

    second = (await manager.start())["session_id"]
    canceled = asyncio.Event()
    task = asyncio.create_task(manager.execute(second, "sleep 5", 5, canceled))
    await asyncio.sleep(0.05)
    canceled.set()
    with pytest.raises(AgentCanceled):
        await task
    assert await manager.list() == []


async def test_shell_sessions_are_bounded_and_output_is_truncated(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    manager = ShellSessionManager(workspace)
    identifiers = [(await manager.start())["session_id"] for _ in range(MAX_SHELL_SESSIONS)]
    with pytest.raises(ValueError, match="At most"):
        await manager.start()

    result = await manager.execute(
        identifiers[0],
        f"{shlex.quote(sys.executable)} -c 'print(\"x\" * 70000)'",
        5,
        None,
    )

    assert result["stdout_truncated"] is True
    assert result["stdout_bytes"] > 50_000
    await manager.close()


@pytest.mark.skipif(os.name != "posix", reason="POSIX shell tracing contract")
async def test_shell_trace_modes_cannot_spoof_internal_completion_marker(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    manager = ShellSessionManager(workspace)
    identifier = (await manager.start())["session_id"]

    assert (await manager.execute(identifier, "set -x", 5, None))["exit_code"] == 0
    traced = await manager.execute(identifier, "printf traced; false", 5, None)
    assert traced["exit_code"] == 1
    assert "traced" in traced["stdout"]
    assert "__KAIROCLI_DONE_" not in traced["stdout"]

    assert (await manager.execute(identifier, "set +x; set -v", 5, None))["exit_code"] == 0
    verbose = await manager.execute(identifier, "printf verbose; false", 5, None)
    assert verbose["exit_code"] == 1
    assert "verbose" in verbose["stdout"]
    assert "__KAIROCLI_DONE_" not in verbose["stdout"]

    final = await manager.execute(identifier, "set +v; printf final", 5, None)
    assert final["exit_code"] == 0
    assert "final" in final["stdout"]
    await manager.close()


async def test_shell_tools_apply_workspace_command_and_approval_policy(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    registry = ToolRegistry(workspace, approval_policy=ApprovalPolicy(False))
    schemas = {item["function"]["name"] for item in registry.schemas()}
    assert {"shell_start", "shell_exec", "shell_list", "shell_stop"} <= schemas
    assert "shell_exec" in ApprovalPolicy.DANGEROUS_TOOLS

    escaped = json.loads(await registry.execute("shell_start", {"cwd": "../outside"}))
    dangerous = json.loads(
        await registry.execute(
            "shell_exec",
            {"session_id": "shell_123456789abc", "command": "rm -rf /"},
        )
    )

    assert escaped["policy_denied"] is True
    assert dangerous["policy_denied"] is True
    await registry.close()


async def test_shell_slash_command_starts_lists_and_stops(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    registry = ToolRegistry(workspace, approval_policy=ApprovalPolicy(False))
    console = RecordingConsole()
    assert parse_command("/shell list").type == CommandType.SHELL

    await _handle_shell("start", registry, console)
    started = json.loads(console.messages[-1])
    identifier = started["session_id"]
    await _handle_shell(f"exec {identifier} printf hello", registry, console)
    executed = json.loads(console.messages[-1])
    assert "hello" in executed["stdout"]
    await _handle_shell(f"stop {identifier}", registry, console)
    assert json.loads(console.messages[-1])["stopped"] is True
    await registry.close()
