import asyncio
import io
import json
import os
import shlex
import sys
import time
from pathlib import Path

import pytest

import kairocli.cancellation as cancellation_module
import kairocli.tools.filesystem as filesystem_module
import kairocli.tools.process as process_module
import kairocli.tools.registry as tools_module
from kairocli.agent import AgentCanceled
from kairocli.models import ToolOutput
from kairocli.paths import KairoPaths
from kairocli.policy import ApprovalPolicy, ApprovalResult, AuditLog
from kairocli.tools import ToolDefinition, ToolRegistry
from kairocli.tools.tool_result import is_failed_tool_text


async def test_registry_close_attempts_all_owned_services(tmp_path: Path) -> None:
    closed: list[str] = []

    class Service:
        def __init__(self, name: str, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        async def close(self) -> None:
            closed.append(self.name)
            if self.fail:
                raise RuntimeError("close failed")

    registry = ToolRegistry(tmp_path)
    registry.shell_sessions = Service("shell", True)  # type: ignore[assignment]
    registry.lsp = Service("lsp")  # type: ignore[assignment]
    registry.snapshot_service = Service("snapshot")  # type: ignore[assignment]

    await registry.close()

    assert set(closed) == {"shell", "lsp", "snapshot"}


async def test_registry_close_is_terminal_and_rejects_late_tools(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path)
    await registry.close()

    output = json.loads(await registry.execute("read_file", {"path": "late.txt"}))
    assert output == {"error": "Tool registry is closed", "registry_closed": True}
    assert registry.schemas() == []
    with pytest.raises(RuntimeError, match="closed"):
        registry.register(
            ToolDefinition(
                "late_tool",
                "must not register",
                {"type": "object"},
                lambda _args: asyncio.sleep(0),
            )
        )


async def test_registry_close_finishes_services_before_propagating_cancel(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()
    closed: list[str] = []

    class Service:
        def __init__(self, name: str) -> None:
            self.name = name

        async def close(self) -> None:
            started.set()
            await release.wait()
            closed.append(self.name)

    registry.shell_sessions = Service("shell")  # type: ignore[assignment]
    registry.lsp = Service("lsp")  # type: ignore[assignment]
    registry.snapshot_service = Service("snapshot")  # type: ignore[assignment]
    closing = asyncio.create_task(registry.close())
    await started.wait()
    closing.cancel()
    await asyncio.sleep(0)

    assert not closing.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert set(closed) == {"shell", "lsp", "snapshot"}


async def test_registry_close_cancels_active_tool_before_owned_services(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(tmp_path)
    handler_started = asyncio.Event()
    handler_stopped = asyncio.Event()
    services_closed: list[str] = []

    async def blocking_handler(_arguments: dict[str, object]) -> str:
        handler_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            handler_stopped.set()
        return "unreachable"

    class Service:
        def __init__(self, name: str) -> None:
            self.name = name

        async def close(self) -> None:
            assert handler_stopped.is_set()
            services_closed.append(self.name)

    registry.register(
        ToolDefinition(
            "blocking_tool",
            "blocks until registry shutdown",
            {"type": "object", "additionalProperties": False},
            blocking_handler,
        )
    )
    registry.shell_sessions = Service("shell")  # type: ignore[assignment]
    registry.lsp = Service("lsp")  # type: ignore[assignment]
    registry.snapshot_service = Service("snapshot")  # type: ignore[assignment]
    execution = asyncio.create_task(registry.execute("blocking_tool", {}))
    await handler_started.wait()

    await registry.close()

    with pytest.raises(asyncio.CancelledError):
        await execution
    assert set(services_closed) == {"shell", "lsp", "snapshot"}


@pytest.mark.parametrize(
    "value",
    [
        '{"error":"failed","error":""}',
        '{"error":""}',
        '{"result":"ok","score":NaN}',
        '{"result":' + "[" * 33 + "0" + "]" * 33 + "}",
    ],
)
def test_ambiguous_json_tool_output_is_conservatively_failed(value: str) -> None:
    assert is_failed_tool_text(value) is True


async def test_file_tools(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path)
    write = json.loads(
        await registry.execute("write_file", {"path": "src/a.py", "content": "x = 1\n"})
    )
    assert write["bytes"] == 6
    assert await registry.execute("read_file", {"path": "src/a.py"}) == "1: x = 1"


async def test_read_file_is_bounded_and_supports_continuation(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_text(
        "\n".join(f"line-{index}" for index in range(1, 8)), encoding="utf-8"
    )
    registry = ToolRegistry(tmp_path)
    first = await registry.execute("read_file", {"path": "large.txt", "offset": 2, "limit": 2})
    assert first.startswith("2: line-2\n3: line-3")
    assert "partial: true; next_offset=4" in first
    second = await registry.execute("read_file", {"path": "large.txt", "offset": 4, "limit": 10})
    assert second.startswith("4: line-4")
    assert "partial: true" not in second


async def test_read_file_rejects_binary_and_reports_invalid_utf8(tmp_path: Path) -> None:
    (tmp_path / "binary.dat").write_bytes(b"abc\x00def")
    (tmp_path / "invalid.txt").write_bytes(b"ok\xff\n")
    registry = ToolRegistry(tmp_path)
    binary = json.loads(await registry.execute("read_file", {"path": "binary.dat"}))
    assert binary["type"] == "ValueError"
    invalid = await registry.execute("read_file", {"path": "invalid.txt"})
    assert "encoding_warning" in invalid


def test_read_file_bounds_single_lines_and_preserves_next_line_offsets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "long-line.txt"
    source.write_text("placeholder", encoding="utf-8")
    content = "x" * 100 + "\nneedle"
    requested: list[int] = []

    class TrackingText(io.StringIO):
        def readline(self, size: int = -1) -> str:
            requested.append(size)
            return super().readline(size)

    def tracking_open(
        _path: Path, mode: str = "r", *args: object, **kwargs: object
    ) -> io.BytesIO | TrackingText:
        if mode == "rb":
            return io.BytesIO(b"text")
        assert mode == "r"
        return TrackingText(content)

    monkeypatch.setattr(Path, "open", tracking_open)

    first = tools_module._read_file_range(source, 1, 10, 8)
    assert first.startswith("1: xxxxx")
    assert "partial: true; next_offset=2" in first
    assert requested == [9]

    requested.clear()
    second = tools_module._read_file_range(source, 2, 10, 20)
    assert second == "2: needle"
    assert requested
    assert set(requested) == {21}


async def test_write_file_returns_syntax_diagnostics(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path)
    result = json.loads(
        await registry.execute("write_file", {"path": "broken.py", "content": "def x(:\n"})
    )
    assert result["diagnostics"][0]["severity"] == "error"


async def test_write_file_is_atomic_preserves_mode_and_rejects_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("original", encoding="utf-8")
    target.chmod(0o640)
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    registry = ToolRegistry(tmp_path)

    rejected = json.loads(
        await registry.execute("write_file", {"path": "link.txt", "content": "changed"})
    )
    assert rejected["policy_denied"] is True
    assert target.read_text(encoding="utf-8") == "original"

    real_replace = os.replace

    def fail_replace(*args: object, **kwargs: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(filesystem_module.os, "replace", fail_replace)
    failed = json.loads(
        await registry.execute("write_file", {"path": "target.txt", "content": "new"})
    )
    assert failed["type"] == "OSError"
    assert target.read_text(encoding="utf-8") == "original"
    names = await asyncio.to_thread(os.listdir, tmp_path)
    assert not any(name.startswith(".target.txt.") for name in names)

    monkeypatch.setattr(filesystem_module.os, "replace", real_replace)
    await registry.execute("write_file", {"path": "target.txt", "content": "new"})
    assert target.read_text(encoding="utf-8") == "new"
    assert target.stat().st_mode & 0o777 == 0o640


async def test_post_edit_diagnostic_failure_preserves_success_and_is_redacted(
    tmp_path: Path,
) -> None:
    class FailingLsp:
        async def diagnose_file_async(self, path: Path) -> list[object]:
            raise RuntimeError("LSP unavailable token=diagnostic-secret")

    registry = ToolRegistry(tmp_path)
    registry.lsp = FailingLsp()  # type: ignore[assignment]

    written = json.loads(
        await registry.execute("write_file", {"path": "value.txt", "content": "old\n"})
    )
    assert written["path"] == "value.txt"
    assert written["diagnostics"] == []
    assert "LSP unavailable" in written["diagnostics_warning"]
    assert "diagnostic-secret" not in written["diagnostics_warning"]
    assert (tmp_path / "value.txt").read_text(encoding="utf-8") == "old\n"

    patch = """diff --git a/value.txt b/value.txt
--- a/value.txt
+++ b/value.txt
@@ -1 +1 @@
-old
+new
"""
    patched = json.loads(await registry.execute("apply_patch", {"patch": patch}))
    assert patched["changed"] == ["value.txt"]
    assert patched["diagnostics"] == []
    assert patched["diagnostics_warnings"][0]["path"] == "value.txt"
    assert "diagnostic-secret" not in patched["diagnostics_warnings"][0]["error"]
    assert (tmp_path / "value.txt").read_text(encoding="utf-8") == "new\n"


async def test_workspace_mutations_are_serialized_in_parallel_batches(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(tmp_path)
    original = registry._tools["write_file"]
    active = 0
    peak = 0
    order: list[str] = []

    async def mutation(arguments: dict[str, object]) -> dict[str, object]:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.03)
            order.append(str(arguments["path"]))
            return {"path": arguments["path"]}
        finally:
            active -= 1

    registry.register(ToolDefinition("write_file", "fixture", original.parameters, mutation))

    outputs = await registry.execute_many(
        [
            ("write_file", {"path": "first.txt", "content": "first"}),
            ("write_file", {"path": "second.txt", "content": "second"}),
        ],
        max_concurrency=2,
    )

    assert peak == 1
    assert order == ["first.txt", "second.txt"]
    assert [json.loads(output)["path"] for output in outputs] == order


async def test_create_project_rejects_symlink_parent(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "linked").symlink_to(outside, target_is_directory=True)
    registry = ToolRegistry(tmp_path)

    result = json.loads(
        await registry.execute("create_project", {"path": "linked/project", "kind": "python"})
    )

    assert result["policy_denied"] is True
    assert not (outside / "project").exists()


@pytest.mark.parametrize(
    ("kind", "expected_files", "expected_directories"),
    [
        (
            "python",
            {"demo/__init__.py", "main.py", "pyproject.toml", "requirements.txt"},
            {"demo", "src", "tests"},
        ),
        ("node", {"package.json"}, {"src"}),
        (
            "java",
            {"pom.xml"},
            {"src", "src/main", "src/main/java", "src/main/resources", "src/test", "src/test/java"},
        ),
    ],
)
async def test_create_project_generates_reference_templates_atomically(
    tmp_path: Path,
    kind: str,
    expected_files: set[str],
    expected_directories: set[str],
) -> None:
    registry = ToolRegistry(tmp_path)

    result = json.loads(
        await registry.execute("create_project", {"path": f"{kind}-root/demo", "kind": kind})
    )
    root = tmp_path / f"{kind}-root" / "demo"

    assert set(result["files"]) == expected_files
    assert set(result["directories"]) == expected_directories
    assert all((root / relative).is_file() for relative in expected_files)
    assert all((root / relative).is_dir() for relative in expected_directories)
    if kind == "node":
        assert json.loads((root / "package.json").read_text(encoding="utf-8")) == {
            "name": "demo",
            "version": "1.0.0",
            "private": True,
        }
    elif kind == "java":
        assert "<artifactId>demo</artifactId>" in (root / "pom.xml").read_text(encoding="utf-8")
    else:
        assert "[project]" in (root / "pyproject.toml").read_text(encoding="utf-8")


async def test_create_project_publish_failure_leaves_no_partial_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ToolRegistry(tmp_path)
    root = tmp_path / "demo"
    root.mkdir()
    real_replace = os.replace

    def fail_project_publish(source: object, destination: object, **kwargs: object) -> None:
        if "kairocli-project" in str(source):
            raise OSError("simulated project publish failure")
        real_replace(source, destination, **kwargs)

    monkeypatch.setattr(filesystem_module.os, "replace", fail_project_publish)

    result = json.loads(
        await registry.execute("create_project", {"path": "demo", "kind": "python"})
    )

    assert result["type"] == "OSError"
    assert root.is_dir() and not any(root.iterdir())
    names = await asyncio.to_thread(os.listdir, tmp_path)
    assert not any("kairocli-project" in name for name in names)


async def test_create_project_refuses_nonempty_target_without_modifying_it(
    tmp_path: Path,
) -> None:
    root = tmp_path / "demo"
    root.mkdir()
    existing = root / "keep.txt"
    existing.write_text("keep", encoding="utf-8")
    approvals = 0

    async def approve(name: str, arguments: dict[str, object]) -> bool:
        nonlocal approvals
        approvals += 1
        return True

    registry = ToolRegistry(
        tmp_path,
        approval_policy=ApprovalPolicy(True),
        approver=approve,
    )

    result = json.loads(
        await registry.execute("create_project", {"path": "demo", "kind": "python"})
    )

    assert result["policy_denied"] is True
    assert approvals == 0
    assert existing.read_text(encoding="utf-8") == "keep"


async def test_apply_patch_updates_file_and_returns_diagnostics(tmp_path: Path) -> None:
    (tmp_path / "value.py").write_text("value = 1\n", encoding="utf-8")
    registry = ToolRegistry(tmp_path)
    patch = """diff --git a/value.py b/value.py
--- a/value.py
+++ b/value.py
@@ -1 +1 @@
-value = 1
+value = 2
"""

    result = json.loads(await registry.execute("apply_patch", {"patch": patch}))

    assert (tmp_path / "value.py").read_text(encoding="utf-8") == "value = 2\n"
    assert result["changed"] == ["value.py"]
    assert result["deleted"] == []
    assert result["diagnostics"] == []


async def test_apply_patch_check_prevents_partial_changes(tmp_path: Path) -> None:
    (tmp_path / "first.txt").write_text("one\n", encoding="utf-8")
    (tmp_path / "second.txt").write_text("two\n", encoding="utf-8")
    registry = ToolRegistry(tmp_path)
    patch = """diff --git a/first.txt b/first.txt
--- a/first.txt
+++ b/first.txt
@@ -1 +1 @@
-one
+changed
diff --git a/second.txt b/second.txt
--- a/second.txt
+++ b/second.txt
@@ -1 +1 @@
-not-the-current-content
+changed
"""

    result = json.loads(await registry.execute("apply_patch", {"patch": patch}))

    assert "Patch check failed" in result["error"]
    assert (tmp_path / "first.txt").read_text(encoding="utf-8") == "one\n"
    assert (tmp_path / "second.txt").read_text(encoding="utf-8") == "two\n"


async def test_apply_patch_creates_and_deletes_files(tmp_path: Path) -> None:
    (tmp_path / "old.txt").write_text("old\n", encoding="utf-8")
    registry = ToolRegistry(tmp_path)
    patch = """diff --git a/new.txt b/new.txt
new file mode 100644
--- /dev/null
+++ b/new.txt
@@ -0,0 +1 @@
+new
diff --git a/old.txt b/old.txt
deleted file mode 100644
--- a/old.txt
+++ /dev/null
@@ -1 +0,0 @@
-old
"""

    result = json.loads(await registry.execute("apply_patch", {"patch": patch}))

    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "new\n"
    assert not (tmp_path / "old.txt").exists()
    assert result["changed"] == ["new.txt"]
    assert result["deleted"] == ["old.txt"]


async def test_apply_patch_rejects_escape_and_symlink_modes(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path)
    escape = """diff --git a/../outside.txt b/../outside.txt
--- a/../outside.txt
+++ b/../outside.txt
@@ -0,0 +1 @@
+bad
"""
    symlink = """diff --git a/link b/link
new file mode 120000
--- /dev/null
+++ b/link
@@ -0,0 +1 @@
+../outside
"""

    escaped = json.loads(await registry.execute("apply_patch", {"patch": escape}))
    linked = json.loads(await registry.execute("apply_patch", {"patch": symlink}))

    assert escaped["policy_denied"] is True
    assert linked["policy_denied"] is True
    assert not (tmp_path.parent / "outside.txt").exists()

    real = tmp_path / "real"
    real.mkdir()
    (real / "value.txt").write_text("old\n", encoding="utf-8")
    (tmp_path / "linked").symlink_to(real, target_is_directory=True)
    through_link = """diff --git a/linked/value.txt b/linked/value.txt
--- a/linked/value.txt
+++ b/linked/value.txt
@@ -1 +1 @@
-old
+changed
"""
    rejected = json.loads(await registry.execute("apply_patch", {"patch": through_link}))
    assert rejected["policy_denied"] is True
    assert (real / "value.txt").read_text(encoding="utf-8") == "old\n"


async def test_apply_patch_rejects_mismatched_path_markers(tmp_path: Path) -> None:
    (tmp_path / "safe.txt").write_text("safe\n", encoding="utf-8")
    (tmp_path / "other.txt").write_text("other\n", encoding="utf-8")
    registry = ToolRegistry(tmp_path)
    patch = """diff --git a/safe.txt b/safe.txt
--- a/other.txt
+++ b/other.txt
@@ -1 +1 @@
-other
+changed
"""

    result = json.loads(await registry.execute("apply_patch", {"patch": patch}))

    assert result["policy_denied"] is True
    assert "does not match diff --git header" in result["error"]
    assert (tmp_path / "safe.txt").read_text(encoding="utf-8") == "safe\n"
    assert (tmp_path / "other.txt").read_text(encoding="utf-8") == "other\n"


async def test_apply_patch_supports_quoted_paths(tmp_path: Path) -> None:
    target = tmp_path / "hello world.txt"
    target.write_text("old\n", encoding="utf-8")
    registry = ToolRegistry(tmp_path)
    patch = """diff --git "a/hello world.txt" "b/hello world.txt"
--- "a/hello world.txt"
+++ "b/hello world.txt"
@@ -1 +1 @@
-old
+new
"""

    result = json.loads(await registry.execute("apply_patch", {"patch": patch}))

    assert result["changed"] == ["hello world.txt"]
    assert target.read_text(encoding="utf-8") == "new\n"


@pytest.mark.skipif(os.name != "posix", reason="executable fixture requires POSIX")
async def test_git_apply_bounds_failure_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_git = tmp_path / "fake-git"
    fake_git.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "sys.stdin.buffer.read()\n"
        "sys.stderr.write('A' * 100_000 + 'TAIL')\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    monkeypatch.setattr(tools_module, "MAX_GIT_APPLY_OUTPUT_BYTES", 1_000)
    registry = ToolRegistry(tmp_path)

    with pytest.raises(ValueError) as exc_info:
        await registry._run_git_apply(str(fake_git), "patch", check=True)

    detail = str(exc_info.value)
    assert len(detail) < 1_200
    assert "output truncated; middle omitted" in detail
    assert detail.endswith("TAIL")


async def test_concurrent_results_preserve_order(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path)

    async def slow(arguments: dict[str, object]) -> str:
        await asyncio.sleep(float(arguments["delay"]))
        return str(arguments["value"])

    registry.register(ToolDefinition("ordered", "test", {"type": "object"}, slow))
    result = await registry.execute_many(
        [
            ("ordered", {"delay": 0.03, "value": "first"}),
            ("ordered", {"delay": 0, "value": "second"}),
        ]
    )
    assert result == ["first", "second"]


async def test_parallel_tool_concurrency_is_clamped_to_one(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path)

    async def immediate(arguments: dict[str, object]) -> str:
        return str(arguments["value"])

    registry.register(ToolDefinition("immediate", "test", {"type": "object"}, immediate))
    result = await asyncio.wait_for(
        registry.execute_many([("immediate", {"value": "ok"})], max_concurrency=0),
        1,
    )
    assert result == ["ok"]


async def test_parallel_tool_timeout_is_structured_and_preserves_other_results(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(tmp_path, tool_timeout_seconds=0.05)
    canceled = asyncio.Event()

    async def slow(arguments: dict[str, object]) -> str:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            canceled.set()
            raise
        return "unreachable"

    async def fast(arguments: dict[str, object]) -> str:
        return "done"

    registry.register(ToolDefinition("slow_timeout", "test", {"type": "object"}, slow))
    registry.register(ToolDefinition("fast", "test", {"type": "object"}, fast))
    outputs = await registry.execute_many_outputs(
        [("slow_timeout", {}), ("fast", {})], max_concurrency=2
    )
    timed_out = json.loads(outputs[0].text)
    assert timed_out["timed_out"] is True
    assert outputs[0].timed_out is True and outputs[0].elapsed_ms >= 40
    assert outputs[1].text == "done" and outputs[1].timed_out is False
    assert canceled.is_set()


async def test_mcp_wall_timeout_respects_chrome_millisecond_argument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tools_module, "MCP_TIMEOUT_GRACE_SECONDS", 0.01)
    registry = ToolRegistry(tmp_path, tool_timeout_seconds=1)
    canceled = asyncio.Event()

    async def blocked(arguments: dict[str, object]) -> str:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            canceled.set()
            raise
        return "unreachable"

    registry.register(ToolDefinition("mcp__chrome-devtools__navigate_page", "test", {}, blocked))
    started = time.monotonic()
    output = (
        await registry.execute_many_outputs(
            [("mcp__chrome-devtools__navigate_page", {"timeout": 10})]
        )
    )[0]

    assert time.monotonic() - started < 0.2
    assert output.timed_out is True
    assert canceled.is_set()


async def test_tool_wall_timeout_detaches_cancellation_suppressing_handler(
    tmp_path: Path,
) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    registry = ToolRegistry(
        paths.workspace,
        audit=AuditLog(paths),
        tool_timeout_seconds=0.05,
    )
    cancellation_seen = asyncio.Event()
    release = asyncio.Event()

    async def stubborn(arguments: dict[str, object]) -> str:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release.wait()
        return "late success must be discarded"

    registry.register(ToolDefinition("write_file", "test", {"type": "object"}, stubborn))
    started = time.monotonic()
    output = (await registry.execute_many_outputs([("write_file", {})], max_concurrency=1))[0]

    assert time.monotonic() - started < 0.5
    assert output.timed_out is True
    assert cancellation_seen.is_set()
    assert cancellation_module._DETACHED_CANCELLATIONS

    release.set()
    for _ in range(50):
        if not cancellation_module._DETACHED_CANCELLATIONS:
            break
        await asyncio.sleep(0.01)
    assert not cancellation_module._DETACHED_CANCELLATIONS
    entries = [
        json.loads(line)
        for line in next(paths.audit_dir.glob("audit-*.jsonl"))
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [(entry["decision"], entry["detail"]) for entry in entries] == [
        ("error", "Tool execution exceeded 0.1s timeout")
    ]


async def test_user_cancel_is_bounded_when_tool_suppresses_cancellation(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()

    async def stubborn(arguments: dict[str, object]) -> str:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()
        return "late"

    registry.register(ToolDefinition("stubborn", "test", {"type": "object"}, stubborn))
    cancel_event = asyncio.Event()
    batch = asyncio.create_task(
        registry.execute_many_outputs([("stubborn", {})], cancel_event=cancel_event)
    )
    await started.wait()
    before_cancel = time.monotonic()
    cancel_event.set()

    with pytest.raises(AgentCanceled):
        await asyncio.wait_for(batch, 0.5)
    assert time.monotonic() - before_cancel < 0.5
    assert cancellation_module._DETACHED_CANCELLATIONS
    release.set()
    await registry.close()
    for _ in range(50):
        if not cancellation_module._DETACHED_CANCELLATIONS:
            break
        await asyncio.sleep(0.01)
    assert not cancellation_module._DETACHED_CANCELLATIONS


async def test_native_task_cancel_is_bounded_without_cancel_event(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()

    async def stubborn(arguments: dict[str, object]) -> str:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()
        return "late"

    registry.register(ToolDefinition("stubborn", "test", {"type": "object"}, stubborn))
    running = asyncio.create_task(registry.execute("stubborn", {}))
    await started.wait()
    before_cancel = time.monotonic()
    running.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(running, 0.5)
    assert time.monotonic() - before_cancel < 0.5
    assert cancellation_module._DETACHED_CANCELLATIONS
    release.set()
    for _ in range(50):
        if not cancellation_module._DETACHED_CANCELLATIONS:
            break
        await asyncio.sleep(0.01)
    assert not cancellation_module._DETACHED_CANCELLATIONS


async def test_invalid_command_timeout_does_not_fail_parallel_batch(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(tmp_path)

    async def fast(arguments: dict[str, object]) -> str:
        return "fast result"

    registry.register(ToolDefinition("fast", "test", {"type": "object"}, fast))
    outputs = await registry.execute_many_outputs(
        [
            ("execute_command", {"command": "echo safe", "timeout": "invalid"}),
            ("fast", {}),
        ],
        max_concurrency=2,
    )

    assert json.loads(outputs[0].text)["invalid_arguments"] is True
    assert outputs[1].text == "fast result"


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("write_file", {"path": ["escape"], "content": "x"}),
        ("write_file", {"path": "safe.txt", "content": ["x"]}),
        ("execute_command", {"command": {"value": "echo unsafe"}}),
        ("execute_command", {"command": "echo safe", "timeout": True}),
        ("read_file", {"path": "missing.txt", "offset": True}),
        ("grep_code", {"pattern": "x", "regex": 1}),
        ("web_search", {"query": "x", "limit": 1.5}),
        ("create_project", {"path": "demo", "kind": "ruby"}),
        ("shell_start", {"cwd": ".", "unexpected": "value"}),
    ],
)
async def test_runtime_tool_schema_rejects_type_enum_and_extra_property_confusion(
    tmp_path: Path, name: str, arguments: dict[str, object]
) -> None:
    registry = ToolRegistry(tmp_path)

    result = json.loads(await registry.execute(name, arguments))  # type: ignore[arg-type]

    assert result["invalid_arguments"] is True
    assert not (tmp_path / "safe.txt").exists()
    assert not (tmp_path / "demo").exists()


async def test_invalid_arguments_are_rejected_before_approval_and_audited(
    tmp_path: Path,
) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    approvals = 0

    async def approve(name: str, arguments: dict[str, object]) -> ApprovalResult:
        nonlocal approvals
        approvals += 1
        return ApprovalResult.approve()

    registry = ToolRegistry(
        paths.workspace,
        audit=AuditLog(paths),
        approval_policy=ApprovalPolicy(True),
        approver=approve,
    )
    result = json.loads(
        await registry.execute(
            "execute_command",
            {"command": ["echo", "unsafe"]},  # type: ignore[dict-item]
        )
    )

    assert result["invalid_arguments"] is True
    assert approvals == 0
    entry = json.loads(next(paths.audit_dir.glob("audit-*.jsonl")).read_text(encoding="utf-8"))
    assert entry["decision"] == "error"
    assert "$.command must be string" in entry["detail"]


async def test_modified_approval_arguments_are_schema_validated_again(
    tmp_path: Path,
) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")

    async def modify(name: str, arguments: dict[str, object]) -> ApprovalResult:
        return ApprovalResult.modify({"command": "echo safe", "timeout": True})

    registry = ToolRegistry(
        paths.workspace,
        audit=AuditLog(paths),
        approval_policy=ApprovalPolicy(True),
        approver=modify,
    )
    result = json.loads(await registry.execute("execute_command", {"command": "echo safe"}))

    assert result["invalid_arguments"] is True
    entry = json.loads(next(paths.audit_dir.glob("audit-*.jsonl")).read_text(encoding="utf-8"))
    assert entry["decision"] == "error"
    assert entry["arguments"] == {"command": "echo safe", "timeout": True}


async def test_nested_tool_schema_is_bounded_and_blocks_handler(
    tmp_path: Path,
) -> None:
    calls = 0

    async def handler(arguments: dict[str, object]) -> str:
        nonlocal calls
        calls += 1
        return "called"

    registry = ToolRegistry(tmp_path)
    registry.register(
        ToolDefinition(
            "nested",
            "test",
            {
                "type": "object",
                "properties": {
                    "payload": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "maxItems": 2,
                                "items": {"type": "integer"},
                            }
                        },
                        "required": ["items"],
                        "additionalProperties": False,
                    }
                },
                "required": ["payload"],
                "additionalProperties": False,
            },
            handler,
        )
    )

    too_many = json.loads(await registry.execute("nested", {"payload": {"items": [1, 2, 3]}}))
    bool_integer = json.loads(await registry.execute("nested", {"payload": {"items": [True]}}))
    valid = await registry.execute("nested", {"payload": {"items": [1, 2]}})

    assert too_many["invalid_arguments"] is True
    assert bool_integer["invalid_arguments"] is True
    assert valid == "called"
    assert calls == 1


@pytest.mark.parametrize(
    "schema",
    [
        {"type": []},
        {"type": ["string", "string"]},
        {"enum": []},
        {"type": "string", "minLength": "1"},
        {"type": "array", "minItems": -1},
        {"type": "number", "minimum": float("nan")},
        {"type": "number", "minimum": 2, "maximum": 1},
        {"type": "object", "required": ["value", "value"]},
        {"type": "object", "properties": {"value": {"type": "unsupported"}}},
    ],
)
async def test_malformed_tool_schema_is_rejected_before_handler(
    tmp_path: Path, schema: dict[str, object]
) -> None:
    calls = 0

    async def handler(arguments: dict[str, object]) -> str:
        nonlocal calls
        calls += 1
        return "called"

    registry = ToolRegistry(tmp_path)
    registry.register(ToolDefinition("malformed", "test", schema, handler))

    result = json.loads(await registry.execute("malformed", {}))

    assert result["invalid_arguments"] is True
    assert calls == 0


async def test_schema_number_validation_preserves_arbitrary_precision_integers(
    tmp_path: Path,
) -> None:
    async def handler(arguments: dict[str, object]) -> str:
        return str(arguments["value"])

    registry = ToolRegistry(tmp_path)
    registry.register(
        ToolDefinition(
            "large_integer",
            "test",
            {
                "type": "object",
                "properties": {"value": {"type": "integer", "minimum": 0}},
                "required": ["value"],
                "additionalProperties": False,
            },
            handler,
        )
    )
    value = 10**1_000

    assert await registry.execute("large_integer", {"value": value}) == str(value)


async def test_policy_denial_is_structured(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path)
    result = json.loads(await registry.execute("read_file", {"path": "../nope"}))
    assert result["policy_denied"] is True


async def test_glob_and_grep(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("needle = True\n", encoding="utf-8")
    registry = ToolRegistry(tmp_path)
    glob = json.loads(await registry.execute("glob_files", {"pattern": "*.py"}))
    assert glob == {"matches": ["a.py"], "partial": False, "partial_reason": ""}
    grep = json.loads(await registry.execute("grep_code", {"pattern": "needle"}))
    assert grep["matches"]
    assert grep["partial"] is False
    assert grep["engine"] in {"rg", "python"}
    assert grep["suggested_reads"][0]["path"] == "a.py"


async def test_python_grep_fallback_bounds_files_that_grow_after_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "growing.py"
    source.write_text("x", encoding="utf-8")
    requested: list[int] = []

    class GrowingReader(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            requested.append(size)
            return super().read(size)

    def growing_open(
        _path: Path, mode: str = "r", *args: object, **kwargs: object
    ) -> GrowingReader:
        assert mode == "rb"
        return GrowingReader(b"needle!!!")

    registry = ToolRegistry(tmp_path)
    monkeypatch.setattr(tools_module, "MAX_GREP_FILE_BYTES", 8)
    monkeypatch.setattr(tools_module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(Path, "open", growing_open)

    result = json.loads(await registry.execute("grep_code", {"pattern": "needle"}))

    assert result["engine"] == "python"
    assert result["matches"] == []
    assert result["partial"] is True
    assert "Skipped 1 file(s) above 8 bytes" in result["partial_reason"]
    assert requested == [9]


async def test_python_grep_fallback_default_glob_finds_workspace_root_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "root.py").write_text("needle = True\n", encoding="utf-8")
    state = tmp_path / ".kairocli"
    state.mkdir()
    (state / "private.txt").write_text("needle private\n", encoding="utf-8")
    monkeypatch.setattr(tools_module.shutil, "which", lambda _name: None)
    registry = ToolRegistry(tmp_path)

    result = json.loads(await registry.execute("grep_code", {"pattern": "needle"}))

    assert result["engine"] == "python"
    assert result["matches"] == ["root.py:1:needle = True"]
    assert result["partial"] is False


def test_diff_capture_bounds_file_growth_after_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "growing.py"
    source.write_text("x", encoding="utf-8")
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

    monkeypatch.setattr(tools_module, "MAX_DISPLAY_DIFF_CHARS", 2)
    monkeypatch.setattr(Path, "open", growing_open)

    value, error = tools_module._capture_diff_text(source)

    assert value is None
    assert error == "file exceeds 2 display characters"
    assert requested == [9]


async def test_list_glob_and_grep_report_partial_results(tmp_path: Path) -> None:
    for index in range(3):
        (tmp_path / f"file-{index}.txt").write_text(f"Needle value {index}\n", encoding="utf-8")
    registry = ToolRegistry(tmp_path)
    listing = json.loads(await registry.execute("list_dir", {"path": ".", "offset": 1, "limit": 1}))
    assert listing["partial"] is True
    assert listing["next_offset"] == 2
    glob = json.loads(
        await registry.execute("glob_files", {"pattern": "*.txt", "path": ".", "max_results": 1})
    )
    assert glob["matches"] == ["file-0.txt"]
    assert glob["partial"] is True
    grep = json.loads(
        await registry.execute(
            "grep_code",
            {"pattern": "needle", "case_sensitive": False, "max_results": 1},
        )
    )
    assert len(grep["matches"]) == 1
    assert grep["partial"] is True
    assert "max_results=1" in grep["partial_reason"]


async def test_hitl_rejects_write(tmp_path: Path) -> None:
    async def reject(name: str, arguments: dict[str, object]) -> bool:
        return False

    registry = ToolRegistry(tmp_path, approval_policy=ApprovalPolicy(True), approver=reject)
    result = json.loads(
        await registry.execute("write_file", {"path": "blocked.txt", "content": "no"})
    )
    assert result["approval_denied"] is True
    assert not (tmp_path / "blocked.txt").exists()


async def test_policy_rejection_precedes_hitl(tmp_path: Path) -> None:
    approvals = 0

    async def approve(name: str, arguments: dict[str, object]) -> bool:
        nonlocal approvals
        approvals += 1
        return True

    registry = ToolRegistry(tmp_path, approval_policy=ApprovalPolicy(True), approver=approve)
    result = json.loads(
        await registry.execute("write_file", {"path": "../escape", "content": "no"})
    )
    assert result["policy_denied"] is True
    assert approvals == 0


async def test_hitl_modified_arguments_are_executed(tmp_path: Path) -> None:
    async def modify(name: str, arguments: dict[str, object]) -> ApprovalResult:
        return ApprovalResult.modify({"path": "safe.txt", "content": "changed"})

    registry = ToolRegistry(tmp_path, approval_policy=ApprovalPolicy(True), approver=modify)
    result = json.loads(
        await registry.execute("write_file", {"path": "original.txt", "content": "original"})
    )
    assert result["path"] == "safe.txt"
    assert not (tmp_path / "original.txt").exists()
    assert (tmp_path / "safe.txt").read_text(encoding="utf-8") == "changed"


async def test_hitl_modified_arguments_are_rechecked(tmp_path: Path) -> None:
    async def escape(name: str, arguments: dict[str, object]) -> ApprovalResult:
        return ApprovalResult.modify({"path": "../escape.txt", "content": "no"})

    registry = ToolRegistry(tmp_path, approval_policy=ApprovalPolicy(True), approver=escape)
    result = json.loads(
        await registry.execute("write_file", {"path": "safe.txt", "content": "original"})
    )
    assert result["policy_denied"] is True


async def test_approve_all_skips_future_prompts_for_same_tool(tmp_path: Path) -> None:
    approvals = 0

    async def approve_all(name: str, arguments: dict[str, object]) -> ApprovalResult:
        nonlocal approvals
        approvals += 1
        return ApprovalResult.approve_all()

    registry = ToolRegistry(tmp_path, approval_policy=ApprovalPolicy(True), approver=approve_all)
    await registry.execute("write_file", {"path": "one.txt", "content": "1"})
    await registry.execute("write_file", {"path": "two.txt", "content": "2"})
    assert approvals == 1


async def test_parallel_approve_all_prompts_once_for_same_tool(tmp_path: Path) -> None:
    approvals = 0
    entered = asyncio.Event()
    release = asyncio.Event()

    async def approve_all(name: str, arguments: dict[str, object]) -> ApprovalResult:
        nonlocal approvals
        approvals += 1
        entered.set()
        await release.wait()
        return ApprovalResult.approve_all()

    registry = ToolRegistry(tmp_path, approval_policy=ApprovalPolicy(True), approver=approve_all)
    first = asyncio.create_task(registry.execute("write_file", {"path": "one.txt", "content": "1"}))
    await asyncio.wait_for(entered.wait(), timeout=1)
    second = asyncio.create_task(
        registry.execute("write_file", {"path": "two.txt", "content": "2"})
    )
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(first, second)

    assert approvals == 1
    assert (tmp_path / "one.txt").read_text(encoding="utf-8") == "1"
    assert (tmp_path / "two.txt").read_text(encoding="utf-8") == "2"


async def test_waiting_for_approval_lock_is_cancelable(tmp_path: Path) -> None:
    approvals = 0
    entered = asyncio.Event()
    release = asyncio.Event()

    async def approve(name: str, arguments: dict[str, object]) -> ApprovalResult:
        nonlocal approvals
        approvals += 1
        entered.set()
        await release.wait()
        return ApprovalResult.approve()

    registry = ToolRegistry(tmp_path, approval_policy=ApprovalPolicy(True), approver=approve)
    first = asyncio.create_task(registry.execute("write_file", {"path": "one.txt", "content": "1"}))
    await asyncio.wait_for(entered.wait(), timeout=1)
    canceled = asyncio.Event()
    second = asyncio.create_task(
        registry.execute("write_file", {"path": "two.txt", "content": "2"}, canceled)
    )
    await asyncio.sleep(0)
    canceled.set()
    with pytest.raises(AgentCanceled):
        await asyncio.wait_for(second, timeout=1)
    release.set()
    await first

    assert approvals == 1
    assert not (tmp_path / "two.txt").exists()


async def test_invalid_approval_handler_result_fails_closed(tmp_path: Path) -> None:
    async def invalid(name: str, arguments: dict[str, object]) -> object:
        return None

    registry = ToolRegistry(
        tmp_path,
        approval_policy=ApprovalPolicy(True),
        approver=invalid,  # type: ignore[arg-type]
    )
    result = json.loads(
        await registry.execute("write_file", {"path": "blocked.txt", "content": "no"})
    )

    assert result["approval_denied"] is True
    assert result["approval_decision"] == "rejected"
    assert "invalid decision" in result["error"]
    assert not (tmp_path / "blocked.txt").exists()


async def test_mcp_transition_drains_calls_and_queued_call_uses_new_handler(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(tmp_path)
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    transition_entered = asyncio.Event()
    release_transition = asyncio.Event()
    new_handler_called = asyncio.Event()

    async def old_handler(arguments: dict[str, object]) -> str:
        first_entered.set()
        await release_first.wait()
        return "old"

    async def new_handler(arguments: dict[str, object]) -> str:
        new_handler_called.set()
        return "new"

    definition = ToolDefinition(
        "mcp__demo__work",
        "work",
        {"type": "object", "properties": {}},
        old_handler,
    )
    registry.register(definition)
    first = asyncio.create_task(registry.execute("mcp__demo__work", {}))
    await asyncio.wait_for(first_entered.wait(), timeout=1)

    async def transition() -> None:
        async with registry.mcp_server_transition("demo"):
            transition_entered.set()
            registry.register(
                ToolDefinition(
                    definition.name,
                    definition.description,
                    definition.parameters,
                    new_handler,
                )
            )
            await release_transition.wait()

    changing = asyncio.create_task(transition())
    await asyncio.sleep(0)
    second = asyncio.create_task(registry.execute("mcp__demo__work", {}))
    await asyncio.sleep(0)
    assert not transition_entered.is_set()
    assert not new_handler_called.is_set()

    release_first.set()
    assert await first == "old"
    await asyncio.wait_for(transition_entered.wait(), timeout=1)
    await asyncio.sleep(0)
    assert not new_handler_called.is_set()

    release_transition.set()
    await changing
    assert await second == "new"
    assert new_handler_called.is_set()


async def test_canceled_mcp_gate_wait_does_not_block_later_calls(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path)

    async def handler(arguments: dict[str, object]) -> str:
        return "ok"

    registry.register(
        ToolDefinition(
            "mcp__demo__read",
            "read",
            {"type": "object", "properties": {}},
            handler,
        )
    )

    async with registry.mcp_server_transition("demo"):
        canceled = asyncio.Event()
        waiting = asyncio.create_task(registry.execute("mcp__demo__read", {}, canceled))
        await asyncio.sleep(0)
        canceled.set()
        with pytest.raises(AgentCanceled):
            await asyncio.wait_for(waiting, timeout=1)

    assert await registry.execute("mcp__demo__read", {}) == "ok"


async def test_unknown_mcp_tools_do_not_allocate_unbounded_server_gates(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(tmp_path)

    for index in range(100):
        result = json.loads(await registry.execute(f"mcp__unknown-{index}__read", {}))
        assert result["error"].startswith("Unknown tool:")

    assert registry._mcp_server_gates == {}


async def test_audit_records_structured_and_raised_tool_failures_as_errors(
    tmp_path: Path,
) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    registry = ToolRegistry(paths.workspace, audit=AuditLog(paths))

    async def structured_error(arguments: dict[str, object]) -> dict[str, str]:
        return {"error": "structured failure token=private-structured"}

    registry.register(ToolDefinition("write_file", "test", {"type": "object"}, structured_error))
    structured_output = await registry.execute("write_file", {})
    assert "structured failure" in structured_output
    assert "private-structured" not in structured_output

    async def raised_error(arguments: dict[str, object]) -> str:
        raise RuntimeError(
            "Authorization: Bearer private-raised "
            "github_pat_abcdefghijklmnopqrstuvwxyz raised failure"
        )

    registry.register(ToolDefinition("execute_command", "test", {"type": "object"}, raised_error))
    raised_output = await registry.execute("execute_command", {"command": "echo safe fixture"})
    assert "raised failure" in raised_output
    assert "private-raised" not in raised_output
    assert "github_pat_abcdefghijklmnopqrstuvwxyz" not in raised_output

    async def modify_to_denied(name: str, arguments: dict[str, object]) -> ApprovalResult:
        return ApprovalResult.modify({"command": "sudo dangerous"})

    guarded = ToolRegistry(
        paths.workspace,
        audit=AuditLog(paths),
        approval_policy=ApprovalPolicy(True),
        approver=modify_to_denied,
    )
    denied = json.loads(await guarded.execute("execute_command", {"command": "echo safe"}))
    assert denied["policy_denied"] is True

    async def never_finishes(arguments: dict[str, object]) -> str:
        await asyncio.Event().wait()
        return "unreachable"

    timed = ToolRegistry(paths.workspace, audit=AuditLog(paths), tool_timeout_seconds=0.01)
    timed.register(ToolDefinition("write_file", "test", {"type": "object"}, never_finishes))
    timed_output = (await timed.execute_many_outputs([("write_file", {})], max_concurrency=1))[0]
    assert timed_output.timed_out is True

    entries = [
        json.loads(line)
        for line in next(paths.audit_dir.glob("audit-*.jsonl"))
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [(entry["tool"], entry["decision"]) for entry in entries] == [
        ("write_file", "error"),
        ("execute_command", "error"),
        ("execute_command", "denied"),
        ("write_file", "error"),
    ]
    assert "structured failure" in entries[0]["detail"]
    assert "raised failure" in entries[1]["detail"]
    assert entries[2]["arguments"] == {"command": "sudo dangerous"}
    assert "timeout" in entries[3]["detail"]


async def test_tool_argument_errors_are_redacted_and_bounded(tmp_path: Path) -> None:
    secret = "long-field-" * 800 + " token=private-value"
    registry = ToolRegistry(tmp_path)
    registry.register(
        ToolDefinition(
            "bounded_error",
            "test",
            {
                "type": "object",
                "properties": {},
                "required": [secret],
                "additionalProperties": False,
            },
            lambda arguments: asyncio.sleep(0, result="unreachable"),
        )
    )

    raw = await registry.execute("bounded_error", {})
    result = json.loads(raw)

    assert result["invalid_arguments"] is True
    assert len(result["error"]) <= 4_000
    assert "private-value" not in result["error"]
    assert "tool error truncated" in result["error"]


async def test_tool_handler_unprintable_error_returns_stable_failure(tmp_path: Path) -> None:
    class UnprintableError(RuntimeError):
        def __str__(self) -> str:
            raise KeyboardInterrupt

    async def fail(arguments: dict[str, object]) -> str:
        raise UnprintableError()

    registry = ToolRegistry(tmp_path)
    registry.register(ToolDefinition("broken", "test", {"type": "object"}, fail))

    result = json.loads(await registry.execute("broken", {}))
    assert result["error"] == "UnprintableError message unavailable"


async def test_write_file_rejects_content_larger_than_limit(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path)
    result = json.loads(
        await registry.execute(
            "write_file", {"path": "huge.txt", "content": "x" * (5 * 1024 * 1024 + 1)}
        )
    )
    assert result["policy_denied"] is True
    assert not (tmp_path / "huge.txt").exists()


async def test_cancel_stops_parallel_tool_batch(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path)
    started = asyncio.Event()
    canceled = 0

    async def slow(arguments: dict[str, object]) -> str:
        nonlocal canceled
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            canceled += 1
            raise
        return "unreachable"

    registry.register(ToolDefinition("slow", "test", {"type": "object"}, slow))
    cancel_event = asyncio.Event()
    batch = asyncio.create_task(
        registry.execute_many([("slow", {}), ("slow", {})], cancel_event=cancel_event)
    )
    await started.wait()
    cancel_event.set()
    with pytest.raises(AgentCanceled):
        await asyncio.wait_for(batch, 1)
    assert canceled == 2


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group assertion")
async def test_cancel_terminates_command_process_group(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path)
    marker = tmp_path / "orphan-marker"
    script = f"import time; time.sleep(0.8); open({str(marker)!r}, 'w').write('orphan')"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"
    cancel_event = asyncio.Event()
    running = asyncio.create_task(
        registry.execute("execute_command", {"command": command, "timeout": 5}, cancel_event)
    )
    await asyncio.sleep(0.15)
    cancel_event.set()
    with pytest.raises(AgentCanceled):
        await asyncio.wait_for(running, 1)
    await asyncio.sleep(0.8)
    assert not marker.exists()


async def test_canceled_dangerous_tool_is_audited_with_ambiguous_side_effect(
    tmp_path: Path,
) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    registry = ToolRegistry(paths.workspace, audit=AuditLog(paths))
    schema = registry._tools["execute_command"].parameters
    started = asyncio.Event()

    async def blocked(arguments: dict[str, object]) -> str:
        started.set()
        await asyncio.Event().wait()
        return "unreachable"

    registry.register(ToolDefinition("execute_command", "fixture", schema, blocked))
    cancel_event = asyncio.Event()
    running = asyncio.create_task(
        registry.execute(
            "execute_command",
            {"command": "echo safe", "timeout": 5},
            cancel_event,
        )
    )
    await started.wait()
    cancel_event.set()

    with pytest.raises(AgentCanceled):
        await running

    entry = json.loads(next(paths.audit_dir.glob("audit-*.jsonl")).read_text(encoding="utf-8"))
    assert entry["tool"] == "execute_command"
    assert entry["decision"] == "canceled"
    assert entry["arguments"] == {"command": "echo safe", "timeout": 5}
    assert "side effects may have completed" in entry["detail"]


async def test_command_timeout_has_stable_result_shape(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path)
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote('import time; time.sleep(1)')}"
    result = json.loads(
        await registry.execute("execute_command", {"command": command, "timeout": 0.1})
    )
    assert result == {
        "exit_code": None,
        "timed_out": True,
        "canceled": False,
        "stdout": "",
        "stderr": "Command exceeded 0.1s timeout",
        "stdout_bytes": 0,
        "stderr_bytes": 0,
        "stdout_truncated": False,
        "stderr_truncated": False,
    }


async def test_process_group_permission_error_falls_back_to_child_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        pid = 123
        returncode: int | None = None
        terminated = False

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        async def wait(self) -> int:
            assert self.returncode is not None
            return self.returncode

    process = Process()

    def deny_process_group(_pid: int, _signal: int) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(process_module.os, "killpg", deny_process_group)

    await process_module._terminate_process_tree(process)  # type: ignore[arg-type]

    assert process.terminated is True


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group assertion")
async def test_command_timeout_includes_background_pipe_owners(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path)
    marker = tmp_path / "background-marker"
    script = f"import time; time.sleep(0.8); open({str(marker)!r}, 'w').write('orphan')"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)} &"

    result = json.loads(
        await registry.execute("execute_command", {"command": command, "timeout": 0.1})
    )

    assert result["timed_out"] is True
    assert result["exit_code"] is None
    await asyncio.sleep(0.8)
    assert not marker.exists()


@pytest.mark.parametrize("command", ["", " \t\n", "x" * (100 * 1024 + 1)])
async def test_command_rejects_empty_and_oversized_input_before_approval(
    tmp_path: Path, command: str
) -> None:
    approvals = 0

    async def approve(name: str, arguments: dict[str, object]) -> bool:
        nonlocal approvals
        approvals += 1
        return True

    registry = ToolRegistry(
        tmp_path,
        approval_policy=ApprovalPolicy(True),
        approver=approve,
    )

    result = json.loads(await registry.execute("execute_command", {"command": command}))

    assert result.get("invalid_arguments") is True or result.get("policy_denied") is True
    assert approvals == 0


async def test_command_output_is_streamed_with_head_and_tail_metadata(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path)
    script = "import sys; print('HEAD' + 'x'*150000 + 'TAIL'); sys.stderr.write('e'*150000)"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"
    result = json.loads(
        await registry.execute("execute_command", {"command": command, "timeout": 5})
    )
    assert result["exit_code"] == 0
    assert result["stdout_bytes"] > 100_000
    assert result["stderr_bytes"] == 150_000
    assert result["stdout_truncated"] is True
    assert result["stderr_truncated"] is True
    assert result["stdout"].startswith("HEAD")
    assert result["stdout"].rstrip().endswith("TAIL")
    assert "middle omitted" in result["stdout"]


async def test_structured_tool_output_preserves_images_and_text_compatibility(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(tmp_path)

    async def image_tool(arguments: dict[str, object]) -> ToolOutput:
        return ToolOutput("text result", ("data:image/png;base64,aGVsbG8=",))

    registry.register(ToolDefinition("image_tool", "test", {"type": "object"}, image_tool))
    output = await registry.execute_output("image_tool", {})
    assert output.has_images
    assert output.image_urls == ("data:image/png;base64,aGVsbG8=",)
    assert await registry.execute("image_tool", {}) == "text result"


async def test_arbitrary_tool_output_has_global_context_budget(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path)

    async def verbose(arguments: dict[str, object]) -> ToolOutput:
        return ToolOutput(
            "HEAD" + "x" * 250_000 + "TAIL",
            ("data:image/png;base64,aGVsbG8=",),
        )

    registry.register(ToolDefinition("verbose", "test", {"type": "object"}, verbose))
    output = await registry.execute_output("verbose", {})
    assert output.truncated is True
    assert output.original_chars == 250_008
    assert output.text.startswith("HEAD") and output.text.endswith("TAIL")
    assert "tool output truncated" in output.text
    assert output.has_images


async def test_large_structured_tool_result_remains_valid_json(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path)

    async def verbose_json(arguments: dict[str, object]) -> dict[str, str]:
        return {"payload": "x" * 300_000}

    registry.register(ToolDefinition("verbose_json", "test", {"type": "object"}, verbose_json))
    output = await registry.execute_output("verbose_json", {})
    envelope = json.loads(output.text)
    assert output.truncated is True
    assert envelope["partial"] is True
    assert envelope["original_chars"] > 300_000
