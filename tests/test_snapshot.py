import asyncio
from pathlib import Path

import pytest

import kairocli.snapshot as snapshot_module
from kairocli.paths import KairoPaths
from kairocli.snapshot import SnapshotConfig, SnapshotError, SnapshotService


async def test_snapshot_capture_and_restore(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    file = workspace / "state.txt"
    file.write_text("one", encoding="utf-8")
    service = SnapshotService(KairoPaths.discover(workspace, tmp_path / "home"))
    first = await service.capture("pre-turn")
    file.write_text("two", encoding="utf-8")
    created_later = workspace / "later.txt"
    created_later.write_text("later", encoding="utf-8")
    await service.capture("post-turn")
    await service.restore(first)
    assert file.read_text(encoding="utf-8") == "one"
    assert not created_later.exists()


async def test_snapshot_excludes_ignored_project_runtime_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    (workspace / ".gitignore").write_text(".kairocli/\n", encoding="utf-8")
    runtime_dir = workspace / ".kairocli"
    runtime_dir.mkdir()
    runtime_file = runtime_dir / "state.db"
    runtime_file.write_text("private-one", encoding="utf-8")
    tracked_file = workspace / "state.txt"
    tracked_file.write_text("one", encoding="utf-8")

    service = SnapshotService(KairoPaths.discover(workspace, tmp_path / "home"))
    first = await service.capture("pre-turn")
    runtime_file.write_text("private-two", encoding="utf-8")
    tracked_file.write_text("two", encoding="utf-8")
    await service.capture("post-turn")
    await service.restore(first)

    assert tracked_file.read_text(encoding="utf-8") == "one"
    assert runtime_file.read_text(encoding="utf-8") == "private-two"
    exclude_file = service.paths.snapshot_dir / "info" / "exclude"
    assert "/.kairocli/" in exclude_file.read_text(encoding="utf-8").splitlines()


async def test_snapshot_never_captures_or_restores_workspace_git_metadata(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    git_dir = workspace / ".git"
    git_dir.mkdir()
    git_config = git_dir / "config"
    git_config.write_text("original metadata", encoding="utf-8")
    state = workspace / "state.txt"
    state.write_text("before", encoding="utf-8")
    service = SnapshotService(KairoPaths.discover(workspace, tmp_path / "home"))
    target = await service.capture("pre-turn protect-git")

    git_config.write_text("current metadata", encoding="utf-8")
    state.write_text("after", encoding="utf-8")
    await service.capture("post-turn protect-git")
    tree = await service._tree_files(  # noqa: SLF001
        target, service._environment(work_tree=True)  # noqa: SLF001
    )
    result = await service.restore(target)

    assert not any(path == ".git" or path.startswith(".git/") for path in tree)
    assert result.success is True
    assert state.read_text(encoding="utf-8") == "before"
    assert git_config.read_text(encoding="utf-8") == "current metadata"


async def test_restore_removes_workspace_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("must survive", encoding="utf-8")
    state = workspace / "state.txt"
    state.write_text("before", encoding="utf-8")
    service = SnapshotService(KairoPaths.discover(workspace, tmp_path / "home"))
    target = await service.capture("pre-turn symlink")
    link = workspace / "outside-link"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")
    await service.capture("post-turn symlink")

    result = await service.restore(target)

    assert result.success is True
    assert not link.exists() and not link.is_symlink()
    assert outside.read_text(encoding="utf-8") == "must survive"
    assert result.removed_files == ("outside-link",)


async def test_restore_pre_turn_ignores_post_turn_and_captures_undo_point(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    state = workspace / "state.txt"
    state.write_text("before", encoding="utf-8")
    service = SnapshotService(KairoPaths.discover(workspace, tmp_path / "home"))

    await service.capture("pre-turn react-1\n\nbefore task")
    state.write_text("after", encoding="utf-8")
    added = workspace / "added.txt"
    added.write_text("new", encoding="utf-8")
    await service.capture("post-turn react-1\n\nafter task")

    result = await service.restore_pre_turn(1)

    assert result.success is True
    assert state.read_text(encoding="utf-8") == "before"
    assert not added.exists()
    snapshots = await service.list(5)
    assert snapshots[0].phase == "pre-restore"
    assert [item.phase for item in snapshots[1:3]] == ["post-turn", "pre-turn"]
    assert result.removed_files == ("added.txt",)


async def test_restore_handles_file_directory_shape_changes(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    node = workspace / "node"
    node.write_text("file", encoding="utf-8")
    service = SnapshotService(KairoPaths.discover(workspace, tmp_path / "home"))
    target = await service.capture("pre-turn react-shape")

    node.unlink()
    node.mkdir()
    (node / "child.txt").write_text("child", encoding="utf-8")
    await service.capture("post-turn react-shape")

    result = await service.restore(target)

    assert result.success is True
    assert node.is_file()
    assert node.read_text(encoding="utf-8") == "file"


async def test_restore_handles_filename_with_newline(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    base = workspace / "base.txt"
    base.write_text("base", encoding="utf-8")
    service = SnapshotService(KairoPaths.discover(workspace, tmp_path / "home"))
    target = await service.capture("pre-turn newline-name")

    unusual = workspace / "line\nbreak.txt"
    unusual.write_text("later", encoding="utf-8")
    await service.capture("post-turn newline-name")

    result = await service.restore(target)

    assert result.success is True
    assert not unusual.exists()
    assert result.removed_files == ("line\nbreak.txt",)


async def test_restore_failure_automatically_rolls_back_pre_restore_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    state = workspace / "state.txt"
    target_only = workspace / "target-only.txt"
    state.write_text("target", encoding="utf-8")
    target_only.write_text("target-only", encoding="utf-8")
    service = SnapshotService(KairoPaths.discover(workspace, tmp_path / "home"))
    target = await service.capture("pre-turn rollback-target")

    state.write_text("current", encoding="utf-8")
    target_only.unlink()
    await service.capture("post-turn rollback-current")
    real_git = service._git
    failed = False

    async def fail_target_checkout(
        *args: str, env: dict[str, str] | None = None
    ) -> str:
        nonlocal failed
        if len(args) >= 2 and args[0] == "checkout" and args[1] == target and not failed:
            failed = True
            state.write_text("partial", encoding="utf-8")
            target_only.write_text("partial-target-only", encoding="utf-8")
            raise SnapshotError("simulated checkout failure")
        return await real_git(*args, env=env)

    monkeypatch.setattr(service, "_git", fail_target_checkout)

    with pytest.raises(SnapshotError, match="simulated checkout failure"):
        await service.restore(target)

    assert failed is True
    assert state.read_text(encoding="utf-8") == "current"
    assert not target_only.exists()
    assert (await service.list(1))[0].phase == "pre-restore"


async def test_background_capture_config_status_and_clean(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    paths = KairoPaths.discover(workspace, tmp_path / "home")
    repository = tmp_path / "custom-side-git"
    service = SnapshotService(
        paths,
        SnapshotConfig(
            max_snapshots=3,
            excludes=("/.kairocli/", "/generated/"),
            snapshots_root=repository,
        ),
    )
    (workspace / "kept.txt").write_text("yes", encoding="utf-8")
    generated = workspace / "generated"
    generated.mkdir()
    (generated / "ignored.txt").write_text("no", encoding="utf-8")

    service.capture_background("post-turn background-1")
    await service.wait_idle()

    snapshots = await service.list()
    assert len(snapshots) == 1
    assert snapshots[0].turn_id == "background-1"
    assert "Maximum listed: 3" in await service.status()
    tree = await service._git(  # noqa: SLF001 - contract-check isolated repository contents
        "ls-tree", "-r", "--name-only", "HEAD", env=service._environment()  # noqa: SLF001
    )
    assert tree.splitlines() == ["kept.txt"]
    assert await service.clean() is True
    assert not service.repository.exists()
    assert await service.clean() is False


async def test_background_capture_consumes_unexpected_and_callback_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = SnapshotService(KairoPaths.discover(tmp_path / "work", tmp_path / "home"))
    received: list[SnapshotError] = []
    loop = asyncio.get_running_loop()
    unhandled: list[dict[str, object]] = []
    previous_handler = loop.get_exception_handler()

    async def fail_capture(_message: str) -> str:
        raise RuntimeError("unexpected capture failure")

    def fail_callback(error: SnapshotError) -> None:
        received.append(error)
        raise RuntimeError("observer failure")

    monkeypatch.setattr(service, "capture", fail_capture)
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    try:
        service.capture_background("post-turn", fail_callback)
        await service.wait_idle()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert len(received) == 1
    assert "unexpected capture failure" in str(received[0])
    assert unhandled == []


async def test_snapshot_close_bounds_uncooperative_background_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = SnapshotService(KairoPaths.discover(tmp_path / "work", tmp_path / "home"))
    release = asyncio.Event()

    async def uncooperative_capture(_message: str) -> str:
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return "late"

    monkeypatch.setattr(service, "capture", uncooperative_capture)
    monkeypatch.setattr(snapshot_module, "SNAPSHOT_SHUTDOWN_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(snapshot_module, "SNAPSHOT_CANCEL_GRACE_SECONDS", 0.01)
    service.capture_background("post-turn")
    await asyncio.sleep(0)

    await asyncio.wait_for(service.close(), 0.2)

    assert service._background  # noqa: SLF001 - verifies bounded isolation
    release.set()
    await service.wait_idle()
    assert service._background == set()  # noqa: SLF001


async def test_snapshot_close_rejects_late_capture_without_spawning_task(
    tmp_path: Path,
) -> None:
    service = SnapshotService(KairoPaths.discover(tmp_path / "work", tmp_path / "home"))
    received: list[SnapshotError] = []

    await service.close()
    service.capture_background("post-turn late", received.append)

    assert len(received) == 1
    assert "closed" in str(received[0]).casefold()
    assert service._background == set()  # noqa: SLF001
    with pytest.raises(SnapshotError, match="closed"):
        await service.capture("post-turn late")


async def test_snapshot_close_finishes_bounded_cleanup_before_propagating_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = SnapshotService(KairoPaths.discover(tmp_path / "work", tmp_path / "home"))
    started = asyncio.Event()
    release = asyncio.Event()

    async def uncooperative_capture(_message: str) -> str:
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return "late"

    monkeypatch.setattr(service, "capture", uncooperative_capture)
    monkeypatch.setattr(snapshot_module, "SNAPSHOT_SHUTDOWN_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(snapshot_module, "SNAPSHOT_CANCEL_GRACE_SECONDS", 0.01)
    service.capture_background("post-turn")
    await started.wait()
    closing = asyncio.create_task(service.close())
    await asyncio.sleep(0)
    closing.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(closing, 0.2)
    assert service._close_task is not None  # noqa: SLF001
    assert service._close_task.done()  # noqa: SLF001
    assert service._background  # noqa: SLF001

    release.set()
    await service.wait_idle()
    assert service._background == set()  # noqa: SLF001


async def test_snapshot_rejects_symlinked_user_container(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (home / ".kairocli").symlink_to(outside, target_is_directory=True)
    service = SnapshotService(KairoPaths.discover(workspace, home))

    with pytest.raises(SnapshotError, match="symlink component"):
        await service.initialize()

    assert list(outside.iterdir()) == []


async def test_snapshot_rejects_oversized_git_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    service = SnapshotService(KairoPaths.discover(workspace, tmp_path / "home"))

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = asyncio.StreamReader()
            self.stdout.feed_data(b"x" * 33)
            self.stdout.feed_eof()
            self.stderr = asyncio.StreamReader()
            self.stderr.feed_eof()
            self.returncode = 0

        async def wait(self) -> int:
            return self.returncode

    async def create_process(*args: object, **kwargs: object) -> FakeProcess:
        return FakeProcess()

    monkeypatch.setattr(snapshot_module, "MAX_GIT_STDOUT_BYTES", 32)
    monkeypatch.setattr(snapshot_module.asyncio, "create_subprocess_exec", create_process)

    with pytest.raises(SnapshotError, match="output exceeds 32 bytes"):
        await service._git("ls-tree")  # noqa: SLF001 - exercise bounded subprocess contract


async def test_snapshot_sanitizes_nul_in_commit_message(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    (workspace / "state.txt").write_text("state", encoding="utf-8")
    service = SnapshotService(KairoPaths.discover(workspace, tmp_path / "home"))

    revision = await service.capture("pre-turn nul\x00message")
    snapshots = await service.list(1)

    assert len(revision) >= 40
    assert snapshots[0].message == "pre-turn nul�message"
