from __future__ import annotations

import asyncio
import builtins
import hashlib
import os
import re
import shutil
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .brand import PROJECT_DIR_NAME
from .paths import KairoPaths, reject_symlink_components
from .policy import PathGuard
from .text_safety import safe_text

DEFAULT_EXCLUDES = (
    "/.git/",
    f"/{PROJECT_DIR_NAME}/",
    "/target/",
    "/node_modules/",
    "/dist/",
    "/.idea/",
    "*.class",
    "*.jar",
)
VALID_PHASES = {"pre-turn", "post-turn", "pre-restore"}
GIT_TIMEOUT_SECONDS = 120
MAX_GIT_STDOUT_BYTES = 16 * 1024 * 1024
MAX_GIT_STDERR_BYTES = 256 * 1024
MAX_SNAPSHOT_LIST_LIMIT = 1_000
MAX_SNAPSHOT_MESSAGE_CHARS = 4_096
SNAPSHOT_SHUTDOWN_GRACE_SECONDS = 2.0
SNAPSHOT_CANCEL_GRACE_SECONDS = 0.5
_PROCESS_READ_CHUNK_BYTES = 64 * 1024
_CHECKOUT_BATCH_PATHS = 256
_CHECKOUT_BATCH_BYTES = 128 * 1024


@dataclass(frozen=True, slots=True)
class SnapshotConfig:
    enabled: bool = True
    max_snapshots: int = 50
    excludes: tuple[str, ...] = DEFAULT_EXCLUDES
    snapshots_root: Path | None = None

    @classmethod
    def from_environment(cls) -> SnapshotConfig:
        enabled = _read_bool("KAIROCLI_SNAPSHOT_ENABLED", True)
        try:
            maximum = min(
                max(1, int(os.getenv("KAIROCLI_SNAPSHOT_MAX", "50"))),
                MAX_SNAPSHOT_LIST_LIMIT,
            )
        except ValueError:
            maximum = 50
        configured = tuple(
            item.strip()
            for item in os.getenv("KAIROCLI_SNAPSHOT_EXCLUDES", "").split(",")
            if item.strip()
        )
        excludes = tuple(dict.fromkeys((*DEFAULT_EXCLUDES, *configured)))
        raw_repository = os.getenv("KAIROCLI_SNAPSHOT_DIR", "").strip()
        snapshots_root = Path(raw_repository).expanduser() if raw_repository else None
        return cls(enabled, maximum, excludes, snapshots_root)


@dataclass(frozen=True, slots=True)
class Snapshot:
    revision: str
    message: str
    created_at: str = ""
    phase: str = "post-turn"
    turn_id: str = ""

    @property
    def short_revision(self) -> str:
        return self.revision[:10]


@dataclass(frozen=True, slots=True)
class RestoreResult:
    success: bool
    revision: str | None
    message: str
    restored_files: tuple[str, ...] = ()
    removed_files: tuple[str, ...] = ()


class SnapshotError(RuntimeError):
    """Raised when the isolated Side-Git repository cannot complete an operation."""


class SnapshotService:
    def __init__(self, paths: KairoPaths, config: SnapshotConfig | None = None) -> None:
        self.paths = paths
        self.config = config or SnapshotConfig.from_environment()
        repository = (
            _side_git_path(self.config.snapshots_root, paths.workspace)
            if self.config.snapshots_root is not None
            else paths.snapshot_dir
        )
        self.repository = Path(os.path.abspath(repository))
        self.guard = PathGuard(paths.workspace)
        self._lock = asyncio.Lock()
        self._background: set[asyncio.Task[None]] = set()
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None

    async def initialize(self) -> None:
        self._ensure_open()
        if not self.config.enabled:
            return
        self._validate_repository_path()
        resolved_repository = self.repository.resolve(strict=False)
        if resolved_repository == self.paths.workspace or resolved_repository.is_relative_to(
            self.paths.workspace
        ):
            raise SnapshotError("Snapshot repository must be outside the workspace")
        self.repository.mkdir(parents=True, exist_ok=True)
        self._validate_repository_path()
        if not (self.repository / "HEAD").exists():
            await self._git("init", "--bare", str(self.repository))
        await asyncio.to_thread(self._write_excludes)

    def _write_excludes(self) -> None:
        self._validate_repository_path()
        exclude_file = self.repository / "info" / "exclude"
        exclude_file.parent.mkdir(parents=True, exist_ok=True)
        self._validate_repository_path()
        try:
            reject_symlink_components(exclude_file, "Snapshot exclude file")
        except ValueError as exc:
            raise SnapshotError(safe_text(exc)) from exc
        body = "# Managed by Kairo CLI side-history snapshots\n"
        body += "".join(f"{item}\n" for item in self.config.excludes)
        exclude_file.write_text(body, encoding="utf-8")

    async def capture(self, message: str) -> str:
        self._ensure_open()
        if not self.config.enabled:
            return ""
        async with self._lock:
            self._ensure_open()
            return await self._capture_locked(message)

    async def _capture_locked(self, message: str) -> str:
        await self.initialize()
        env = self._environment(work_tree=True)
        await self._git("add", "-A", env=env)
        await self._git(
            "-c",
            "user.name=Kairo CLI Snapshot",
            "-c",
            "user.email=snapshot@kairocli.local",
            "commit",
            "--allow-empty",
            "-m",
            _safe_commit_message(message),
            env=env,
        )
        return (await self._git("rev-parse", "HEAD", env=env)).strip()

    def capture_background(
        self,
        message: str,
        on_error: Callable[[SnapshotError], None] | None = None,
    ) -> None:
        if not self.config.enabled:
            return
        if self._closing:
            if on_error is not None:
                try:
                    on_error(SnapshotError("Snapshot service is closed"))
                except Exception:
                    pass
            return

        async def run() -> None:
            try:
                await self.capture(message)
            except Exception as exc:
                error = exc if isinstance(exc, SnapshotError) else SnapshotError(safe_text(exc))
                if on_error is not None:
                    try:
                        on_error(error)
                    except Exception:
                        pass

        task = asyncio.create_task(run(), name="kairocli-post-turn-snapshot")
        self._background.add(task)
        task.add_done_callback(self._finish_background_task)

    def _finish_background_task(self, task: asyncio.Task[None]) -> None:
        self._background.discard(task)
        try:
            task.exception()
        except (asyncio.CancelledError, Exception):
            pass

    async def wait_idle(self) -> None:
        while self._background:
            await asyncio.gather(*tuple(self._background), return_exceptions=True)

    async def list(self, limit: int | None = None) -> list[Snapshot]:
        await self.wait_idle()
        if not self.config.enabled or not (self.repository / "HEAD").exists():
            return []
        requested = self.config.max_snapshots if limit is None or limit <= 0 else limit
        maximum = min(max(requested, 1), MAX_SNAPSHOT_LIST_LIMIT)
        output = await self._git(
            "log",
            f"-{maximum}",
            "--format=%H%x00%cI%x00%s",
            env=self._environment(),
        )
        result: builtins.list[Snapshot] = []
        for line in output.splitlines():
            fields = line.split("\x00", 2)
            if len(fields) != 3:
                continue
            revision, created_at, message = fields
            phase, turn_id = _parse_subject(message)
            result.append(Snapshot(revision, message, created_at, phase, turn_id))
        return result

    async def list_pre_turn(self, limit: int | None = None) -> builtins.list[Snapshot]:
        maximum = self.config.max_snapshots if limit is None or limit <= 0 else limit
        snapshots = await self.list(max(maximum * 4, maximum))
        return [item for item in snapshots if item.phase == "pre-turn"][:maximum]

    async def restore_pre_turn(self, offset: int) -> RestoreResult:
        normalized = max(1, offset)
        candidates = await self.list_pre_turn(max(normalized, self.config.max_snapshots))
        if len(candidates) < normalized:
            return RestoreResult(False, None, f"Snapshot {normalized} was not found.")
        return await self.restore(candidates[normalized - 1].revision)

    async def restore(self, revision: str) -> RestoreResult:
        self._ensure_open()
        if not self.config.enabled:
            return RestoreResult(False, None, "Snapshots are disabled.")
        if not re.fullmatch(r"[0-9a-fA-F]{7,64}", revision):
            raise SnapshotError("Invalid snapshot revision")
        await self.wait_idle()
        async with self._lock:
            self._ensure_open()
            await self.initialize()
            env = self._environment(work_tree=True)
            target = (
                await self._git("rev-parse", "--verify", f"{revision}^{{commit}}", env=env)
            ).strip()
            current_revision = await self._capture_locked(
                f"pre-restore restore-{int(datetime.now(UTC).timestamp() * 1000)}\n\n"
                f"Before restoring {target[:10]}"
            )
            current_files: set[str] = set(await self._tree_files(current_revision, env))
            target_files: set[str] = set(await self._tree_files(target, env))
            try:
                removed = await self._checkout_tree(target, current_files, target_files, env)
            except BaseException as restore_error:
                try:
                    await asyncio.shield(
                        self._checkout_tree(current_revision, target_files, current_files, env)
                    )
                except BaseException as rollback_error:
                    raise SnapshotError(
                        "Snapshot restore failed and automatic rollback also failed: "
                        f"restore={safe_text(restore_error)}; "
                        f"rollback={safe_text(rollback_error)}"
                    ) from restore_error
                raise
            restored = tuple(sorted(item for item in target_files if not self._excluded(item)))
            return RestoreResult(
                True,
                target,
                f"Restored snapshot {target[:10]}.",
                restored,
                tuple(removed),
            )

    async def _checkout_tree(
        self,
        revision: str,
        source_files: set[str],
        target_files: set[str],
        env: dict[str, str],
    ) -> builtins.list[str]:
        removed: builtins.list[str] = []
        for relative in sorted(source_files - target_files):
            if self._excluded(relative):
                continue
            candidate = self._unlink_candidate(relative)
            if candidate.is_file() or candidate.is_symlink():
                await asyncio.to_thread(candidate.unlink)
                removed.append(relative)
                await asyncio.to_thread(
                    _prune_empty_parents, candidate.parent, self.paths.workspace
                )
        checkout_paths = sorted(
            relative for relative in target_files if not self._excluded(relative)
        )
        for batch in _literal_checkout_batches(checkout_paths):
            await self._git("checkout", revision, "--", *batch, env=env)
        return removed

    def _unlink_candidate(self, relative: str) -> Path:
        path = Path(relative)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise SnapshotError(f"Unsafe snapshot path: {relative!r}")
        try:
            parent = self.guard.resolve_for_write(path.parent)
        except (OSError, ValueError) as exc:
            raise SnapshotError(safe_text(exc)) from exc
        return parent / path.name

    async def status(self) -> str:
        latest = (await self.list(1)) if self.config.enabled else []
        recent = (
            f"{latest[0].phase} {latest[0].short_revision} {latest[0].created_at}"
            if latest
            else "none"
        )
        return (
            "Side-Git snapshot status\n"
            f"Enabled: {self.config.enabled}\n"
            f"Workspace: {self.paths.workspace}\n"
            f"Repository: {self.repository}\n"
            f"Maximum listed: {self.config.max_snapshots}\n"
            f"Excludes: {', '.join(self.config.excludes)}\n"
            f"Latest: {recent}"
        )

    async def clean(self) -> bool:
        self._ensure_open()
        await self.wait_idle()
        async with self._lock:
            self._ensure_open()
            if not self.repository.exists():
                return False
            self._validate_repository_path()
            await asyncio.to_thread(shutil.rmtree, self.repository)
            return True

    async def close(self) -> None:
        self._closing = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._close_once(), name="kairocli-snapshot-shutdown"
            )
        await _await_snapshot_shutdown(self._close_task)

    async def _close_once(self) -> None:
        active = tuple(self._background)
        if not active:
            return
        done, pending = await asyncio.wait(active, timeout=SNAPSHOT_SHUTDOWN_GRACE_SECONDS)
        if done:
            await asyncio.gather(*done, return_exceptions=True)
        for task in pending:
            task.cancel()
        if pending:
            done_after_cancel, _ = await asyncio.wait(
                pending, timeout=SNAPSHOT_CANCEL_GRACE_SECONDS
            )
            if done_after_cancel:
                await asyncio.gather(*done_after_cancel, return_exceptions=True)

    def _ensure_open(self) -> None:
        if self._closing:
            raise SnapshotError("Snapshot service is closed")

    async def _tree_files(self, revision: str, env: dict[str, str]) -> builtins.list[str]:
        output = await self._git("ls-tree", "-rz", "--name-only", revision, env=env)
        return [item for item in output.split("\x00") if item]

    def _environment(self, work_tree: bool = False) -> dict[str, str]:
        env = os.environ | {"GIT_DIR": str(self.repository)}
        if work_tree:
            env["GIT_WORK_TREE"] = str(self.paths.workspace)
        return env

    def _excluded(self, relative: str) -> bool:
        normalized = relative.replace("\\", "/")
        for raw in self.config.excludes:
            pattern = raw.strip().replace("\\", "/").strip("/")
            if not pattern:
                continue
            if "*" not in pattern and (
                normalized == pattern or normalized.startswith(f"{pattern}/")
            ):
                return True
            if pattern.startswith("*.") and normalized.rsplit("/", 1)[-1].endswith(pattern[1:]):
                return True
        return False

    def _validate_repository_path(self) -> None:
        try:
            reject_symlink_components(self.repository, "Snapshot repository")
        except ValueError as exc:
            raise SnapshotError(safe_text(exc)) from exc

    async def _git(self, *args: str, env: dict[str, str] | None = None) -> str:
        self._validate_repository_path()
        try:
            process = await asyncio.create_subprocess_exec(
                "git",
                *args,
                cwd=self.paths.workspace,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name == "posix",
            )
        except OSError as exc:
            raise SnapshotError("Unable to start Git: " + safe_text(exc)) from exc
        stdout_reader = asyncio.create_task(
            _read_bounded_process_stream(process.stdout, MAX_GIT_STDOUT_BYTES)
        )
        stderr_reader = asyncio.create_task(
            _read_bounded_process_stream(process.stderr, MAX_GIT_STDERR_BYTES)
        )
        try:
            await asyncio.wait_for(process.wait(), GIT_TIMEOUT_SECONDS)
        except TimeoutError as exc:
            await _terminate_git_process(process)
            await asyncio.gather(stdout_reader, stderr_reader, return_exceptions=True)
            raise SnapshotError(
                f"Git snapshot operation timed out after {GIT_TIMEOUT_SECONDS} seconds"
            ) from exc
        except asyncio.CancelledError:
            await _terminate_git_process(process)
            stdout_reader.cancel()
            stderr_reader.cancel()
            await asyncio.gather(stdout_reader, stderr_reader, return_exceptions=True)
            raise
        stdout, stdout_exceeded = await stdout_reader
        stderr, stderr_exceeded = await stderr_reader
        if process.returncode:
            detail = stderr.decode(errors="replace").strip() or "Git operation failed"
            if stderr_exceeded:
                detail += f"\n...[Git stderr truncated at {MAX_GIT_STDERR_BYTES} bytes]"
            raise SnapshotError(detail)
        if stdout_exceeded:
            raise SnapshotError(
                f"Git snapshot output exceeds {MAX_GIT_STDOUT_BYTES} bytes; "
                "reduce the workspace or snapshot excludes"
            )
        return stdout.decode(errors="surrogateescape")


def _parse_subject(message: str) -> tuple[str, str]:
    phase, _, turn_id = message.partition(" ")
    return (phase if phase in VALID_PHASES else "post-turn", turn_id.strip())


def _safe_commit_message(message: str) -> str:
    normalized = message.replace("\x00", "�").strip() or "post-turn turn"
    if len(normalized) > MAX_SNAPSHOT_MESSAGE_CHARS:
        return normalized[:MAX_SNAPSHOT_MESSAGE_CHARS] + "..."
    return normalized


def turn_snapshot_messages(mode: str, user_input: str) -> tuple[str, str]:
    safe_mode = re.sub(r"[^a-z0-9_-]", "-", mode.casefold()).strip("-") or "turn"
    normalized = re.sub(r"\s+", " ", user_input).strip()
    if len(normalized) > 120:
        normalized = f"{normalized[:120]}..."
    turn_id = f"{safe_mode}-{time.time_ns()}"
    summary = f"mode={safe_mode}\ninput={normalized}"
    return f"pre-turn {turn_id}\n\n{summary}", f"post-turn {turn_id}\n\n{summary}"


def _side_git_path(root: Path, workspace: Path) -> Path:
    parent_digest = hashlib.sha256(str(workspace.parent).encode()).hexdigest()[:16]
    workspace_digest = hashlib.sha256(str(workspace).encode()).hexdigest()[:16]
    return root / parent_digest / workspace_digest / ".git"


def _literal_checkout_batches(paths: builtins.list[str]) -> builtins.list[tuple[str, ...]]:
    batches: builtins.list[tuple[str, ...]] = []
    current: builtins.list[str] = []
    current_bytes = 0
    for path in paths:
        pathspec = f":(literal){path}"
        path_bytes = len(pathspec.encode("utf-8", errors="surrogateescape")) + 1
        if current and (
            len(current) >= _CHECKOUT_BATCH_PATHS
            or current_bytes + path_bytes > _CHECKOUT_BATCH_BYTES
        ):
            batches.append(tuple(current))
            current = []
            current_bytes = 0
        current.append(pathspec)
        current_bytes += path_bytes
    if current:
        batches.append(tuple(current))
    return batches


def _read_bool(name: str, fallback: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return fallback
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return fallback


def _prune_empty_parents(directory: Path, workspace: Path) -> None:
    current = directory
    while current != workspace and current.is_relative_to(workspace):
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


async def _read_bounded_process_stream(
    stream: asyncio.StreamReader | None, maximum: int
) -> tuple[bytes, bool]:
    if stream is None:
        return b"", False
    chunks: list[bytes] = []
    retained = 0
    exceeded = False
    while True:
        chunk = await stream.read(_PROCESS_READ_CHUNK_BYTES)
        if not chunk:
            break
        remaining = maximum - retained
        if remaining > 0:
            kept = chunk[:remaining]
            chunks.append(kept)
            retained += len(kept)
        if len(chunk) > remaining:
            exceeded = True
    return b"".join(chunks), exceeded


async def _terminate_git_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), 1)
        return
    except TimeoutError:
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        return
    await process.wait()


async def _await_snapshot_shutdown(task: asyncio.Task[None]) -> None:
    canceled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            canceled = True
    task.result()
    if canceled:
        raise asyncio.CancelledError
