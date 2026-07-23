from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from ...channels.wechat.formatting import (
    format_wechat_text as format_wechat_text,
)
from ...channels.wechat.formatting import (
    split_message as split_message,
)
from ...paths import KairoPaths
from .accounts import WechatAccountStore, _reject_wechat_symlinks

DEFAULT_BASE_URL = "https://ilinkai.weixin.qq.com"
MAX_WECHAT_ACCOUNT_BYTES = 128 * 1024
MAX_WECHAT_REQUEST_BYTES = 1024 * 1024
MAX_WECHAT_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_WECHAT_API_JSON_DEPTH = 32
MAX_WECHAT_API_JSON_NODES = 200_000
MAX_WECHAT_MESSAGE_CHARS = 100_000
MAX_WECHAT_MESSAGES_PER_UPDATE = 1_000
MAX_WECHAT_DAEMON_LOG_BYTES = 5 * 1024 * 1024
MAX_WECHAT_ACCOUNT_JSON_DEPTH = 16
MAX_WECHAT_ACCOUNT_JSON_NODES = 10_000
_WECHAT_ACCOUNT_THREAD_LOCK = threading.RLock()


def daemon_paths(paths: KairoPaths) -> tuple[Path, Path, Path]:
    root = paths.user_dir / "wechat"
    return root / "daemon.pid", root / "logs" / "stdout.log", root / "logs" / "stderr.log"


def daemon_command(paths: KairoPaths, action: str) -> str:
    pid_file, stdout_file, stderr_file = daemon_paths(paths)
    _reject_wechat_symlinks(
        pid_file.parent.parent,
        pid_file.parent,
        stdout_file.parent,
        pid_file,
        stdout_file,
        stderr_file,
    )
    if action == "start":
        if _read_live_pid(pid_file):
            return "WeChat daemon is already running."
        if WechatAccountStore(paths).load() is None:
            raise RuntimeError("No WeChat account is bound; run `kairocli wechat setup` first")
        stdout_file.parent.mkdir(parents=True, exist_ok=True)
        _reject_wechat_symlinks(pid_file.parent, stdout_file.parent)
        _secure_wechat_directory(pid_file.parent)
        _secure_wechat_directory(stdout_file.parent)
        _rotate_daemon_log(stdout_file)
        _rotate_daemon_log(stderr_file)
        options: dict[str, Any] = {}
        if os.name == "posix":
            options["start_new_session"] = True
        elif os.name == "nt":
            options["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        with (
            stdout_file.open("a", encoding="utf-8") as stdout,
            stderr_file.open("a", encoding="utf-8") as stderr,
        ):
            process = subprocess.Popen(
                [sys.executable, "-m", "kairocli", "wechat", "start"],
                cwd=paths.workspace,
                stdout=stdout,
                stderr=stderr,
                **options,
            )
        _secure_wechat_file(stdout_file)
        _secure_wechat_file(stderr_file)
        try:
            _write_daemon_pid(pid_file, process.pid)
        except BaseException:
            _stop_daemon_process(process.pid)
            raise
        return f"WeChat daemon started (PID {process.pid}). Logs: {stdout_file}"
    if action == "stop":
        pid = _read_live_pid(pid_file)
        if pid:
            _stop_daemon_process(pid)
        pid_file.unlink(missing_ok=True)
        return "WeChat daemon stopped."
    if action == "restart":
        daemon_command(paths, "stop")
        return daemon_command(paths, "start")
    if action == "logs":
        if not stdout_file.is_file():
            return "No WeChat daemon logs."
        return _tail_daemon_log(stdout_file)
    pid = _read_live_pid(pid_file)
    return f"WeChat daemon running (PID {pid})." if pid else "WeChat daemon is not running."


def _read_live_pid(pid_file: Path) -> int | None:
    if pid_file.is_symlink() or not pid_file.is_file():
        return None
    try:
        if pid_file.stat().st_size > 64:
            return None
        with pid_file.open("rb") as handle:
            encoded = handle.read(65)
        if len(encoded) > 64:
            return None
        pid = int(encoded.decode("utf-8").strip())
        if pid <= 1:
            return None
        os.kill(pid, 0)
        if os.name == "posix" and not _is_wechat_daemon_process(pid):
            return None
        return pid
    except (OSError, ValueError):
        return None


def _is_wechat_daemon_process(pid: int) -> bool:
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    command = result.stdout.casefold()
    return result.returncode == 0 and "kairocli" in command and "wechat" in command


def _stop_daemon_process(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _write_daemon_pid(pid_file: Path, pid: int) -> None:
    if pid_file.is_symlink():
        raise ValueError("WeChat daemon PID file cannot be a symlink")
    descriptor, temporary = tempfile.mkstemp(prefix=".daemon-", dir=pid_file.parent)
    try:
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(str(pid))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, pid_file)
        _secure_wechat_file(pid_file)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(temporary).unlink(missing_ok=True)
        raise


def _rotate_daemon_log(path: Path, keep: int = 3) -> None:
    if path.is_symlink():
        raise ValueError("WeChat daemon log cannot be a symlink")
    if not path.is_file() or path.stat().st_size <= MAX_WECHAT_DAEMON_LOG_BYTES:
        return
    for index in range(keep, 0, -1):
        source = path if index == 1 else path.with_name(f"{path.name}.{index - 1}")
        target = path.with_name(f"{path.name}.{index}")
        if source.exists() and not source.is_symlink():
            os.replace(source, target)
            _secure_wechat_file(target)


def _tail_daemon_log(path: Path, max_bytes: int = 64 * 1024) -> str:
    if path.is_symlink():
        raise ValueError("WeChat daemon log cannot be a symlink")
    with path.open("rb") as handle:
        size = path.stat().st_size
        handle.seek(max(0, size - max_bytes))
        data = handle.read(max_bytes)
    lines = data.decode("utf-8", errors="replace").splitlines()
    return "\n".join(lines[-100:])


def _secure_wechat_directory(path: Path) -> None:
    if os.name != "nt":
        path.chmod(0o700)


def _secure_wechat_file(path: Path) -> None:
    if os.name != "nt" and path.exists() and not path.is_symlink():
        path.chmod(0o600)
