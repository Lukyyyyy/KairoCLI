"""Private, bounded, secret-aware input history persistence."""

from __future__ import annotations

import os
import re
import stat
import time
from pathlib import Path

from ..paths import reject_symlink_components
from ..private_lock import private_file_lock
from ..trace import redact_sensitive_text

MAX_INPUT_HISTORY_READ_BYTES = 5 * 1024 * 1024
MAX_INPUT_HISTORY_FILE_BYTES = 5 * 1024 * 1024
MAX_INPUT_HISTORY_ENTRIES = 2_000
_INPUT_HISTORY_LOCK_FILE = ".input-history.lock"


def _prepare_input_history(path: Path) -> None:
    reject_symlink_components(path.parent, "Input history directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    reject_symlink_components(path, "Input history file")
    if path.exists() and not path.is_file():
        raise ValueError("Input history path must be a regular file")
    if os.name == "posix":
        path.parent.chmod(0o700)


def _open_input_history(path: Path, flags: int, mode: int = 0o600) -> int:
    reject_symlink_components(path, "Input history file")
    safe_flags = flags | getattr(os, "O_NOFOLLOW", 0)
    if os.name != "posix":  # pragma: no cover - exercised on Windows CI
        return os.open(path, safe_flags, mode)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(path.parent, directory_flags)
    try:
        opened = os.fstat(directory_fd)
        current = os.stat(path.parent, follow_symlinks=False)
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise ValueError("Input history directory changed during validation")
        reject_symlink_components(path, "Input history file")
        return os.open(path.name, safe_flags, mode, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)


def _load_input_history(path: Path) -> list[str]:
    try:
        with private_file_lock(path.parent, _INPUT_HISTORY_LOCK_FILE, "Input history"):
            return _load_input_history_unlocked(path)
    except (OSError, ValueError):
        return []


def _load_input_history_unlocked(path: Path) -> list[str]:
    try:
        descriptor = _open_input_history(path, os.O_RDONLY)
    except FileNotFoundError:
        return []
    except (OSError, ValueError):
        return []
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            return []
        start = max(0, metadata.st_size - MAX_INPUT_HISTORY_READ_BYTES)
        os.lseek(descriptor, start, os.SEEK_SET)
        remaining = MAX_INPUT_HISTORY_READ_BYTES
        chunks: list[bytes] = []
        while remaining > 0:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    except OSError:
        return []
    finally:
        os.close(descriptor)
    encoded = b"".join(chunks)
    if start:
        boundary = encoded.find(b"\n# ")
        if boundary < 0:
            return []
        encoded = encoded[boundary + 1 :]
    return _parse_input_history(encoded)


def _parse_input_history(encoded: bytes) -> list[str]:
    strings: list[str] = []
    lines: list[bytes] = []

    def add() -> None:
        if lines:
            raw_value = b"".join(lines)
            if raw_value.endswith(b"\n"):
                raw_value = raw_value[:-1]
            try:
                value = raw_value.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                return
            if not _is_sensitive_history_input(value):
                strings.append(value)

    for line in encoded.splitlines(keepends=True):
        if line.startswith(b"+"):
            lines.append(line[1:])
        else:
            add()
            lines = []
    add()
    return list(reversed(strings[-MAX_INPUT_HISTORY_ENTRIES:]))


def _append_input_history(path: Path, value: str) -> bool:
    if _is_sensitive_history_input(value):
        return False
    payload_text = f"\n# {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    payload_text += "".join(f"+{line}\n" for line in value.split("\n"))
    payload = payload_text.encode("utf-8")
    try:
        with private_file_lock(path.parent, _INPUT_HISTORY_LOCK_FILE, "Input history"):
            _prepare_input_history(path)
            descriptor = _open_input_history(path, os.O_RDWR | os.O_CREAT | os.O_APPEND)
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    return False
                if os.name == "posix":
                    os.fchmod(descriptor, 0o600)
                _trim_input_history_for_append(descriptor, metadata.st_size, len(payload))
                _write_input_history_bytes(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return True
    except (OSError, ValueError):
        return False


def _trim_input_history_for_append(descriptor: int, current_size: int, payload_size: int) -> None:
    if current_size + payload_size <= MAX_INPUT_HISTORY_FILE_BYTES:
        return
    keep_bytes = max(0, MAX_INPUT_HISTORY_FILE_BYTES - payload_size)
    start = max(0, current_size - keep_bytes)
    os.lseek(descriptor, start, os.SEEK_SET)
    remaining = min(current_size - start, keep_bytes)
    chunks: list[bytes] = []
    while remaining > 0:
        chunk = os.read(descriptor, min(remaining, 64 * 1024))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    retained = b"".join(chunks)
    if start:
        boundary = retained.find(b"\n# ")
        retained = b"" if boundary < 0 else retained[boundary + 1 :]
    os.ftruncate(descriptor, 0)
    os.lseek(descriptor, 0, os.SEEK_SET)
    _write_input_history_bytes(descriptor, retained)


def _write_input_history_bytes(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("Input history write made no progress")
        view = view[written:]


def _clear_input_history(path: Path) -> bool:
    try:
        with private_file_lock(path.parent, _INPUT_HISTORY_LOCK_FILE, "Input history"):
            return _clear_input_history_unlocked(path)
    except (OSError, ValueError):
        return False


def _clear_input_history_unlocked(path: Path) -> bool:
    try:
        reject_symlink_components(path, "Input history file")
        if os.name != "posix":  # pragma: no cover - exercised on Windows CI
            if path.exists() and not path.is_file():
                return False
            path.unlink(missing_ok=True)
            return True
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            directory_fd = os.open(path.parent, directory_flags)
        except FileNotFoundError:
            return True
        try:
            opened = os.fstat(directory_fd)
            current = os.stat(path.parent, follow_symlinks=False)
            if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
                return False
            reject_symlink_components(path, "Input history file")
            try:
                target = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                return True
            if not stat.S_ISREG(target.st_mode):
                return False
            os.unlink(path.name, dir_fd=directory_fd)
        finally:
            os.close(directory_fd)
        return True
    except (OSError, ValueError):
        return False


def _is_sensitive_history_input(value: str) -> bool:
    stripped = value.strip()
    if not stripped or len(stripped) > 8_000:
        return True
    if redact_sensitive_text(stripped) != stripped:
        return True
    patterns = (
        r"(?i)\b[\w-]*(?:api[_-]?key|token|secret|password|authorization)\b\s*[=:]",
        r"(?i)\bauthorization\s*:\s*bearer\b",
        r"(?i)--(?:api[-_]?key|access[-_]?token|token|secret|password|authorization)(?:=|\s+)",
        r"(?i)[?&](?:api[_-]?key|access[_-]?token|token|secret|password)=[^&\s]+",
        r"(?i)https?://[^/\s:@]+:[^/\s@]+@",
        r"(?i)-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        r"\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{16,}|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,})\b",
        r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
        r"(?i)data:image/|@image:data:",
        r"[A-Za-z0-9+/]{240,}={0,2}",
    )
    return any(re.search(pattern, stripped) for pattern in patterns)
