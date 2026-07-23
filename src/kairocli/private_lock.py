from __future__ import annotations

import os
import stat
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .paths import reject_symlink_components

_PRIVATE_FILE_THREAD_LOCK = threading.RLock()


@contextmanager
def private_file_lock(directory: Path, filename: str, label: str) -> Iterator[None]:
    reject_symlink_components(directory, label)
    directory.mkdir(parents=True, exist_ok=True)
    reject_symlink_components(directory, label)
    if os.name == "posix":
        directory.chmod(0o700)
    lock_file = directory / filename
    reject_symlink_components(lock_file, f"{label} lock")
    descriptor = os.open(
        lock_file,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    locked = False
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{label} lock must be a regular file")
        if os.name == "posix":
            os.fchmod(descriptor, 0o600)
        with _PRIVATE_FILE_THREAD_LOCK:
            if os.name == "posix":
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX)
                locked = True
            elif os.name == "nt":  # pragma: no cover - exercised on Windows CI
                import msvcrt

                if metadata.st_size == 0:
                    os.write(descriptor, b"\0")
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined]
                locked = True
            yield
    finally:
        if locked and os.name == "posix":
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_UN)
        elif locked and os.name == "nt":  # pragma: no cover - exercised on Windows CI
            import msvcrt

            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
        os.close(descriptor)
