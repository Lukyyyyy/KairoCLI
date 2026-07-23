from __future__ import annotations

import atexit
import logging
import logging.handlers
import os
import queue
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .paths import KairoPaths, reject_symlink_components
from .private_lock import private_file_lock
from .text_safety import safe_text
from .trace import safe_redacted_text

DEFAULT_LOG_FILE_BYTES = 10 * 1024 * 1024
DEFAULT_LOG_FILES = 7
MAX_LOG_RECORD_BYTES = 64 * 1024
_HANDLER_MARKER = "_kairocli_application_handler"
_LOGGER_NAME = "kairocli"
_LOG_QUEUE_SIZE = 2048
_queue_handler: logging.Handler | None = None
_queue_listener: _DrainQueueListener | None = None
_configuration_lock = threading.Lock()
_atexit_registered = False


class _BoundedQueueHandler(logging.handlers.QueueHandler):
    """Never block the application when diagnostics cannot keep up."""

    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            return


class _DrainQueueListener(logging.handlers.QueueListener):
    """Guarantee orderly shutdown even when the bounded queue is full."""

    def enqueue_sentinel(self) -> None:
        # QueueListener's private protocol uses this sentinel; its types expose
        # only the non-blocking queue surface even though queue.Queue is supplied.
        self.queue.put(self._sentinel)  # type: ignore[attr-defined]


class PrivateApplicationLogHandler(logging.Handler):
    """Size-rotated private application log that never follows symlinks."""

    def __init__(self, directory: Path, max_bytes: int, max_files: int) -> None:
        super().__init__()
        self.directory = directory
        self.max_bytes = max_bytes
        self.max_files = max_files
        self._emit_lock = threading.RLock()
        setattr(self, _HANDLER_MARKER, True)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = (
                safe_redacted_text(
                    _safe_record_message(record),
                    MAX_LOG_RECORD_BYTES,
                    "...[application log record truncated]",
                )
                .replace("\r", "\\r")
                .replace("\n", "\\n")
            )
            encoded = (
                f"{datetime.now(UTC).isoformat()} {record.levelname} {record.name} {message}\n"
            ).encode("utf-8", errors="replace")
            with (
                self._emit_lock,
                private_file_lock(
                    self.directory, ".application-log.lock", "Kairo CLI application log"
                ),
            ):
                target = self._target(len(encoded))
                reject_symlink_components(target, "Kairo CLI application log")
                flags = os.O_CREAT | os.O_APPEND | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(target, flags, 0o600)
                with os.fdopen(descriptor, "ab") as stream:
                    if os.name == "posix":
                        os.fchmod(stream.fileno(), 0o600)
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                self._prune()
        except (OSError, RecursionError, TypeError, UnicodeError, ValueError):
            # Diagnostics must not alter product behavior or recurse through logging.
            return

    def _target(self, incoming_bytes: int) -> Path:
        stem = f"kairocli-{datetime.now(UTC).date().isoformat()}"
        for suffix in range(self.max_files):
            target = self.directory / f"{stem}{f'-{suffix}' if suffix else ''}.log"
            reject_symlink_components(target, "Kairo CLI application log")
            try:
                size = target.stat().st_size
            except OSError:
                size = 0
            if size + incoming_bytes <= self.max_bytes:
                return target
        return self.directory / f"{stem}-{uuid.uuid4().hex[:8]}.log"

    def _prune(self) -> None:
        candidates: list[tuple[float, Path]] = []
        for path in self.directory.glob("kairocli-*.log"):
            try:
                reject_symlink_components(path, "Kairo CLI application log")
                candidates.append((path.stat().st_mtime, path))
            except (OSError, ValueError):
                continue
        for _, stale in sorted(candidates, reverse=True)[self.max_files :]:
            try:
                stale.unlink()
            except OSError:
                continue


def configure_application_logging(paths: KairoPaths) -> Path | None:
    """Configure one process-wide Kairo-only asynchronous log; return its directory."""
    global _atexit_registered, _queue_handler, _queue_listener
    if not _env_bool("KAIROCLI_LOG_ENABLED", True):
        return None
    with _configuration_lock:
        if _queue_handler is not None:
            return paths.user_dir / "logs"
        level_name = os.getenv("KAIROCLI_LOG_LEVEL", "INFO").strip().upper()
        level = getattr(logging, level_name, logging.INFO)
        max_bytes = _env_int("KAIROCLI_LOG_MAX_BYTES", DEFAULT_LOG_FILE_BYTES, 64 * 1024, 1024**3)
        max_files = _env_int("KAIROCLI_LOG_MAX_FILES", DEFAULT_LOG_FILES, 1, 100)
        sink = PrivateApplicationLogHandler(paths.user_dir / "logs", max_bytes, max_files)
        sink.setLevel(level)
        sink.setFormatter(logging.Formatter("%(message)s"))
        records: queue.Queue[logging.LogRecord] = queue.Queue(maxsize=_LOG_QUEUE_SIZE)
        handler = _BoundedQueueHandler(records)
        handler.setLevel(level)
        setattr(handler, _HANDLER_MARKER, True)
        logger = logging.getLogger(_LOGGER_NAME)
        logger.addHandler(handler)
        logger.setLevel(min(logger.level, level) if logger.level else level)
        logger.propagate = False
        listener = _DrainQueueListener(records, sink, respect_handler_level=True)
        listener.start()
        _queue_handler = handler
        _queue_listener = listener
        if not _atexit_registered:
            atexit.register(shutdown_application_logging)
            _atexit_registered = True
    logging.getLogger(__name__).info("application_logging_configured level=%s", level_name)
    return sink.directory


def shutdown_application_logging() -> None:
    """Drain and close the diagnostics worker, primarily for orderly process exit."""
    global _queue_handler, _queue_listener
    with _configuration_lock:
        handler, listener = _queue_handler, _queue_listener
        _queue_handler = None
        _queue_listener = None
        if handler is not None:
            logging.getLogger(_LOGGER_NAME).removeHandler(handler)
    if listener is not None:
        listener.stop()


def _safe_record_message(record: logging.LogRecord) -> str:
    template = safe_text(record.msg, fallback=f"{type(record.msg).__name__} message")
    if record.args:
        if isinstance(record.args, dict):
            arguments: object = {
                safe_text(key): _safe_log_argument(value) for key, value in record.args.items()
            }
        else:
            arguments = tuple(_safe_log_argument(value) for value in record.args)
        try:
            template = template % arguments
        except (KeyError, TypeError, ValueError):
            template += f" args={safe_text(arguments)}"
    if record.exc_info and record.exc_info[1] is not None:
        error = record.exc_info[1]
        template += f" exception={type(error).__name__}: {safe_text(error)}"
    return template


def _safe_log_argument(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return safe_text(value)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return min(max(value, minimum), maximum)
