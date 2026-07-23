import logging
import os
from pathlib import Path

from kairocli.diagnostics import (
    PrivateApplicationLogHandler,
    configure_application_logging,
    shutdown_application_logging,
)
from kairocli.paths import KairoPaths


def test_application_log_is_private_redacted_and_rotated(tmp_path: Path) -> None:
    directory = tmp_path / "logs"
    handler = PrivateApplicationLogHandler(directory, max_bytes=180, max_files=3)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("kairocli-test-private-log")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)

    logger.info("request token=top-secret-value\nsecond line")
    logger.info("x" * 160)
    logger.info("y" * 160)

    files = sorted(directory.glob("kairocli-*.log"))
    assert 2 <= len(files) <= 3
    content = "".join(path.read_text(encoding="utf-8") for path in files)
    assert "top-secret-value" not in content
    assert "token=***" in content
    assert "\\nsecond line" in content
    if os.name == "posix":
        assert directory.stat().st_mode & 0o777 == 0o700
        assert all(path.stat().st_mode & 0o777 == 0o600 for path in files)


def test_application_log_refuses_symlink_container(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "logs"
    linked.symlink_to(outside, target_is_directory=True)
    handler = PrivateApplicationLogHandler(linked, max_bytes=1024, max_files=2)
    record = logging.LogRecord("kairocli.test", logging.INFO, "", 0, "hello", (), None)

    handler.emit(record)

    assert list(outside.iterdir()) == []


def test_application_log_preserves_format_types_and_excludes_third_party(
    tmp_path: Path,
) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    shutdown_application_logging()
    try:
        configure_application_logging(paths)
        logging.getLogger("httpx").info(
            "GET https://example.test/?X-Amz-Signature=highly-secret-signature"
        )
        logging.getLogger("kairocli.test").info("duration_ms=%d", 123)
    finally:
        shutdown_application_logging()

    content = "".join(
        path.read_text(encoding="utf-8")
        for path in (paths.user_dir / "logs").glob("kairocli-*.log")
    )
    assert "duration_ms=123" in content
    assert "highly-secret-signature" not in content
    assert "example.test" not in content
