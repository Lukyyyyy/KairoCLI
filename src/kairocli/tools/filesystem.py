from __future__ import annotations

import asyncio
import fnmatch
import html
import json
import logging
import os
import re
import secrets
import shutil
import stat
import tempfile
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from ..brand import PROJECT_DIR_NAME
from ..policy import (
    ApprovalResult,
    PolicyDenied,
)

ToolHandler = Callable[[dict[str, Any]], Awaitable[Any]]
ApprovalHandler = Callable[[str, dict[str, Any]], Awaitable[bool | ApprovalResult]]
_cancel_event: ContextVar[asyncio.Event | None] = ContextVar("tool_cancel_event", default=None)
log = logging.getLogger(__name__)
DEFAULT_READ_FILE_CHARS = 100_000
MAX_COMMAND_OUTPUT_BYTES = 50_000
MAX_GIT_APPLY_OUTPUT_BYTES = 50_000
DEFAULT_GREP_MAX_CHARS = 24_000
MAX_GREP_FILE_BYTES = 5 * 1024 * 1024
MAX_TOOL_OUTPUT_CHARS = 200_000
MAX_TOOL_ERROR_CHARS = 4_000
MAX_DISPLAY_DIFF_CHARS = 200_000
MAX_DISPLAY_DIFF_LINES = 2_000
MAX_DISPLAY_DIFF_FILES = 20
TOOL_CANCEL_GRACE_SECONDS = 0.1
SERIALIZED_WORKSPACE_MUTATION_TOOLS = frozenset(
    {
        "write_file",
        "apply_patch",
        "create_project",
        "revert_turn",
        "execute_command",
        "shell_exec",
    }
)


def _create_project_atomic(root: Path, kind: str, workspace: Path) -> dict[str, Any]:
    if kind not in {"python", "node", "java"}:
        raise ValueError("kind must be python, node, or java")
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise PolicyDenied("create_project target must not exist or must be empty")
    root.parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = root.parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(workspace.resolve())
    except ValueError as exc:
        raise PolicyDenied(f"Project parent escapes workspace: {root}") from exc

    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.kairocli-project-", dir=root.parent))
    existed_empty = root.exists()
    project_name = root.name
    created_files: list[str] = []
    created_directories: set[str] = set()

    def directory(relative: str) -> None:
        (staging / relative).mkdir(parents=True, exist_ok=True)
        current = Path(relative)
        while current.parts:
            created_directories.add(current.as_posix())
            current = current.parent
            if current == Path("."):
                break

    def text_file(relative: str, content: str) -> None:
        target = staging / relative
        _atomic_write_text(target, content, workspace)
        created_files.append(relative)
        parent = Path(relative).parent
        while parent != Path("."):
            created_directories.add(parent.as_posix())
            parent = parent.parent

    try:
        if kind == "python":
            package_name = re.sub(r"\W", "_", project_name)
            if not package_name or package_name[0].isdigit():
                package_name = f"_{package_name}"
            directory("src")
            directory("tests")
            directory(project_name)
            text_file(f"{project_name}/__init__.py", "")
            text_file("main.py", "# Main entry point\n")
            text_file("requirements.txt", "# Dependencies\n")
            text_file(
                "pyproject.toml",
                "[project]\n"
                f"name = {json.dumps(project_name)}\n"
                'version = "0.1.0"\n'
                'requires-python = ">=3.11"\n\n'
                "[tool.pytest.ini_options]\n"
                'testpaths = ["tests"]\n'
                f"# import package: {package_name}\n",
            )
        elif kind == "node":
            directory("src")
            text_file(
                "package.json",
                json.dumps(
                    {
                        "name": project_name.casefold(),
                        "version": "1.0.0",
                        "private": True,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
            )
        else:
            directory("src/main/java")
            directory("src/main/resources")
            directory("src/test/java")
            artifact = html.escape(project_name, quote=False)
            text_file(
                "pom.xml",
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                "<project>\n"
                "    <modelVersion>4.0.0</modelVersion>\n"
                "    <groupId>com.example</groupId>\n"
                f"    <artifactId>{artifact}</artifactId>\n"
                "    <version>1.0</version>\n"
                "</project>\n",
            )

        if os.name == "posix":
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            directory_fd = os.open(root.parent, flags)
            try:
                opened = os.fstat(directory_fd)
                current = os.stat(root.parent, follow_symlinks=False)
                if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
                    raise PolicyDenied("Project parent changed during creation")
                if existed_empty:
                    os.rmdir(root.name, dir_fd=directory_fd)
                try:
                    os.replace(
                        staging.name,
                        root.name,
                        src_dir_fd=directory_fd,
                        dst_dir_fd=directory_fd,
                    )
                except BaseException:
                    if existed_empty:
                        try:
                            os.mkdir(root.name, 0o755, dir_fd=directory_fd)
                        except FileExistsError:
                            pass
                    raise
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        else:  # pragma: no cover - exercised on Windows CI
            if existed_empty:
                root.rmdir()
            try:
                os.replace(staging, root)
            except BaseException:
                if existed_empty:
                    root.mkdir(exist_ok=True)
                raise
        return {
            "path": str(root.relative_to(workspace)),
            "kind": kind,
            "files": sorted(created_files),
            "directories": sorted(created_directories),
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _atomic_write_text(path: Path, content: str, workspace: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = path.parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(workspace.resolve())
    except ValueError as exc:
        raise PolicyDenied(f"Write path parent escapes workspace: {path}") from exc
    encoded = content.encode("utf-8")
    temporary_name = f".{path.name}.{secrets.token_hex(8)}.tmp"
    directory_fd: int | None = None
    temporary_path = path.parent / temporary_name
    try:
        if os.name == "posix":
            directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            directory_flags |= getattr(os, "O_NOFOLLOW", 0)
            directory_fd = os.open(path.parent, directory_flags)
            opened = os.fstat(directory_fd)
            current = os.stat(path.parent, follow_symlinks=False)
            if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
                raise PolicyDenied("Write path parent changed during validation")
            try:
                path.parent.resolve(strict=True).relative_to(workspace.resolve())
            except ValueError as exc:
                raise PolicyDenied("Write path parent changed outside workspace") from exc
            try:
                target = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                mode = 0o644
            else:
                if stat.S_ISLNK(target.st_mode):
                    raise PolicyDenied("write_file refuses a symlink target")
                if not stat.S_ISREG(target.st_mode):
                    raise ValueError("write_file target is not a regular file")
                mode = stat.S_IMODE(target.st_mode)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(temporary_name, flags, 0o600, dir_fd=directory_fd)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary_name, mode, dir_fd=directory_fd)
            os.replace(
                temporary_name,
                path.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            os.fsync(directory_fd)
            return

        if path.is_symlink():  # pragma: no cover - exercised on Windows CI
            raise PolicyDenied("write_file refuses a symlink target")
        mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
        descriptor = os.open(temporary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    finally:
        if directory_fd is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            os.close(directory_fd)
        else:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _read_file_range(path: Path, start: int, limit: int, max_chars: int) -> str:
    with path.open("rb") as raw:
        sample = raw.read(8_192)
    if b"\x00" in sample:
        raise ValueError("read_file refuses binary files; use a binary-aware tool")

    output: list[str] = []
    rendered_chars = 0
    selected = 0
    partial = False
    next_offset = start
    replacement_seen = False
    saw_line = False
    chunk_chars = max_chars + 1
    with path.open("r", encoding="utf-8", errors="replace", newline=None) as stream:
        line_number = 0
        while True:
            raw_line = stream.readline(chunk_chars)
            if not raw_line:
                break
            line_number += 1
            if "\x00" in raw_line:
                raise ValueError("read_file refuses binary files; use a binary-aware tool")
            line_is_fragmented = len(raw_line) == chunk_chars and not raw_line.endswith("\n")
            if line_number < start:
                while line_is_fragmented:
                    raw_line = stream.readline(chunk_chars)
                    if not raw_line:
                        break
                    if "\x00" in raw_line:
                        raise ValueError("read_file refuses binary files; use a binary-aware tool")
                    line_is_fragmented = len(raw_line) == chunk_chars and not raw_line.endswith(
                        "\n"
                    )
                continue
            saw_line = True
            if selected >= limit:
                partial = True
                next_offset = line_number
                break
            line = raw_line.rstrip("\r\n")
            replacement_seen = replacement_seen or "\ufffd" in line
            rendered = f"{line_number}: {line}"
            separator_chars = 1 if output else 0
            remaining = max_chars - rendered_chars - separator_chars
            if line_is_fragmented or len(rendered) > remaining:
                if remaining > 0:
                    output.append(rendered[:remaining])
                    rendered_chars += separator_chars + remaining
                partial = True
                next_offset = line_number + 1
                break
            output.append(rendered)
            rendered_chars += separator_chars + len(rendered)
            selected += 1

    if not saw_line:
        return f"[read_file: offset {start} is beyond end of file]"
    notices: list[str] = []
    if partial:
        notices.append(
            f"[partial: true; next_offset={next_offset}; limits: lines={limit}, chars={max_chars}]"
        )
    if replacement_seen:
        notices.append("[encoding_warning: invalid UTF-8 bytes were replaced]")
    return "\n".join([*output, *notices])


def _glob_paths(workspace: Path, root: Path, pattern: str, limit: int) -> list[str]:
    ignored = {
        ".git",
        ".venv",
        PROJECT_DIR_NAME,
        "node_modules",
        "target",
        "dist",
        "build",
        "__pycache__",
    }
    matches: list[str] = []
    for current, directories, files in os.walk(root, followlinks=False):
        directories[:] = sorted(item for item in directories if item not in ignored)
        current_path = Path(current)
        for file_name in sorted(files):
            path = current_path / file_name
            relative_root = path.relative_to(root).as_posix()
            patterns = (pattern, pattern[3:]) if pattern.startswith("**/") else (pattern,)
            if any(
                fnmatch.fnmatch(relative_root, candidate) or fnmatch.fnmatch(file_name, candidate)
                for candidate in patterns
            ):
                matches.append(path.relative_to(workspace).as_posix())
                if len(matches) >= limit:
                    return matches
    return matches


def _take_text_budget(lines: list[str], max_results: int, max_chars: int) -> tuple[list[str], bool]:
    result: list[str] = []
    used = 0
    for line in lines[:max_results]:
        required = len(line) + (1 if result else 0)
        if used + required > max_chars:
            remaining = max_chars - used - (1 if result else 0)
            if remaining > 0:
                result.append(line[:remaining])
            return result, True
        result.append(line)
        used += required
    return result, False


def _suggested_reads(matches: list[str]) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for match in matches:
        parsed = re.match(r"^(.*?):(\d+)(?::|-)", match)
        if parsed is None:
            continue
        path, line_text = parsed.groups()
        if path in seen:
            continue
        seen.add(path)
        suggestions.append({"path": path, "offset": max(1, int(line_text) - 20), "limit": 80})
        if len(suggestions) >= 3:
            break
    return suggestions
