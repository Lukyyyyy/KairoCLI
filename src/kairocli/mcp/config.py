from __future__ import annotations

import json
import logging
import os
import re
import secrets
import stat
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..paths import KairoPaths, reject_symlink_components
from ..text_safety import safe_text
from .constants import (
    _DEFAULT_CHROME_DEVTOOLS_MCP,
    _MCP_ENV_NAME,
    _MCP_HEADER_NAME,
    _MCP_SERVER_NAME,
    _MCP_STATE_THREAD_LOCK,
    MAX_MCP_CONFIG_BYTES,
    MAX_MCP_JSON_DEPTH,
    MAX_MCP_JSON_NODES,
    MAX_MCP_SERVERS,
    MAX_MCP_STATE_BYTES,
)
from .protocol import (
    format_tool_result as format_tool_result,
)

log = logging.getLogger(__name__)


@dataclass(slots=True)
class McpServerConfig:
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    url: str | None = None
    enabled: bool = True
    error: str | None = None


@dataclass(frozen=True, slots=True)
class McpConfigBootstrapResult:
    created: bool
    message: str = ""


def ensure_default_mcp_config(paths: KairoPaths) -> McpConfigBootstrapResult:
    file = paths.user_dir / "mcp.json"
    reject_symlink_components(file, "MCP config")
    if file.exists():
        try:
            if not file.is_file():
                raise ValueError(f"MCP config is not a regular file: {file}")
            encoded = _read_bounded_file(
                file,
                MAX_MCP_CONFIG_BYTES,
                f"MCP config exceeds the 1 MiB limit: {file}",
            )
        except OSError as exc:
            raise ValueError(f"Cannot read MCP config {file}: {type(exc).__name__}") from exc
        payload = _decode_mcp_json(encoded, file, "MCP config")
        servers = (
            payload.get("mcpServers", payload.get("servers", {}))
            if isinstance(payload, dict)
            else {}
        )
        if isinstance(servers, dict) and "chrome-devtools" in servers:
            return McpConfigBootstrapResult(False)
        return McpConfigBootstrapResult(
            False,
            f"Existing {file} does not configure chrome-devtools; "
            "see README for the browser MCP setup.",
        )

    paths.user_dir.mkdir(parents=True, exist_ok=True)
    reject_symlink_components(file, "MCP config")
    if os.name != "nt":
        paths.user_dir.chmod(0o700)
    encoded = (json.dumps(_DEFAULT_CHROME_DEVTOOLS_MCP, indent=2) + "\n").encode()
    temporary = paths.user_dir / f".mcp-bootstrap.{secrets.token_hex(8)}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, file)
        except FileExistsError:
            return ensure_default_mcp_config(paths)
        if os.name != "nt":
            file.chmod(0o600)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return McpConfigBootstrapResult(
        True,
        f"Created default MCP config: {file}\nchrome-devtools is enabled in isolated mode.",
    )


def load_mcp_config(paths: KairoPaths) -> dict[str, McpServerConfig]:
    merged: dict[str, dict[str, Any]] = {}
    for file in (paths.user_dir / "mcp.json", paths.project_dir / "mcp.json"):
        reject_symlink_components(file, "MCP config")
        if not file.is_file():
            continue
        try:
            encoded = _read_bounded_file(
                file,
                MAX_MCP_CONFIG_BYTES,
                f"MCP config exceeds the 1 MiB limit: {file}",
            )
        except OSError as exc:
            raise ValueError(f"Cannot read MCP config {file}: {type(exc).__name__}") from exc
        payload = _decode_mcp_json(encoded, file, "MCP config")
        if not isinstance(payload, dict):
            raise ValueError(f"MCP config root must be an object: {file}")
        servers = payload.get("mcpServers", payload.get("servers", {}))
        if not isinstance(servers, dict):
            raise ValueError(f"MCP servers must be an object: {file}")
        if os.name != "nt" and file == paths.user_dir / "mcp.json":
            paths.user_dir.chmod(0o700)
            file.chmod(0o600)
        for name, value in servers.items():
            if not isinstance(name, str) or not _MCP_SERVER_NAME.fullmatch(name):
                raise ValueError(f"Invalid MCP server name: {name!r}")
            if not isinstance(value, dict):
                raise ValueError(f"MCP server config must be an object: {name}")
            merged[name] = value
            if len(merged) > MAX_MCP_SERVERS:
                raise ValueError(f"MCP config exceeds {MAX_MCP_SERVERS} servers")
    variables = _mcp_variables(paths)
    configs: dict[str, McpServerConfig] = {}
    for name, value in merged.items():
        enabled = value.get("enabled", True)
        if not isinstance(enabled, bool):
            configs[name] = McpServerConfig(enabled=False, error="enabled must be a boolean")
            continue
        try:
            configs[name] = _prepare_mcp_server_config(value, variables, enabled)
        except ValueError as exc:
            configs[name] = McpServerConfig(enabled=enabled, error=safe_text(exc))
    return configs


def _load_mcp_state(paths: KairoPaths) -> dict[str, bool]:
    file = paths.user_dir / "mcp-state.json"
    reject_symlink_components(file, "MCP state")
    if not file.is_file():
        return {}
    try:
        encoded = _read_bounded_file(
            file,
            MAX_MCP_STATE_BYTES,
            f"MCP state exceeds the 128 KiB limit: {file}",
        )
    except OSError as exc:
        raise ValueError(f"Cannot read MCP state {file}: {type(exc).__name__}") from exc
    payload = _decode_mcp_json(encoded, file, "MCP state")
    if not isinstance(payload, dict):
        raise ValueError(f"MCP state root must be an object: {file}")
    if len(payload) > MAX_MCP_SERVERS:
        raise ValueError(f"MCP state exceeds {MAX_MCP_SERVERS} servers")
    state: dict[str, bool] = {}
    for name, enabled in payload.items():
        if not isinstance(name, str) or not _MCP_SERVER_NAME.fullmatch(name):
            raise ValueError(f"Invalid MCP state server name: {name!r}")
        if not isinstance(enabled, bool):
            raise ValueError(f"MCP state for {name} must be a boolean")
        state[name] = enabled
    if os.name != "nt":
        paths.user_dir.chmod(0o700)
        file.chmod(0o600)
    return state


def _save_mcp_state(paths: KairoPaths, state: dict[str, bool]) -> None:
    with _mcp_state_file_lock(paths):
        _publish_mcp_state(paths, state)


def _update_mcp_state(paths: KairoPaths, name: str, enabled: bool) -> None:
    if not _MCP_SERVER_NAME.fullmatch(name):
        raise ValueError(f"Invalid MCP state server name: {name!r}")
    if not isinstance(enabled, bool):
        raise ValueError(f"MCP state for {name} must be a boolean")
    with _mcp_state_file_lock(paths):
        state = _load_mcp_state(paths)
        state[name] = enabled
        _publish_mcp_state(paths, state)


def _publish_mcp_state(paths: KairoPaths, state: dict[str, bool]) -> None:
    directory = paths.user_dir
    file = directory / "mcp-state.json"
    reject_symlink_components(file, "MCP state")
    if len(state) > MAX_MCP_SERVERS:
        raise ValueError(f"MCP state exceeds {MAX_MCP_SERVERS} servers")
    for name, enabled in state.items():
        if not _MCP_SERVER_NAME.fullmatch(name):
            raise ValueError(f"Invalid MCP state server name: {name!r}")
        if not isinstance(enabled, bool):
            raise ValueError(f"MCP state for {name} must be a boolean")
    encoded = (json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    if len(encoded) > MAX_MCP_STATE_BYTES:
        raise ValueError("MCP state exceeds the 128 KiB limit")
    temporary = directory / f".mcp-state.{secrets.token_hex(8)}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, file)
        if os.name != "nt":
            file.chmod(0o600)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def _mcp_state_file_lock(paths: KairoPaths) -> Iterator[None]:
    directory = paths.user_dir
    reject_symlink_components(directory, "MCP state directory")
    directory.mkdir(parents=True, exist_ok=True)
    reject_symlink_components(directory, "MCP state directory")
    if os.name != "nt":
        directory.chmod(0o700)
    lock_file = directory / ".mcp-state.lock"
    reject_symlink_components(lock_file, "MCP state lock")
    descriptor = os.open(
        lock_file,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    locked = False
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"MCP state lock is not a regular file: {lock_file}")
        if os.name == "posix":
            os.fchmod(descriptor, 0o600)
        with _MCP_STATE_THREAD_LOCK:
            if os.name == "posix":
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX)
                locked = True
            elif os.name == "nt":  # pragma: no cover - exercised on Windows CI
                import msvcrt

                if os.fstat(descriptor).st_size == 0:
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


def _mcp_variables(paths: KairoPaths) -> dict[str, str]:
    variables = _read_dotenv(paths.home / ".env") | _read_dotenv(paths.workspace / ".env")
    variables.update(os.environ)
    variables["HOME"] = str(paths.home)
    variables["PROJECT_DIR"] = str(paths.workspace)
    return variables


def _read_dotenv(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        return {}
    try:
        encoded = _read_bounded_file(path, MAX_MCP_CONFIG_BYTES, "dotenv exceeds size limit")
        text = encoded.decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return {}
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        key = key.strip()
        if key:
            values[key] = value.strip().strip('"').strip("'")
    return values


def _read_bounded_file(path: Path, max_bytes: int, oversized_message: str) -> bytes:
    if path.stat().st_size > max_bytes:
        raise ValueError(oversized_message)
    with path.open("rb") as source:
        content = source.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError(oversized_message)
    return content


def _decode_mcp_json(encoded: bytes, file: Path, label: str) -> Any:
    try:
        payload = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_mcp_object_without_duplicates,
            parse_constant=_reject_mcp_json_constant,
        )
        _validate_mcp_json_shape(payload)
    except (OverflowError, RecursionError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Cannot read {label} {file}: {type(exc).__name__}") from exc
    return payload


def _mcp_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate MCP JSON key: {key}")
        result[key] = value
    return result


def _reject_mcp_json_constant(value: str) -> Any:
    raise ValueError(f"Non-standard JSON constant is not allowed: {value}")


def _validate_mcp_json_shape(value: Any) -> None:
    stack: list[tuple[Any, int]] = [(value, 0)]
    visited = 0
    while stack:
        current, depth = stack.pop()
        visited += 1
        if visited > MAX_MCP_JSON_NODES:
            raise ValueError("MCP JSON exceeds the node limit")
        if depth > MAX_MCP_JSON_DEPTH:
            raise ValueError("MCP JSON exceeds the nesting limit")
        children: Iterable[Any]
        if isinstance(current, dict):
            children = current.values()
        elif isinstance(current, list):
            children = current
        else:
            continue
        child_count = len(current)
        if visited + len(stack) + child_count > MAX_MCP_JSON_NODES:
            raise ValueError("MCP JSON exceeds the node limit")
        stack.extend((child, depth + 1) for child in children)


def _expand_env(value: str, variables: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        replacement = variables.get(name)
        if replacement is None or not replacement.strip():
            raise ValueError(f"MCP config references an unset environment variable: {name}")
        return replacement

    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)}", replace, value)


def _prepare_mcp_server_config(
    value: dict[str, Any], variables: dict[str, str], enabled: bool
) -> McpServerConfig:
    command = _optional_mcp_text(value.get("command"), "command", 4096)
    url = _optional_mcp_text(value.get("url"), "url", 8192)
    if bool(command) == bool(url):
        raise ValueError("exactly one of command or url is required")
    args = _mcp_text_list(value.get("args", []), "args", 100, 16_384)
    env = _mcp_text_map(value.get("env", {}), "env", _MCP_ENV_NAME, variables, 100, 65_536)
    headers = _mcp_text_map(
        value.get("headers", {}),
        "headers",
        _MCP_HEADER_NAME,
        variables,
        100,
        65_536,
    )
    expanded_command = _expand_env(command, variables) if command else None
    expanded_url = _expand_env(url, variables) if url else None
    expanded_args = [_expand_env(item, variables) for item in args]
    if expanded_command and len(expanded_command) > 4096:
        raise ValueError("expanded command is too long")
    if expanded_url and len(expanded_url) > 8192:
        raise ValueError("expanded url is too long")
    if any(len(item) > 16_384 or "\x00" in item for item in expanded_args):
        raise ValueError("expanded args contains an invalid value")
    if expanded_url:
        parsed = urlsplit(expanded_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("url must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("url cannot contain embedded credentials")
    return McpServerConfig(
        command=expanded_command,
        args=expanded_args,
        env=env,
        headers=headers,
        url=expanded_url,
        enabled=enabled,
    )


def _optional_mcp_text(value: Any, field_name: str, max_chars: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    stripped = value.strip()
    if not stripped:
        return None
    if len(stripped) > max_chars or "\x00" in stripped:
        raise ValueError(f"{field_name} is invalid or too long")
    return stripped


def _mcp_text_list(value: Any, field_name: str, max_items: int, max_chars: int) -> list[str]:
    if not isinstance(value, list) or len(value) > max_items:
        raise ValueError(f"{field_name} must be a list of at most {max_items} strings")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or len(item) > max_chars or "\x00" in item:
            raise ValueError(f"{field_name} contains an invalid value")
        result.append(item)
    return result


def _mcp_text_map(
    value: Any,
    field_name: str,
    key_pattern: re.Pattern[str],
    variables: dict[str, str],
    max_items: int,
    max_chars: int,
) -> dict[str, str]:
    if not isinstance(value, dict) or len(value) > max_items:
        raise ValueError(f"{field_name} must be an object with at most {max_items} entries")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key_pattern.fullmatch(key):
            raise ValueError(f"{field_name} contains an invalid key")
        if not isinstance(item, str) or len(item) > max_chars or "\x00" in item:
            raise ValueError(f"{field_name} contains an invalid value")
        expanded = _expand_env(item, variables)
        if len(expanded) > max_chars or "\x00" in expanded:
            raise ValueError(f"{field_name} contains an expanded value that is too long")
        if field_name == "headers" and ("\r" in expanded or "\n" in expanded):
            raise ValueError("headers cannot contain line breaks")
        result[key] = expanded
    return result
