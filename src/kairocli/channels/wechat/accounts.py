from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from ...channels.wechat.formatting import (
    format_wechat_text as format_wechat_text,
)
from ...channels.wechat.formatting import (
    split_message as split_message,
)
from ...paths import KairoPaths

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


def _reject_wechat_symlinks(*paths: Path) -> None:
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"WeChat state cannot use a symlink: {path.name}")


@dataclass(slots=True)
class WechatAccount:
    token: str
    account_id: str
    base_url: str
    bound_user_id: str
    workspace: str
    sync_buf: str = ""
    created_at: str = ""


@dataclass(slots=True)
class QrLogin:
    qrcode_id: str
    qrcode_url: str


@dataclass(slots=True)
class LoginResult:
    connected: bool
    expired: bool
    status: str
    token: str = ""
    account_id: str = ""
    base_url: str = DEFAULT_BASE_URL
    user_id: str = ""
    message: str = ""


@dataclass(slots=True)
class WechatMediaItem:
    type: str
    file_name: str = ""
    mime_type: str = ""
    encrypt_query_param: str = ""
    aes_key: str = ""

    @property
    def is_image(self) -> bool:
        return self.type.casefold() == "image" or self.mime_type.casefold().startswith("image/")


@dataclass(slots=True)
class WechatMessage:
    message_id: str
    from_user_id: str
    context_token: str
    text: str
    media_items: tuple[WechatMediaItem, ...] = ()


@dataclass(slots=True)
class WechatUpdate:
    code: int
    message: str
    sync_buf: str
    timeout_ms: int
    messages: list[WechatMessage]


class WechatAccountStore:
    def __init__(self, paths: KairoPaths) -> None:
        self.root = paths.user_dir / "wechat"
        self.file = self.root / "account.json"

    def _check_paths(self, *extra: Path) -> None:
        _reject_wechat_symlinks(self.root.parent, self.root, self.file, *extra)

    def media_dir(self) -> Path:
        directory = self.root / "media"
        self._check_paths(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._check_paths(directory)
        if os.name != "nt":
            self.root.chmod(0o700)
            directory.chmod(0o700)
        return directory

    def load(self) -> WechatAccount | None:
        self._check_paths()
        if not self.file.is_file():
            return None
        try:
            if self.file.stat().st_size > MAX_WECHAT_ACCOUNT_BYTES:
                raise ValueError("WeChat account file exceeds the 128 KiB limit")
            with self.file.open("rb") as handle:
                encoded = handle.read(MAX_WECHAT_ACCOUNT_BYTES + 1)
            if len(encoded) > MAX_WECHAT_ACCOUNT_BYTES:
                raise ValueError("WeChat account file exceeds the 128 KiB limit")
        except OSError as exc:
            raise ValueError(f"Cannot read WeChat account: {type(exc).__name__}") from exc
        try:
            payload = json.loads(
                encoded.decode("utf-8"),
                object_pairs_hook=_wechat_object_without_duplicates,
                parse_constant=_reject_wechat_json_constant,
            )
            _validate_wechat_json_shape(payload)
        except (
            OverflowError,
            RecursionError,
            UnicodeError,
            ValueError,
        ) as exc:
            raise ValueError(f"Cannot read WeChat account: {type(exc).__name__}") from exc
        if not isinstance(payload, dict):
            raise ValueError("WeChat account root must be a JSON object")
        allowed = {
            "token",
            "account_id",
            "base_url",
            "bound_user_id",
            "workspace",
            "sync_buf",
            "created_at",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unknown WeChat account fields: {', '.join(sorted(unknown))}")
        account = WechatAccount(**{key: payload.get(key, "") for key in allowed})
        _validate_wechat_account(account)
        if os.name != "nt":
            self.file.parent.chmod(0o700)
            self.file.chmod(0o600)
        return account

    def save(self, account: WechatAccount) -> None:
        self._check_paths()
        _validate_wechat_account(account)
        with _wechat_account_file_lock(self.root, self.file):
            self._write_unlocked(account)

    def update_sync_buf(self, expected: WechatAccount, sync_buf: str) -> bool:
        if not isinstance(sync_buf, str) or len(sync_buf) > 65_536:
            raise ValueError("WeChat sync buffer is invalid or too long")
        with _wechat_account_file_lock(self.root, self.file):
            current = self.load()
            if current is None or current != expected:
                return False
            self._write_unlocked(replace(current, sync_buf=sync_buf))
            return True

    def _write_unlocked(self, account: WechatAccount) -> None:
        self._check_paths()
        encoded = json.dumps(
            asdict(account),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        if len(encoded) > MAX_WECHAT_ACCOUNT_BYTES:
            raise ValueError("WeChat account file exceeds the 128 KiB limit")
        descriptor, temporary = tempfile.mkstemp(
            prefix=".account-", suffix=".tmp", dir=self.file.parent
        )
        try:
            if os.name != "nt":
                os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.file)
            if os.name != "nt":
                self.file.chmod(0o600)
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            Path(temporary).unlink(missing_ok=True)
            raise

    def clear(self) -> None:
        self._check_paths()
        with _wechat_account_file_lock(self.root, self.file):
            self._check_paths()
            self.file.unlink(missing_ok=True)


def _wechat_object_without_duplicates(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate WeChat account key: {key}")
        result[key] = value
    return result


def _reject_wechat_json_constant(value: str) -> Any:
    raise ValueError(f"Non-standard JSON constant is not allowed: {value}")


def _validate_wechat_json_shape(value: Any) -> None:
    stack: list[tuple[Any, int]] = [(value, 0)]
    visited = 0
    while stack:
        current, depth = stack.pop()
        visited += 1
        if visited > MAX_WECHAT_ACCOUNT_JSON_NODES:
            raise ValueError("WeChat account JSON exceeds the node limit")
        if depth > MAX_WECHAT_ACCOUNT_JSON_DEPTH:
            raise ValueError("WeChat account JSON exceeds the nesting limit")
        children: Iterable[Any]
        if isinstance(current, dict):
            children = current.values()
        elif isinstance(current, list):
            children = current
        else:
            continue
        child_count = len(current)
        if visited + len(stack) + child_count > MAX_WECHAT_ACCOUNT_JSON_NODES:
            raise ValueError("WeChat account JSON exceeds the node limit")
        stack.extend((child, depth + 1) for child in children)


@contextmanager
def _wechat_account_file_lock(root: Path, account_file: Path) -> Iterator[None]:
    lock_file = root / ".account.lock"
    _reject_wechat_symlinks(root.parent, root, account_file, lock_file)
    root.mkdir(parents=True, exist_ok=True)
    _reject_wechat_symlinks(root.parent, root, account_file, lock_file)
    if os.name != "nt":
        root.chmod(0o700)
    descriptor = os.open(
        lock_file,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    locked = False
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"WeChat account lock is not a regular file: {lock_file}")
        if os.name == "posix":
            os.fchmod(descriptor, 0o600)
        with _WECHAT_ACCOUNT_THREAD_LOCK:
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


def _validate_wechat_account(account: WechatAccount) -> None:
    values = (
        account.token,
        account.account_id,
        account.base_url,
        account.bound_user_id,
        account.workspace,
        account.sync_buf,
        account.created_at,
    )
    if not all(isinstance(value, str) for value in values):
        raise ValueError("WeChat account fields must be strings")
    if not account.token.strip() or len(account.token) > 16_384:
        raise ValueError("WeChat account token is empty or too long")
    if not account.account_id.strip() or len(account.account_id) > 1_024:
        raise ValueError("WeChat account ID is empty or too long")
    if not account.bound_user_id.strip() or len(account.bound_user_id) > 1_024:
        raise ValueError("WeChat bound user ID is empty or too long")
    if len(account.sync_buf) > 65_536 or len(account.created_at) > 256:
        raise ValueError("WeChat account metadata is too long")
    if not account.workspace.strip() or len(account.workspace) > 4_096:
        raise ValueError("WeChat workspace is empty or too long")
    workspace = Path(account.workspace).expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError("WeChat workspace does not exist or is not a directory")
    account.base_url = _normalize_wechat_base_url(account.base_url)


def _normalize_wechat_base_url(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 8_192
        or "\\" in value
        or any(character.isspace() for character in value)
    ):
        raise ValueError("WeChat base URL is invalid or too long")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("WeChat base URL is invalid") from exc
    if parsed.scheme.casefold() != "https" or not parsed.hostname:
        raise ValueError("WeChat base URL must be an absolute HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("WeChat base URL cannot contain embedded credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("WeChat base URL cannot contain query or fragment")
    hostname = parsed.hostname.casefold()
    host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = f"{host}:{port}" if port is not None else host
    return urlunsplit(("https", netloc, parsed.path.rstrip("/"), "", ""))


def _validate_wechat_request_url(value: str) -> None:
    if len(value) > 8_192 or "\\" in value or any(character.isspace() for character in value):
        raise ValueError("WeChat request URL is invalid or too long")
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("WeChat request URL is invalid") from exc
    if parsed.scheme.casefold() != "https" or not parsed.hostname:
        raise ValueError("WeChat request URL must be an absolute HTTPS URL")
    if parsed.username is not None or parsed.password is not None or parsed.fragment:
        raise ValueError("WeChat request URL contains forbidden URL components")


def _safe_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and re.fullmatch(r"-?[0-9]{1,19}", value):
        parsed = int(value)
    else:
        return default
    return parsed if -(2**63) <= parsed <= 2**63 - 1 else default


def _wechat_string(
    value: Any, maximum: int, *, truncate: bool = False, allow_int: bool = False
) -> str:
    if isinstance(value, bool):
        return ""
    if allow_int and isinstance(value, int):
        value = str(value)
    if not isinstance(value, str):
        return ""
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError:
        return ""
    if len(value) <= maximum:
        return value
    return value[:maximum] if truncate else ""
