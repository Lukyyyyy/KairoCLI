from __future__ import annotations

import json
import logging
import os
import re
import stat
import threading
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import jieba  # type: ignore[import-untyped]

from .agent.context import estimate_text_tokens
from .paths import KairoPaths, reject_symlink_components

jieba.setLogLevel(logging.WARNING)

_MEMORY_LOCK = threading.RLock()
_WORD = re.compile(r"[a-z0-9][a-z0-9_.+-]*", re.I)
_URL = re.compile(r"https?://[^\s，。！？、)）]+", re.I)
_MEMORY_ID = re.compile(r"[0-9a-f]{12}\Z")
MAX_MEMORY_FILE_BYTES = 10 * 1024 * 1024
MAX_MEMORY_ENTRIES = 1_000
MAX_MEMORY_FACT_CHARS = 10_000
MAX_MEMORY_JSON_DEPTH = 16
MAX_MEMORY_JSON_NODES = 10_000


@dataclass(slots=True)
class MemoryEntry:
    id: str
    fact: str
    scope: str
    project: str | None
    created_at: str

    @property
    def token_count(self) -> int:
        return estimate_text_tokens(self.fact)


def tokenize_memory_query(value: str) -> set[str]:
    normalized = value.casefold().strip()
    tokens = {word for word in _WORD.findall(normalized) if len(word) >= 2}
    for word in jieba.lcut(normalized):
        word = word.strip()
        if len(word) >= 2:
            tokens.add(word)
    return tokens


def browser_login_fact(user_input: str, recent_texts: list[str]) -> str | None:
    current = user_input or ""
    remember = any(
        marker in current
        for marker in (
            "记一下",
            "记住",
            "记下来",
            "以后记得",
            "下次记得",
            "保存这个偏好",
            "保存到长期记忆",
        )
    )
    lower = current.casefold()
    browser = "chrome" in lower or "浏览器" in current
    login = any(marker in current for marker in ("登录态", "已登录", "登录好的"))
    reuse = any(marker in current for marker in ("复用", "直接用", "连接"))
    if not remember or not browser or not (login or reuse):
        return None
    joined = "\n".join([*recent_texts, current])
    host: str | None = None
    for match in _URL.finditer(joined):
        parsed = urlparse(match.group(0))
        if parsed.hostname:
            host = parsed.hostname.casefold().removeprefix("www.")
    if host is None and ("yuque" in joined.casefold() or "语雀" in joined):
        host = "yuque.com"
    if host is None:
        return "用户明确允许在需要登录态的网站访问中复用已登录 Chrome。"
    label = "（语雀）" if "yuque.com" in host else ""
    separator = "时" if label else " 时"
    return f"访问 {host}{label}{separator}优先复用用户已登录的 Chrome 登录态。"


class MemoryStore:
    def __init__(self, paths: KairoPaths) -> None:
        self.paths = paths
        self.file = paths.memory_file

    def list_entries(self, include_global: bool = True) -> list[MemoryEntry]:
        project = str(self.paths.workspace)
        return [
            entry
            for entry in self._load()
            if entry.project == project or (include_global and entry.scope == "global")
        ]

    def save(self, fact: str, scope: str = "project") -> MemoryEntry:
        fact = " ".join(fact.split())
        if not fact:
            raise ValueError("Memory fact cannot be empty")
        if len(fact) > MAX_MEMORY_FACT_CHARS:
            raise ValueError(f"Memory fact cannot exceed {MAX_MEMORY_FACT_CHARS} characters")
        if scope not in {"project", "global"}:
            raise ValueError("Memory scope must be project or global")
        with _MEMORY_LOCK:
            with _memory_file_lock(self.file):
                entries = self._load_unlocked()
                normalized = fact.casefold()
                duplicate = next(
                    (
                        entry
                        for entry in entries
                        if " ".join(entry.fact.split()).casefold() == normalized
                    ),
                    None,
                )
                if duplicate is not None:
                    return duplicate
                entry = MemoryEntry(
                    id=uuid.uuid4().hex[:12],
                    fact=fact,
                    scope=scope,
                    project=None if scope == "global" else str(self.paths.workspace),
                    created_at=datetime.now(UTC).isoformat(),
                )
                entries.append(entry)
                entries = entries[-MAX_MEMORY_ENTRIES:]
                self._write_unlocked(entries)
                return entry

    def search(self, query: str, limit: int = 20) -> list[MemoryEntry]:
        scored = [(self._score(entry, query), entry) for entry in self.list_entries()]
        return [
            entry
            for score, entry in sorted(
                (item for item in scored if item[0] > 0),
                key=lambda item: (item[0], item[1].created_at),
                reverse=True,
            )[: max(0, limit)]
        ]

    def context_for_query(self, query: str, max_tokens: int, limit: int = 10) -> str:
        lines: list[str] = []
        used = 0
        for entry in self.search(query, limit):
            if used + entry.token_count > max_tokens:
                break
            lines.append(f"- [{entry.scope}] {entry.fact}")
            used += entry.token_count
        return "\n".join(lines)

    def delete(self, entry_id: str) -> bool:
        with _MEMORY_LOCK:
            with _memory_file_lock(self.file):
                entries = self._load_unlocked()
                retained = [entry for entry in entries if entry.id != entry_id]
                if len(retained) == len(entries):
                    return False
                self._write_unlocked(retained)
                return True

    def clear_visible(self) -> int:
        with _MEMORY_LOCK:
            with _memory_file_lock(self.file):
                entries = self._load_unlocked()
                project = str(self.paths.workspace)
                visible_ids = {
                    entry.id
                    for entry in entries
                    if entry.project == project or entry.scope == "global"
                }
                retained = [entry for entry in entries if entry.id not in visible_ids]
                self._write_unlocked(retained)
                return len(entries) - len(retained)

    def context(self, max_chars: int = 8_000) -> str:
        lines = [f"- [{entry.scope}] {entry.fact}" for entry in self.list_entries()]
        return "\n".join(lines)[:max_chars]

    def status_summary(self) -> str:
        entries = self.list_entries()
        project_count = sum(entry.scope == "project" for entry in entries)
        global_count = sum(entry.scope == "global" for entry in entries)
        tokens = sum(entry.token_count for entry in entries)
        return (
            f"长期记忆：{len(entries)} 条 / {tokens:,} token "
            f"（项目 {project_count} 条，全局 {global_count} 条）"
        )

    @staticmethod
    def _score(entry: MemoryEntry, query: str) -> float:
        content = entry.fact.casefold()
        normalized_query = query.casefold().strip()
        if not normalized_query:
            return 0
        if normalized_query in content:
            return 1.0
        tokens = tokenize_memory_query(normalized_query)
        if not tokens:
            return 0
        matched = sum(1 for token in tokens if token in content)
        if matched == 0:
            return 0
        try:
            created = datetime.fromisoformat(entry.created_at)
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            age_hours = max(0.0, (datetime.now(UTC) - created).total_seconds() / 3600)
        except ValueError:
            age_hours = 24
        decay = max(0.5, 1.0 - age_hours / 24)
        return matched / len(tokens) * decay * 1.2

    def _load(self) -> list[MemoryEntry]:
        with _MEMORY_LOCK:
            return self._load_unlocked()

    def _load_unlocked(self) -> list[MemoryEntry]:
        try:
            reject_symlink_components(self.file, "Memory storage")
        except ValueError:
            return []
        if not self.file.is_file():
            return []
        try:
            if self.file.stat().st_size > MAX_MEMORY_FILE_BYTES:
                return []
            with self.file.open("rb") as source:
                encoded = source.read(MAX_MEMORY_FILE_BYTES + 1)
            if len(encoded) > MAX_MEMORY_FILE_BYTES:
                return []
            raw = json.loads(
                encoded.decode("utf-8"),
                object_pairs_hook=_memory_object_without_duplicates,
                parse_constant=_reject_memory_json_constant,
            )
            _validate_memory_json_shape(raw)
        except (
            OSError,
            OverflowError,
            RecursionError,
            UnicodeError,
            ValueError,
        ):
            return []
        if not isinstance(raw, list):
            return []
        entries: list[MemoryEntry] = []
        for item in reversed(raw):
            if len(entries) >= MAX_MEMORY_ENTRIES:
                break
            if not isinstance(item, dict):
                continue
            try:
                identifier = str(item["id"])
            except KeyError:
                continue
            fact = item.get("fact", item.get("content", ""))
            scope = item.get("scope", "global")
            project = item.get("project")
            created_at = item.get("created_at", _legacy_timestamp(item))
            if (
                not _MEMORY_ID.fullmatch(identifier)
                or not isinstance(fact, str)
                or not fact.strip()
                or len(fact) > MAX_MEMORY_FACT_CHARS
                or scope not in {"project", "global"}
                or (project is not None and not isinstance(project, str))
                or not isinstance(created_at, str)
                or len(created_at) > 128
            ):
                continue
            entries.append(MemoryEntry(identifier, fact, scope, project, created_at))
        entries.reverse()
        return entries

    def _write_unlocked(self, entries: list[MemoryEntry]) -> None:
        reject_symlink_components(self.file, "Memory storage")
        self.file.parent.mkdir(parents=True, exist_ok=True)
        reject_symlink_components(self.file, "Memory storage")
        if os.name == "posix":
            self.file.parent.chmod(0o700)
        retained = entries[-MAX_MEMORY_ENTRIES:]
        encoded = b""
        while retained:
            encoded = (
                json.dumps(
                    [asdict(entry) for entry in retained],
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            if len(encoded) <= MAX_MEMORY_FILE_BYTES:
                break
            retained = retained[1:]
        temporary = self.file.with_name(f".{self.file.name}.{uuid.uuid4().hex}.tmp")
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(temporary, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded or b"[]\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.file)
            if os.name == "posix":
                self.file.chmod(0o600)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


def handle_memory_command(payload: str | None, memory: MemoryStore) -> str:
    normalized = (payload or "list").strip()
    operation, _, argument = normalized.partition(" ")
    operation = operation.casefold() or "list"
    argument = argument.strip()
    if operation == "list":
        entries = memory.list_entries()
    elif operation == "search" and argument:
        entries = memory.search(argument)
    elif operation == "delete" and argument:
        return "Deleted." if memory.delete(argument) else "Memory not found."
    elif operation == "clear":
        return f"Cleared {memory.clear_visible()} memories."
    else:
        return "Usage: /memory [list|search QUERY|delete ID|clear]"
    if not entries:
        return "No visible memories."
    return "\n".join(f"{entry.id} [{entry.scope}] {entry.fact}" for entry in entries)


def handle_save_command(payload: str | None, memory: MemoryStore) -> str:
    normalized = (payload or "").strip()
    global_scope = normalized.casefold().startswith("--global ")
    project_scope = normalized.casefold().startswith("--project ")
    fact = (
        normalized[9:].strip()
        if global_scope
        else normalized[10:].strip()
        if project_scope
        else normalized
    )
    if not fact:
        return "Usage: /save [--global|--project] <durable fact>"
    entry = memory.save(fact, "global" if global_scope else "project")
    return f"Saved memory {entry.id} ({entry.scope})."


def _legacy_timestamp(item: dict[str, object]) -> object:
    return item.get("timestamp", datetime.now(UTC).isoformat())


def _memory_object_without_duplicates(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate memory JSON key: {key}")
        result[key] = value
    return result


def _reject_memory_json_constant(value: str) -> object:
    raise ValueError(f"Non-standard JSON constant is not allowed: {value}")


def _validate_memory_json_shape(value: object) -> None:
    stack: list[tuple[object, int]] = [(value, 0)]
    visited = 0
    while stack:
        current, depth = stack.pop()
        visited += 1
        if visited > MAX_MEMORY_JSON_NODES:
            raise ValueError("Memory JSON exceeds the node limit")
        if depth > MAX_MEMORY_JSON_DEPTH:
            raise ValueError("Memory JSON exceeds the nesting limit")
        children: Iterable[object]
        if isinstance(current, dict):
            children = current.values()
        elif isinstance(current, list):
            children = current
        else:
            continue
        child_count = len(current)
        if visited + len(stack) + child_count > MAX_MEMORY_JSON_NODES:
            raise ValueError("Memory JSON exceeds the node limit")
        stack.extend((child, depth + 1) for child in children)


@contextmanager
def _memory_file_lock(file: Path) -> Iterator[None]:
    directory = file.parent
    reject_symlink_components(directory, "Memory storage")
    directory.mkdir(parents=True, exist_ok=True)
    reject_symlink_components(directory, "Memory storage")
    if os.name == "posix":
        directory.chmod(0o700)
    lock_file = directory / ".long_term_memory.lock"
    reject_symlink_components(lock_file, "Memory lock")
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_file, flags, 0o600)
    locked = False
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"Memory lock is not a regular file: {lock_file}")
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
        if os.name == "posix":
            os.fchmod(descriptor, 0o600)
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
