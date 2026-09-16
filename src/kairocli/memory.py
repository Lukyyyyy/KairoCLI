from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import stat
import threading
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import jieba  # type: ignore[import-untyped]

from .agent.context import estimate_text_tokens
from .paths import KairoPaths, reject_symlink_components
from .rag import (
    MAX_STORED_VECTOR_BYTES,
    EmbeddingClient,
    _validated_vector,
    cosine_similarity,
    embedding_signature,
)

jieba.setLogLevel(logging.WARNING)
log = logging.getLogger(__name__)

_MEMORY_LOCK = threading.RLock()
_WORD = re.compile(r"[a-z0-9][a-z0-9_.+-]*", re.I)
_URL = re.compile(r"https?://[^\s，。！？、)）]+", re.I)
_MEMORY_ID = re.compile(r"[0-9a-f]{12}\Z")
_MEMORY_KEY = re.compile(r"[a-z0-9][a-z0-9._:-]{0,127}\Z")
MAX_MEMORY_FILE_BYTES = 10 * 1024 * 1024
MAX_MEMORY_ENTRIES = 1_000
MAX_MEMORY_FACT_CHARS = 10_000
MAX_MEMORY_JSON_DEPTH = 16
MAX_MEMORY_JSON_NODES = 10_000
MAX_MEMORY_SEARCH_LIMIT = 20
MEMORY_RRF_K = 60
MEMORY_SEMANTIC_CANDIDATES = 50
MEMORY_SEMANTIC_MIN_SCORE = 0.45


@dataclass(slots=True)
class MemoryEntry:
    id: str
    fact: str
    scope: str
    project: str | None
    created_at: str
    key: str | None = None
    status: str = "active"
    superseded_by: str | None = None

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
    def __init__(
        self,
        paths: KairoPaths,
        embedding: EmbeddingClient | None = None,
        semantic_min_score: float = MEMORY_SEMANTIC_MIN_SCORE,
    ) -> None:
        if not -1 <= semantic_min_score <= 1:
            raise ValueError("Memory semantic minimum score must be between -1 and 1")
        self.paths = paths
        self.file = paths.memory_file
        self.embedding = embedding
        self.semantic_min_score = semantic_min_score
        self.embedding_file = paths.memory_embeddings_file

    def list_entries(
        self, include_global: bool = True, *, include_superseded: bool = False
    ) -> list[MemoryEntry]:
        project = str(self.paths.workspace)
        visible = [
            entry
            for entry in self._load()
            if (include_superseded or entry.status == "active")
            and (entry.project == project or (include_global and entry.scope == "global"))
        ]
        if include_superseded:
            return visible
        project_keys = {
            entry.key for entry in visible if entry.scope == "project" and entry.key is not None
        }
        return [
            entry
            for entry in visible
            if not (entry.scope == "global" and entry.key in project_keys)
        ]

    def save(
        self,
        fact: str,
        scope: str = "project",
        *,
        key: str | None = None,
        replaces: Iterable[str] = (),
    ) -> MemoryEntry:
        fact = " ".join(fact.split())
        if not fact:
            raise ValueError("Memory fact cannot be empty")
        if len(fact) > MAX_MEMORY_FACT_CHARS:
            raise ValueError(f"Memory fact cannot exceed {MAX_MEMORY_FACT_CHARS} characters")
        if scope not in {"project", "global"}:
            raise ValueError("Memory scope must be project or global")
        normalized_key = key.casefold().strip() if key is not None else None
        if normalized_key is not None and not _MEMORY_KEY.fullmatch(normalized_key):
            raise ValueError("Memory key must contain 1-128 lowercase letters, numbers, . _ : or -")
        replace_ids = set(replaces)
        if any(not _MEMORY_ID.fullmatch(identifier) for identifier in replace_ids):
            raise ValueError("Replacement memory IDs must be 12 lowercase hexadecimal characters")
        with _MEMORY_LOCK:
            with _memory_file_lock(self.file):
                entries = self._load_unlocked()
                normalized = fact.casefold()
                duplicate = next(
                    (
                        entry
                        for entry in entries
                        if entry.status == "active"
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
                    key=normalized_key,
                )
                replace_ids.update(
                    candidate.id
                    for candidate in entries
                    if normalized_key is not None
                    and candidate.status == "active"
                    and candidate.key == normalized_key
                    and candidate.scope == scope
                    and candidate.project == entry.project
                )
                replaceable_ids = {
                    candidate.id
                    for candidate in entries
                    if candidate.scope == scope and candidate.project == entry.project
                }
                if not replace_ids <= replaceable_ids:
                    raise ValueError("Cannot replace a memory from another scope or project")
                entries = [
                    replace(candidate, status="superseded", superseded_by=entry.id)
                    if candidate.id in replace_ids and candidate.status == "active"
                    else candidate
                    for candidate in entries
                ]
                entries.append(entry)
                entries = entries[-MAX_MEMORY_ENTRIES:]
                self._write_unlocked(entries)
                return entry

    async def save_with_embedding(
        self,
        fact: str,
        scope: str = "project",
        *,
        key: str | None = None,
        replaces: Iterable[str] = (),
    ) -> tuple[MemoryEntry, bool]:
        entry = self.save(fact, scope, key=key, replaces=replaces)
        if self.embedding is None:
            return entry, False
        signature = embedding_signature(self.embedding)
        try:
            cached = self._load_embedding_cache(signature).get(entry.id)
            if cached is not None and cached[0] == _fact_hash(entry.fact):
                return entry, True
            vectors = await _embed_all(self.embedding, [entry.fact])
            self._store_embeddings(signature, [entry], vectors)
        except Exception as exc:
            log.warning("memory_embedding_deferred error=%s", type(exc).__name__)
            return entry, False
        try:
            self._prune_embedding_cache()
        except (OSError, sqlite3.Error, ValueError) as exc:
            log.warning("memory_embedding_cleanup_deferred error=%s", type(exc).__name__)
        return entry, True

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

    async def search_hybrid(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        query = query.strip()
        if not query:
            return []
        bounded_limit = min(max(limit, 1), MAX_MEMORY_SEARCH_LIMIT)
        lexical = self.search(query, MEMORY_SEMANTIC_CANDIDATES)
        if self.embedding is None:
            return lexical[:bounded_limit]
        try:
            semantic = await self._semantic_search(query, MEMORY_SEMANTIC_CANDIDATES)
        except Exception:
            return lexical[:bounded_limit]
        entries = {entry.id: entry for entry in [*lexical, *semantic]}
        scores: dict[str, float] = {}
        for ranking in (lexical, semantic):
            for rank, entry in enumerate(ranking, 1):
                scores[entry.id] = scores.get(entry.id, 0.0) + 1 / (MEMORY_RRF_K + rank)
        normalized = query.casefold()
        return sorted(
            entries.values(),
            key=lambda entry: (
                normalized in entry.fact.casefold(),
                scores[entry.id],
                entry.created_at,
            ),
            reverse=True,
        )[:bounded_limit]

    async def context_for_hybrid_query(self, query: str, max_tokens: int, limit: int = 10) -> str:
        lines: list[str] = []
        used = 0
        for entry in await self.search_hybrid(query, limit):
            if used + entry.token_count > max_tokens:
                break
            key = f" key={entry.key}" if entry.key else ""
            lines.append(f"- [{entry.scope}{key}] {entry.fact}")
            used += entry.token_count
        return "\n".join(lines)

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
                self._delete_cached_embeddings({entry_id})
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
                self._delete_cached_embeddings(visible_ids)
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
        return matched / len(tokens)

    async def _semantic_search(self, query: str, limit: int) -> list[MemoryEntry]:
        entries = self.list_entries()
        if not entries or self.embedding is None:
            return []
        signature = embedding_signature(self.embedding)
        cached = self._load_embedding_cache(signature)
        missing = [
            entry
            for entry in entries
            if entry.id not in cached or cached[entry.id][0] != _fact_hash(entry.fact)
        ]
        texts = [query, *(entry.fact for entry in missing)]
        vectors = await _embed_all(self.embedding, texts)
        query_vector = vectors[0]
        if missing:
            self._store_embeddings(signature, missing, vectors[1:])
            cached.update(
                (entry.id, (_fact_hash(entry.fact), vector))
                for entry, vector in zip(missing, vectors[1:], strict=True)
            )
        incompatible = [
            entry
            for entry in entries
            if entry.id in cached and len(cached[entry.id][1]) != len(query_vector)
        ]
        if incompatible:
            repaired = await _embed_all(self.embedding, [entry.fact for entry in incompatible])
            self._store_embeddings(signature, incompatible, repaired)
            cached.update(
                (entry.id, (_fact_hash(entry.fact), vector))
                for entry, vector in zip(incompatible, repaired, strict=True)
            )
        scored = [
            (cosine_similarity(query_vector, cached[entry.id][1]), entry)
            for entry in entries
            if entry.id in cached
        ]
        return [
            entry
            for score, entry in sorted(
                (item for item in scored if item[0] >= self.semantic_min_score),
                key=lambda item: (item[0], item[1].created_at),
                reverse=True,
            )[:limit]
        ]

    def _load_embedding_cache(self, signature: str) -> dict[str, tuple[str, list[float]]]:
        with self._embedding_connection() as connection:
            rows = connection.execute(
                "SELECT memory_id,fact_hash,CASE WHEN typeof(vector)='text' AND length(vector)<=? "
                "THEN vector ELSE NULL END FROM memory_embeddings "
                "WHERE embedding_signature=?",
                (MAX_STORED_VECTOR_BYTES, signature),
            ).fetchall()
        cached: dict[str, tuple[str, list[float]]] = {}
        for memory_id, fact_hash, encoded in rows:
            try:
                if not isinstance(encoded, str):
                    continue
                vector = _validated_vector(json.loads(encoded))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            cached[str(memory_id)] = (str(fact_hash), vector)
        return cached

    def _store_embeddings(
        self, signature: str, entries: list[MemoryEntry], vectors: list[list[float]]
    ) -> None:
        encoded_vectors = [
            json.dumps(_validated_vector(vector), allow_nan=False, separators=(",", ":"))
            for vector in vectors
        ]
        if any(len(encoded.encode()) > MAX_STORED_VECTOR_BYTES for encoded in encoded_vectors):
            raise ValueError("Memory embedding vector exceeds the storage limit")
        with self._embedding_connection() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO memory_embeddings "
                "(memory_id,fact_hash,embedding_signature,vector) VALUES (?,?,?,?)",
                [
                    (
                        entry.id,
                        _fact_hash(entry.fact),
                        signature,
                        encoded,
                    )
                    for entry, encoded in zip(entries, encoded_vectors, strict=True)
                ],
            )

    def _delete_cached_embeddings(self, entry_ids: set[str]) -> None:
        if not entry_ids or not self.embedding_file.exists():
            return
        try:
            with self._embedding_connection() as connection:
                connection.executemany(
                    "DELETE FROM memory_embeddings WHERE memory_id=?",
                    [(entry_id,) for entry_id in entry_ids],
                )
        except (OSError, sqlite3.Error, ValueError) as exc:
            log.warning("memory_embedding_cleanup_deferred error=%s", type(exc).__name__)

    def _prune_embedding_cache(self) -> None:
        active_ids = {entry.id for entry in self._load() if entry.status == "active"}
        with self._embedding_connection() as connection:
            cached_ids = {
                str(row[0]) for row in connection.execute("SELECT memory_id FROM memory_embeddings")
            }
            connection.executemany(
                "DELETE FROM memory_embeddings WHERE memory_id=?",
                [(entry_id,) for entry_id in cached_ids - active_ids],
            )

    @contextmanager
    def _embedding_connection(self) -> Iterator[sqlite3.Connection]:
        reject_symlink_components(self.embedding_file, "Memory embedding cache")
        self.embedding_file.parent.mkdir(parents=True, exist_ok=True)
        reject_symlink_components(self.embedding_file, "Memory embedding cache")
        if os.name == "posix":
            self.embedding_file.parent.chmod(0o700)
        connection = sqlite3.connect(self.embedding_file, timeout=10)
        try:
            connection.execute("PRAGMA busy_timeout=10000")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS memory_embeddings ("
                "memory_id TEXT PRIMARY KEY, fact_hash TEXT NOT NULL, "
                "embedding_signature TEXT NOT NULL, vector TEXT NOT NULL)"
            )
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
            if os.name == "posix" and self.embedding_file.is_file():
                self.embedding_file.chmod(0o600)

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
            key = item.get("key")
            status = item.get("status", "active")
            superseded_by = item.get("superseded_by")
            if (
                not _MEMORY_ID.fullmatch(identifier)
                or not isinstance(fact, str)
                or not fact.strip()
                or len(fact) > MAX_MEMORY_FACT_CHARS
                or scope not in {"project", "global"}
                or (project is not None and not isinstance(project, str))
                or not isinstance(created_at, str)
                or len(created_at) > 128
                or (
                    key is not None and (not isinstance(key, str) or not _MEMORY_KEY.fullmatch(key))
                )
                or status not in {"active", "superseded"}
                or (
                    superseded_by is not None
                    and (
                        not isinstance(superseded_by, str)
                        or not _MEMORY_ID.fullmatch(superseded_by)
                    )
                )
            ):
                continue
            entries.append(
                MemoryEntry(
                    identifier,
                    fact,
                    scope,
                    project,
                    created_at,
                    key,
                    status,
                    superseded_by,
                )
            )
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


async def handle_memory_command(payload: str | None, memory: MemoryStore) -> str:
    normalized = (payload or "list").strip()
    operation, _, argument = normalized.partition(" ")
    operation = operation.casefold() or "list"
    argument = argument.strip()
    if operation == "list":
        entries = memory.list_entries()
    elif operation == "search" and argument:
        entries = await memory.search_hybrid(argument)
    elif operation == "delete" and argument:
        return "Deleted." if memory.delete(argument) else "Memory not found."
    elif operation == "clear":
        return f"Cleared {memory.clear_visible()} memories."
    else:
        return "Usage: /memory [list|search QUERY|delete ID|clear]"
    if not entries:
        return "No visible memories."
    return "\n".join(
        f"{entry.id}  {f'[{entry.scope}]':<9}  "
        f"{entry.created_at.replace('T', ' ')[:19]:<19}  {entry.fact}"
        for entry in entries
    )


async def handle_save_command(payload: str | None, memory: MemoryStore) -> str:
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
    entry, indexed = await memory.save_with_embedding(fact, "global" if global_scope else "project")
    suffix = "" if memory.embedding is None or indexed else " Semantic indexing deferred."
    return f"Saved memory {entry.id} ({entry.scope}).{suffix}"


def _legacy_timestamp(item: dict[str, object]) -> object:
    return item.get("timestamp", datetime.now(UTC).isoformat())


def _fact_hash(fact: str) -> str:
    return hashlib.sha256(fact.encode()).hexdigest()


async def _embed_all(embedding: EmbeddingClient, texts: list[str]) -> list[list[float]]:
    raw_batch_size = getattr(embedding, "max_batch_size", len(texts) or 1)
    if (
        isinstance(raw_batch_size, bool)
        or not isinstance(raw_batch_size, int)
        or raw_batch_size <= 0
    ):
        raise ValueError("Embedding max_batch_size must be a positive integer")
    vectors: list[list[float]] = []
    for offset in range(0, len(texts), raw_batch_size):
        batch_texts = texts[offset : offset + raw_batch_size]
        batch = await embedding.embed(batch_texts)
        if len(batch) != len(batch_texts):
            raise ValueError("Embedding client returned an unexpected vector count")
        vectors.extend(_validated_vector(vector) for vector in batch)
    return vectors


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
