from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
from collections import Counter
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

import jieba  # type: ignore[import-untyped]

jieba.setLogLevel(logging.WARNING)

MAX_EMBEDDING_INPUT_CHARS = 2_000
MAX_EMBEDDING_BATCH = 128
MAX_CONCURRENT_EMBEDDING_BATCHES = 2
MAX_EMBEDDING_RESPONSE_BYTES = 10 * 1024 * 1024
MAX_EMBEDDING_DIMENSIONS = 65_536
MAX_EMBEDDING_JSON_DEPTH = 16
MAX_EMBEDDING_JSON_NODES = 5_000_000
MAX_STORED_VECTOR_BYTES = 2 * 1024 * 1024
RELATION_INDEX_VERSION = "3"


@dataclass(slots=True)
class CodeChunk:
    id: str
    path: str
    start_line: int
    end_line: int
    content: str
    kind: str = "file"
    name: str = ""


@dataclass(frozen=True, slots=True)
class CodeRelation:
    path: str
    line: int
    from_name: str
    to_name: str
    kind: str


class EmbeddingClient(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbeddingClient:
    """Deterministic offline embedding used when no remote embedding service is configured."""

    def __init__(self, dimensions: int = 384) -> None:
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError("Embedding dimensions must be positive")
        if dimensions > MAX_EMBEDDING_DIMENSIONS:
            raise ValueError(f"Embedding dimensions cannot exceed {MAX_EMBEDDING_DIMENSIONS}")
        self.dimensions = dimensions

    @property
    def signature(self) -> str:
        return f"hash-sha256:{self.dimensions}"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            counts = Counter(re.findall(r"[\w$]+", text.lower()))
            for token, count in counts.items():
                digest = hashlib.sha256(token.encode()).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                sign = 1.0 if digest[4] & 1 else -1.0
                vector[index] += sign * count
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


class HttpEmbeddingClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        *,
        provider: str = "openai",
        dimensions: int | None = None,
        max_batch_size: int = MAX_EMBEDDING_BATCH,
        transport: Any = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.provider = provider.casefold()
        self.dimensions = dimensions
        self.max_batch_size = max_batch_size
        self.transport = transport
        self._dimensions: int | None = None
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Embedding base URL must be HTTP(S)")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Embedding base URL cannot contain credentials, query or fragment")
        if self.provider == "alicloud" and (
            parsed.scheme != "https"
            or not (parsed.hostname or "").casefold().endswith(".aliyuncs.com")
            or parsed.path.rstrip("/") != "/compatible-mode/v1"
        ):
            raise ValueError(
                "AliCloud embedding base URL must be an HTTPS aliyuncs.com "
                "/compatible-mode/v1 endpoint"
            )
        if not self.model.strip() or len(self.model) > 200:
            raise ValueError("Embedding model must contain 1-200 characters")
        if len(self.api_key) > 16_384:
            raise ValueError("Embedding API key exceeds 16384 characters")
        if self.provider == "alicloud" and not self.api_key:
            raise ValueError(
                "AliCloud embedding requires KAIROCLI_EMBEDDING_API_KEY or DASHSCOPE_API_KEY"
            )
        if dimensions is not None and (
            isinstance(dimensions, bool)
            or not isinstance(dimensions, int)
            or not 1 <= dimensions <= MAX_EMBEDDING_DIMENSIONS
        ):
            raise ValueError("Embedding dimensions must be a positive bounded integer")
        if (
            isinstance(max_batch_size, bool)
            or not isinstance(max_batch_size, int)
            or not 1 <= max_batch_size <= MAX_EMBEDDING_BATCH
        ):
            raise ValueError(f"Embedding batch size must be between 1 and {MAX_EMBEDDING_BATCH}")

    @property
    def signature(self) -> str:
        return f"http:{self.provider}:{self.base_url}:{self.model}:{self.dimensions or 'default'}"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import httpx

        if not texts:
            return []
        if len(texts) > self.max_batch_size:
            raise ValueError(f"Embedding batch exceeds {self.max_batch_size} texts")
        inputs = [text[:MAX_EMBEDDING_INPUT_CHARS] for text in texts]
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        if self.provider == "ollama":
            url = self.base_url + "/api/embed"
            payload: dict[str, Any] = {"model": self.model, "input": inputs}
        else:
            url = self.base_url + "/embeddings"
            payload = {"model": self.model, "input": inputs}
            if self.dimensions is not None:
                payload["dimensions"] = self.dimensions
            if self.provider == "alicloud":
                payload["encoding_format"] = "float"
        async with httpx.AsyncClient(timeout=120, transport=self.transport) as client:
            for attempt in range(3):
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    if response.status_code == 429 and attempt < 2:
                        try:
                            delay = float(response.headers.get("retry-after", 2**attempt))
                        except ValueError:
                            delay = float(2**attempt)
                    else:
                        response.raise_for_status()
                        body = bytearray()
                        async for chunk in response.aiter_bytes():
                            body.extend(chunk)
                            if len(body) > MAX_EMBEDDING_RESPONSE_BYTES:
                                raise ValueError("Embedding response exceeds the 10 MiB limit")
                        break
                await asyncio.sleep(delay if math.isfinite(delay) and 0 <= delay <= 60 else 1)
        try:
            raw = json.loads(
                body,
                object_pairs_hook=_embedding_object_without_duplicates,
                parse_constant=_reject_embedding_json_constant,
            )
            _validate_embedding_json_tree(raw)
        except (RecursionError, TypeError, UnicodeError, ValueError) as exc:
            raise ValueError("Embedding response is not valid JSON") from exc
        vectors = _parse_embedding_vectors(raw, len(inputs))
        if vectors:
            dimensions = len(vectors[0])
            if self._dimensions is None:
                self._dimensions = dimensions
            elif self._dimensions != dimensions:
                raise ValueError("Embedding dimensions changed during this client session")
        return vectors


def embedding_client_from_environment(
    environment: Mapping[str, str] | None = None,
) -> EmbeddingClient:
    env = os.environ if environment is None else environment
    provider = env.get("KAIROCLI_EMBEDDING_PROVIDER", "hash").strip().casefold()
    if provider in {"", "hash", "offline"}:
        raw_dimensions = env.get("KAIROCLI_EMBEDDING_DIMENSIONS") or "384"
        try:
            dimensions = int(raw_dimensions)
        except ValueError as exc:
            raise ValueError("KAIROCLI_EMBEDDING_DIMENSIONS must be an integer") from exc
        if dimensions > MAX_EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"KAIROCLI_EMBEDDING_DIMENSIONS cannot exceed {MAX_EMBEDDING_DIMENSIONS}"
            )
        return HashEmbeddingClient(dimensions)
    defaults = {
        "ollama": ("nomic-embed-text:latest", "http://localhost:11434"),
        "openai": ("text-embedding-3-small", "https://api.openai.com/v1"),
        "zhipu": ("embedding-3", "https://open.bigmodel.cn/api/paas/v4"),
        "glm": ("embedding-3", "https://open.bigmodel.cn/api/paas/v4"),
        "alicloud": ("qwen3.7-text-embedding-flash", ""),
    }
    if provider not in defaults:
        raise ValueError(f"Unsupported KAIROCLI_EMBEDDING_PROVIDER: {provider}")
    default_model, default_url = defaults[provider]
    model = env.get("KAIROCLI_EMBEDDING_MODEL") or default_model
    base_url = env.get("KAIROCLI_EMBEDDING_BASE_URL") or default_url
    if provider == "alicloud" and not base_url:
        raise ValueError(
            "KAIROCLI_EMBEDDING_BASE_URL is required for the AliCloud workspace endpoint"
        )
    remote_dimensions = _optional_positive_int(
        env.get("KAIROCLI_EMBEDDING_DIMENSIONS"),
        1024 if provider == "alicloud" else None,
        "KAIROCLI_EMBEDDING_DIMENSIONS",
        MAX_EMBEDDING_DIMENSIONS,
    )
    default_batch_size = (
        _alicloud_batch_size(model) if provider == "alicloud" else MAX_EMBEDDING_BATCH
    )
    max_batch_size = _optional_positive_int(
        env.get("KAIROCLI_EMBEDDING_MAX_BATCH_SIZE"),
        default_batch_size,
        "KAIROCLI_EMBEDDING_MAX_BATCH_SIZE",
        MAX_EMBEDDING_BATCH,
    )
    assert max_batch_size is not None
    return HttpEmbeddingClient(
        base_url,
        model,
        env.get("KAIROCLI_EMBEDDING_API_KEY")
        or (env.get("DASHSCOPE_API_KEY", "") if provider == "alicloud" else ""),
        provider="zhipu" if provider == "glm" else provider,
        dimensions=remote_dimensions,
        max_batch_size=max_batch_size,
    )


def _optional_positive_int(
    value: str | None,
    default: int | None,
    name: str,
    maximum: int,
) -> int | None:
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not 1 <= parsed <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return parsed


def _alicloud_batch_size(model: str) -> int:
    normalized = model.casefold()
    if normalized.startswith("qwen3.7-text-embedding"):
        return 20
    if normalized in {"text-embedding-v1", "text-embedding-v2"}:
        return 25
    return 10


def _parse_embedding_vectors(raw: Any, expected: int) -> list[list[float]]:
    candidates: Any
    if isinstance(raw, dict) and isinstance(raw.get("embeddings"), list):
        candidates = raw["embeddings"]
    elif isinstance(raw, dict) and isinstance(raw.get("embedding"), list):
        candidates = [raw["embedding"]]
    elif isinstance(raw, dict) and isinstance(raw.get("data"), list):
        rows = raw["data"]
        if not all(isinstance(item, dict) for item in rows):
            raise ValueError("Embedding response data must contain objects")
        indexed: list[tuple[int, dict[str, Any]]] = []
        for position, item in enumerate(rows):
            index = item.get("index", position)
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ValueError("Embedding response indices must be non-negative integers")
            indexed.append((index, item))
        if sorted(index for index, _ in indexed) != list(range(expected)):
            raise ValueError("Embedding response indices must uniquely cover the requested inputs")
        candidates = [
            item.get("embedding") for _, item in sorted(indexed, key=lambda pair: pair[0])
        ]
    else:
        raise ValueError("Embedding response does not contain vectors")
    if len(candidates) != expected:
        raise ValueError(
            f"Embedding response returned {len(candidates)} vectors for {expected} texts"
        )
    vectors: list[list[float]] = []
    dimensions: int | None = None
    for candidate in candidates:
        vector = _validated_vector(candidate)
        if dimensions is None:
            dimensions = len(vector)
        elif dimensions != len(vector):
            raise ValueError("Embedding vectors must have consistent dimensions")
        vectors.append(vector)
    return vectors


class CodeChunker:
    def __init__(
        self,
        max_lines: int = 120,
        overlap: int = 15,
        tree_sitter_cache: Path | None = None,
    ) -> None:
        if overlap >= max_lines:
            raise ValueError("overlap must be less than max_lines")
        self.max_lines = max_lines
        self.overlap = overlap
        self.tree_sitter_cache = tree_sitter_cache
        self._unavailable_tree_sitter_languages: set[str] = set()

    def chunk(self, path: str, content: str) -> list[CodeChunk]:
        structural = self._structural_chunks(path, content)
        if structural:
            return structural
        return self._window_chunks(path, content)

    def chunk_with_relations(
        self, path: str, content: str
    ) -> tuple[list[CodeChunk], list[CodeRelation]]:
        return (
            self.chunk(path, content),
            _tree_sitter_relations(
                path,
                content,
                self.tree_sitter_cache or Path(".kairocli/tree-sitter-cache"),
            ),
        )

    def _window_chunks(
        self,
        path: str,
        content: str,
        *,
        line_offset: int = 0,
        kind: str = "file",
        name: str = "",
    ) -> list[CodeChunk]:
        lines = content.splitlines()
        chunks: list[CodeChunk] = []
        step = self.max_lines - self.overlap
        for start in range(0, len(lines), step):
            selected = lines[start : start + self.max_lines]
            if not selected:
                break
            body = "\n".join(selected)
            absolute_start = line_offset + start + 1
            digest = hashlib.sha256(
                f"{path}:{kind}:{name}:{absolute_start}:{body}".encode()
            ).hexdigest()[:20]
            chunks.append(
                CodeChunk(
                    digest,
                    path,
                    absolute_start,
                    line_offset + start + len(selected),
                    body,
                    kind,
                    name,
                )
            )
            if start + self.max_lines >= len(lines):
                break
        return chunks

    def _structural_chunks(self, path: str, content: str) -> list[CodeChunk]:
        if Path(path).suffix.casefold() == ".py":
            try:
                tree = ast.parse(content)
            except SyntaxError:
                return []
            nodes = [
                node
                for node in ast.walk(tree)
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and getattr(node, "end_lineno", None)
            ]
            return self._chunks_from_ranges(
                path,
                content,
                [
                    (
                        int(node.lineno),
                        int(node.end_lineno or node.lineno),
                        "class" if isinstance(node, ast.ClassDef) else "method",
                        node.name,
                    )
                    for node in nodes
                ],
            )
        ranges = _tree_sitter_declaration_ranges(
            path,
            content,
            self.tree_sitter_cache,
            self._unavailable_tree_sitter_languages,
        )
        if not ranges:
            ranges = _declaration_ranges(path, content)
        return self._chunks_from_ranges(path, content, ranges)

    def _chunks_from_ranges(
        self,
        path: str,
        content: str,
        ranges: list[tuple[int, int, str, str]],
    ) -> list[CodeChunk]:
        lines = content.splitlines()
        chunks: list[CodeChunk] = []
        seen: set[tuple[int, int, str, str]] = set()
        bounded_ranges: list[tuple[int, int, str, str]] = []
        for start, end, kind, name in sorted(ranges):
            bounded_start = max(1, start)
            bounded_end = min(max(bounded_start, end), len(lines))
            key = (bounded_start, bounded_end, kind, name)
            if key in seen:
                continue
            seen.add(key)
            bounded_ranges.append(key)
            body = "\n".join(lines[bounded_start - 1 : bounded_end])
            chunks.extend(
                self._window_chunks(
                    path,
                    body,
                    line_offset=bounded_start - 1,
                    kind=kind,
                    name=name,
                )
            )
        covered: list[tuple[int, int]] = []
        for start, end, _, _ in sorted(bounded_ranges, key=lambda item: (item[0], -item[1])):
            if covered and start <= covered[-1][1] + 1:
                covered[-1] = (covered[-1][0], max(covered[-1][1], end))
            else:
                covered.append((start, end))
        cursor = 1
        for start, end in [*covered, (len(lines) + 1, len(lines))]:
            if cursor < start:
                body = "\n".join(lines[cursor - 1 : start - 1])
                if body.strip():
                    chunks.extend(
                        self._window_chunks(path, body, line_offset=cursor - 1, kind="file")
                    )
            cursor = max(cursor, end + 1)
        chunks.sort(key=lambda chunk: (chunk.start_line, chunk.end_line, chunk.kind, chunk.name))
        return chunks


class VectorStore:
    def __init__(self, database: Path) -> None:
        if database.is_symlink() or database.parent.is_symlink():
            raise ValueError("Vector database cannot be a symlink")
        database.parent.mkdir(parents=True, exist_ok=True)
        self.database = database
        if os.name == "posix" and not database.exists():
            descriptor = os.open(database, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY, path TEXT NOT NULL, start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL, content TEXT NOT NULL, vector TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'file', name TEXT NOT NULL DEFAULT '')"""
            )
            columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(chunks)")}
            if "kind" not in columns:
                connection.execute(
                    "ALTER TABLE chunks ADD COLUMN kind TEXT NOT NULL DEFAULT 'file'"
                )
            if "name" not in columns:
                connection.execute("ALTER TABLE chunks ADD COLUMN name TEXT NOT NULL DEFAULT ''")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS file_state (
                path TEXT PRIMARY KEY, content_hash TEXT NOT NULL,
                chunk_count INTEGER NOT NULL, embedding_signature TEXT NOT NULL)"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY, value TEXT NOT NULL)"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS relations (
                path TEXT NOT NULL, line INTEGER NOT NULL, from_name TEXT NOT NULL,
                to_name TEXT NOT NULL, kind TEXT NOT NULL,
                PRIMARY KEY (path, line, from_name, to_name, kind))"""
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path)")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(from_name)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_relations_to ON relations(to_name)"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self._check_database_paths()
        connection = sqlite3.connect(self.database, timeout=10)
        try:
            connection.execute("PRAGMA busy_timeout=10000")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
            self._harden_database_files()

    def _check_database_paths(self) -> None:
        if self.database.parent.is_symlink():
            raise ValueError("Vector database cannot use a symlink parent")
        for path in self._database_files():
            if path.is_symlink():
                raise ValueError(f"Vector database file cannot be a symlink: {path.name}")

    def _harden_database_files(self) -> None:
        if os.name != "posix":
            return
        for path in self._database_files():
            if path.is_symlink():
                raise ValueError(f"Vector database file cannot be a symlink: {path.name}")
            if path.is_file():
                path.chmod(0o600)

    def _database_files(self) -> tuple[Path, Path, Path]:
        return (
            self.database,
            Path(str(self.database) + "-wal"),
            Path(str(self.database) + "-shm"),
        )

    def upsert(self, chunks: list[CodeChunk], vectors: list[list[float]]) -> None:
        _validate_vectors(chunks, vectors)
        with self._connect() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO chunks "
                "(id,path,start_line,end_line,content,vector,kind,name) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        chunk.id,
                        chunk.path,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.content,
                        json.dumps(vector, allow_nan=False, separators=(",", ":")),
                        chunk.kind,
                        chunk.name,
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ],
            )

    def search(self, vector: list[float], limit: int = 10) -> list[tuple[CodeChunk, float]]:
        query_vector = _validated_vector(vector)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id,path,start_line,end_line,content,"
                "CASE WHEN typeof(vector)='text' AND length(vector)<=? "
                "THEN vector ELSE NULL END,kind,name FROM chunks",
                (MAX_STORED_VECTOR_BYTES,),
            ).fetchall()
        scored: list[tuple[CodeChunk, float]] = []
        corrupt_paths: set[str] = set()
        for row in rows:
            try:
                stored = json.loads(row[5], parse_constant=_reject_embedding_json_constant)
                stored_vector = _validated_vector(stored, expected_dimensions=len(query_vector))
                chunk = _stored_code_chunk(row)
            except (OverflowError, TypeError, UnicodeError, ValueError):
                if isinstance(row[1], str):
                    corrupt_paths.add(row[1])
                continue
            scored.append((chunk, _cosine(query_vector, stored_vector)))
        self._invalidate_file_state(corrupt_paths)
        return sorted(
            scored,
            key=lambda item: (item[1], item[0].path, -item[0].start_line),
            reverse=True,
        )[: max(1, min(limit, 100))]

    def replace_path(
        self,
        path: str,
        chunks: list[CodeChunk],
        vectors: list[list[float]],
        content_hash: str = "",
        embedding_signature: str = "",
        relations: list[CodeRelation] | None = None,
    ) -> None:
        _validate_vectors(chunks, vectors)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM chunks WHERE path=?", (path,))
            connection.executemany(
                "INSERT INTO chunks "
                "(id,path,start_line,end_line,content,vector,kind,name) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        chunk.id,
                        chunk.path,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.content,
                        json.dumps(vector, allow_nan=False, separators=(",", ":")),
                        chunk.kind,
                        chunk.name,
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ],
            )
            connection.execute(
                "INSERT OR REPLACE INTO file_state VALUES (?, ?, ?, ?)",
                (path, content_hash, len(chunks), embedding_signature),
            )
            if relations is not None:
                self._replace_relations(connection, path, relations)

    def replace_relations(self, path: str, relations: list[CodeRelation]) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._replace_relations(connection, path, relations)

    @staticmethod
    def _replace_relations(
        connection: sqlite3.Connection, path: str, relations: list[CodeRelation]
    ) -> None:
        connection.execute("DELETE FROM relations WHERE path=?", (path,))
        connection.executemany(
            "INSERT OR IGNORE INTO relations (path,line,from_name,to_name,kind) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (item.path, item.line, item.from_name, item.to_name, item.kind)
                for item in relations
            ],
        )

    def file_state(self, path: str) -> tuple[str, int, str] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT content_hash,chunk_count,embedding_signature FROM file_state WHERE path=?",
                (path,),
            ).fetchone()
        return (str(row[0]), int(row[1]), str(row[2])) if row else None

    def paths(self) -> set[str]:
        with self._connect() as connection:
            rows = connection.execute("SELECT DISTINCT path FROM chunks").fetchall()
        return {str(row[0]) for row in rows}

    def delete_paths(self, paths: set[str]) -> None:
        if not paths:
            return
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.executemany("DELETE FROM chunks WHERE path=?", [(path,) for path in paths])
            connection.executemany(
                "DELETE FROM relations WHERE path=?", [(path,) for path in paths]
            )
            connection.executemany(
                "DELETE FROM file_state WHERE path=?", [(path,) for path in paths]
            )

    def relations(self, symbol: str, limit: int = 200) -> list[CodeRelation]:
        escaped = symbol.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        suffix = f"%.{escaped}"
        prefix = f"{escaped}.%"
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT path,line,from_name,to_name,kind FROM relations "
                "WHERE from_name=? OR from_name LIKE ? ESCAPE '\\' "
                "OR to_name=? OR to_name LIKE ? ESCAPE '\\' "
                "ORDER BY path,line,kind,to_name LIMIT ?",
                (symbol, prefix, symbol, suffix, max(1, min(limit, 200))),
            ).fetchall()
        return [CodeRelation(str(row[0]), int(row[1]), *map(str, row[2:])) for row in rows]

    def _invalidate_file_state(self, paths: set[str]) -> None:
        if not paths:
            return
        with self._connect() as connection:
            connection.executemany(
                "DELETE FROM file_state WHERE path=?", [(path,) for path in paths]
            )

    def embedding_signature(self) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key='embedding_signature'"
            ).fetchone()
        return str(row[0]) if row else ""

    def relation_index_version(self) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key='relation_index_version'"
            ).fetchone()
        return str(row[0]) if row else ""

    def set_relation_index_version(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO metadata VALUES ('relation_index_version', ?)",
                (RELATION_INDEX_VERSION,),
            )

    def reset_for_embedding(self, signature: str) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM relations")
            connection.execute("DELETE FROM file_state")
            connection.execute(
                "INSERT OR REPLACE INTO metadata VALUES ('embedding_signature', ?)",
                (signature,),
            )


class CodeIndex:
    IGNORED = {".git", ".kairocli", ".venv", "node_modules", "target", "dist", "build"}
    EXTENSIONS = {
        ".py",
        ".java",
        ".kt",
        ".js",
        ".jsx",
        ".cjs",
        ".mjs",
        ".ts",
        ".tsx",
        ".cts",
        ".mts",
        ".go",
        ".rs",
        ".c",
        ".h",
        ".cc",
        ".cpp",
        ".cxx",
        ".hh",
        ".hpp",
        ".hxx",
        ".cs",
        ".rb",
        ".php",
        ".swift",
        ".scala",
        ".sql",
        ".md",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
    }
    MAX_FILE_BYTES = 2 * 1024 * 1024

    def __init__(
        self,
        workspace: Path,
        database: Path,
        embedding: EmbeddingClient | None = None,
        chunker: CodeChunker | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.store = VectorStore(database)
        self.embedding = embedding or embedding_client_from_environment()
        self.chunker = chunker or CodeChunker(
            tree_sitter_cache=self.workspace / ".kairocli" / "tree-sitter-cache"
        )

    async def index(
        self,
        path: Path | None = None,
        progress: Callable[[int, int, str], Awaitable[None] | None] | None = None,
        file_seen: Callable[[str], None] | None = None,
    ) -> dict[str, int]:
        root = (path or self.workspace).resolve()
        root.relative_to(self.workspace)
        signature = embedding_signature(self.embedding)
        stored_signature = self.store.embedding_signature()
        if stored_signature != signature and root != self.workspace and self.store.paths():
            raise ValueError(
                "Embedding configuration changed; run /index without a path to rebuild all code"
            )
        if stored_signature != signature:
            self.store.reset_for_embedding(signature)
        rebuild_relations = self.store.relation_index_version() != RELATION_INDEX_VERSION
        files = sorted(file for file in root.rglob("*") if self._indexable(file))
        current: set[str] = set()
        chunk_count = 0
        pending: list[tuple[str, list[CodeChunk], str, list[CodeRelation]]] = []
        pending_chunks = 0
        total_files = len(files)

        async def flush_pending() -> None:
            nonlocal chunk_count, pending_chunks
            texts = [
                f"[{chunk.kind}:{chunk.name}] {chunk.content}"
                for _, chunks, _, _ in pending
                for chunk in chunks
            ]
            vectors = await self._embed_all(texts)
            cursor = 0
            for relative, chunks, content_hash, relations in pending:
                next_cursor = cursor + len(chunks)
                self.store.replace_path(
                    relative,
                    chunks,
                    vectors[cursor:next_cursor],
                    content_hash,
                    signature,
                    relations,
                )
                chunk_count += len(chunks)
                cursor = next_cursor
            pending.clear()
            pending_chunks = 0

        for position, file in enumerate(files, 1):
            relative = file.relative_to(self.workspace).as_posix()
            if progress is not None and (
                position == 1 or position == total_files or position % 25 == 0
            ):
                try:
                    update = progress(position, total_files, relative)
                    if update is not None:
                        await update
                except Exception:
                    # Progress rendering must not corrupt a valid index transaction.
                    pass
            current.add(relative)
            if file_seen is not None:
                try:
                    file_seen(relative)
                except Exception:
                    # Result rendering must not corrupt a valid index transaction.
                    pass
            raw, excluded = await asyncio.to_thread(_read_index_source, file, self.MAX_FILE_BYTES)
            if raw is None and not excluded:
                state = self.store.file_state(relative)
                chunk_count += state[1] if state else 0
                continue
            if raw is None:
                self.store.replace_path(
                    relative,
                    [],
                    [],
                    f"excluded:{self.MAX_FILE_BYTES}",
                    signature,
                    [],
                )
                continue
            content_hash = hashlib.sha256(raw).hexdigest()
            state = self.store.file_state(relative)
            if state and state[0] == content_hash and state[2] == signature:
                if rebuild_relations:
                    content = raw.decode("utf-8", errors="replace")
                    self.store.replace_relations(
                        relative,
                        _tree_sitter_relations(
                            relative,
                            content,
                            self.workspace / ".kairocli" / "tree-sitter-cache",
                        ),
                    )
                chunk_count += state[1]
                continue
            content = raw.decode("utf-8", errors="replace")
            chunks, relations = self.chunker.chunk_with_relations(relative, content)
            if not chunks:
                self.store.replace_path(relative, [], [], content_hash, signature, relations)
                continue
            pending.append((relative, chunks, content_hash, relations))
            pending_chunks += len(chunks)
            if pending_chunks >= self._embedding_batch_size() * MAX_CONCURRENT_EMBEDDING_BATCHES:
                await flush_pending()
        if pending:
            await flush_pending()
        relative_root = root.relative_to(self.workspace).as_posix()
        existing_in_scope = {
            stored
            for stored in self.store.paths()
            if relative_root == "."
            or stored == relative_root
            or stored.startswith(relative_root + "/")
        }
        self.store.delete_paths(existing_in_scope - current)
        if root == self.workspace:
            self.store.set_relation_index_version()
        return {"files": len(files), "chunks": chunk_count}

    async def search(self, query: str, limit: int = 10) -> list[dict[str, object]]:
        query = query.strip()
        if not query:
            raise ValueError("Search query cannot be empty")
        signature = embedding_signature(self.embedding)
        stored_signature = self.store.embedding_signature()
        if stored_signature != signature:
            if not self.store.paths():
                return []
            raise ValueError("Embedding configuration changed; run /index to rebuild code vectors")
        limit = max(1, min(limit, 30))
        vector = (await self._embed_all([query]))[0]
        candidates = self.store.search(vector, min(limit * 4, 100))
        query_tokens = _query_tokens(query)
        rescored: list[tuple[CodeChunk, float]] = []
        for chunk, vector_score in candidates:
            content_tokens = _query_tokens(f"{chunk.name} {chunk.content}")
            lexical = (
                len(query_tokens & content_tokens) / len(query_tokens) if query_tokens else 0.0
            )
            kind_boost = (
                0.03
                if chunk.kind in {"function", "method"}
                else 0.015
                if chunk.kind == "class"
                else 0
            )
            rescored.append((chunk, vector_score * 0.75 + lexical * 0.25 + kind_boost))
        rescored.sort(key=lambda item: (item[1], item[0].path), reverse=True)
        return [
            {
                "path": chunk.path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "score": round(score, 6),
                "kind": chunk.kind,
                "name": chunk.name,
                "content": chunk.content,
            }
            for chunk, score in rescored[:limit]
        ]

    async def _embed_all(self, texts: list[str]) -> list[list[float]]:
        raw_size = self._embedding_batch_size(len(texts) or 1)
        vectors: list[list[float]] = []
        batches = [texts[offset : offset + raw_size] for offset in range(0, len(texts), raw_size)]
        # ponytail: keep concurrency low; raise only if measured throughput needs it.
        for offset in range(0, len(batches), MAX_CONCURRENT_EMBEDDING_BATCHES):
            group = batches[offset : offset + MAX_CONCURRENT_EMBEDDING_BATCHES]
            results = await asyncio.gather(
                *(self.embedding.embed(batch) for batch in group), return_exceptions=True
            )
            for batch, result in zip(group, results, strict=True):
                if isinstance(result, BaseException):
                    raise result
                if len(result) != len(batch):
                    raise ValueError("Embedding client returned an unexpected vector count")
                vectors.extend(result)
        return vectors

    def _embedding_batch_size(self, default: int = MAX_EMBEDDING_BATCH) -> int:
        raw_size = getattr(self.embedding, "max_batch_size", default)
        if isinstance(raw_size, bool) or not isinstance(raw_size, int) or raw_size <= 0:
            raise ValueError("Embedding max_batch_size must be a positive integer")
        return raw_size

    def graph(self, symbol: str) -> list[dict[str, object]]:
        return [
            {
                "path": item.path,
                "line": item.line,
                "kind": item.kind,
                "from_name": item.from_name,
                "to_name": item.to_name,
                "text": f"{item.from_name} -> {item.to_name}",
            }
            for item in self.store.relations(symbol)
        ]

    def _indexable(self, file: Path) -> bool:
        if not file.is_file() or file.suffix.lower() not in self.EXTENSIONS:
            return False
        try:
            file.resolve().relative_to(self.workspace)
        except ValueError:
            return False
        relative = file.relative_to(self.workspace)
        return not any(part in self.IGNORED for part in relative.parts)


def _read_index_source(file: Path, max_bytes: int) -> tuple[bytes | None, bool]:
    try:
        if file.stat().st_size > max_bytes:
            return None, True
        with file.open("rb") as stream:
            raw = stream.read(max_bytes + 1)
    except OSError:
        return None, False
    if len(raw) > max_bytes or b"\x00" in raw[:8_192]:
        return None, True
    return raw, False


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    denominator = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return numerator / denominator if denominator else 0.0


def _validate_vectors(chunks: list[CodeChunk], vectors: list[list[float]]) -> None:
    if len(chunks) != len(vectors):
        raise ValueError("Chunk/vector count mismatch")
    normalized = [_validated_vector(vector) for vector in vectors]
    dimensions = {len(vector) for vector in normalized}
    if len(dimensions) > 1:
        raise ValueError("Embedding vectors must have one positive consistent dimension")


def _validated_vector(candidate: Any, *, expected_dimensions: int | None = None) -> list[float]:
    if not isinstance(candidate, list) or not candidate:
        raise ValueError("Embedding vectors must be non-empty arrays")
    if len(candidate) > MAX_EMBEDDING_DIMENSIONS:
        raise ValueError("Embedding vector exceeds the dimension limit")
    if expected_dimensions is not None and len(candidate) != expected_dimensions:
        raise ValueError("Embedding vector dimensions do not match the query")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in candidate):
        raise ValueError("Embedding vector values must be numbers")
    try:
        vector = [float(value) for value in candidate]
    except OverflowError as exc:
        raise ValueError("Embedding vector values must be finite") from exc
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("Embedding vectors must contain only finite numbers")
    return vector


def _stored_code_chunk(row: tuple[Any, ...]) -> CodeChunk:
    identifier, path, start_line, end_line, content, _, kind, name = row
    if not all(isinstance(value, str) for value in (identifier, path, content, kind, name)):
        raise ValueError("Stored code chunk text fields are invalid")
    if (
        isinstance(start_line, bool)
        or not isinstance(start_line, int)
        or isinstance(end_line, bool)
        or not isinstance(end_line, int)
        or start_line < 1
        or end_line < start_line
    ):
        raise ValueError("Stored code chunk line range is invalid")
    return CodeChunk(identifier, path, start_line, end_line, content, kind, name)


def _reject_embedding_json_constant(value: str) -> Any:
    raise ValueError(f"Invalid embedding JSON constant: {value}")


def _embedding_object_without_duplicates(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate embedding response key: {key}")
        result[key] = value
    return result


def _validate_embedding_json_tree(root: Any) -> None:
    nodes = 0
    stack: list[tuple[Any, int]] = [(root, 1)]
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > MAX_EMBEDDING_JSON_NODES:
            raise ValueError("Embedding response JSON is too complex")
        if depth > MAX_EMBEDDING_JSON_DEPTH:
            raise ValueError("Embedding response JSON is too deeply nested")
        if isinstance(value, dict):
            stack.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            stack.extend((item, depth + 1) for item in value)


def embedding_signature(embedding: EmbeddingClient) -> str:
    explicit = getattr(embedding, "signature", "")
    if explicit:
        return str(explicit)
    dimensions = getattr(embedding, "dimensions", "")
    model = getattr(embedding, "model", "")
    cls = type(embedding)
    return f"{cls.__module__}.{cls.__qualname__}:{model}:{dimensions}"


# Backward-compatible private aliases for callers and tests predating the shared helpers.
_cosine = cosine_similarity
_embedding_signature = embedding_signature


def _query_tokens(value: str) -> set[str]:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    tokens = {
        token.casefold()
        for token in re.findall(r"[A-Za-z_$][A-Za-z0-9_$]*|[\u3400-\u9fff]+", expanded)
        if len(token) >= 2
    }
    for sequence in re.findall(r"[\u3400-\u9fff]+", expanded):
        tokens.update(word for word in jieba.lcut(sequence) if len(word) >= 2)
        tokens.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tokens


_CLASS_DECLARATION = re.compile(
    r"^\s*(?:(?:public|private|protected|internal|export|abstract|sealed|final|open|data)\s+)*"
    r"(?:class|interface|enum|struct|trait|record|object|module)\s+([A-Za-z_$][\w$]*)"
)
_FUNCTION_DECLARATION = re.compile(
    r"^\s*(?:(?:public|private|protected|internal|static|final|virtual|override|export|default|async)\s+)*"
    r"(?:function|func|fun|fn|def)\s+(?:\([^)]*\)\s*)?([A-Za-z_$][\w$]*)"
)
_C_STYLE_METHOD = re.compile(
    r"^\s*(?:(?:public|private|protected|internal|static|final|virtual|override|abstract|synchronized|async)\s+)*"
    r"[A-Za-z_$][\w$<>,.?\[\] :]*\s+([A-Za-z_$][\w$]*)\s*\([^;{}]*\)\s*"
    r"(?:throws\s+[^{}]+)?\{"
)


_TREE_SITTER_NODE_KINDS: dict[str, dict[str, str]] = {
    "java": {
        "annotation_type_declaration": "class",
        "class_declaration": "class",
        "constructor_declaration": "method",
        "enum_declaration": "class",
        "interface_declaration": "class",
        "method_declaration": "method",
        "record_declaration": "class",
    },
    "go": {
        "function_declaration": "function",
        "method_declaration": "method",
        "type_spec": "class",
    },
    "rust": {
        "enum_item": "class",
        "function_item": "function",
        "impl_item": "class",
        "struct_item": "class",
        "trait_item": "class",
        "union_item": "class",
    },
    "c": {
        "enum_specifier": "class",
        "function_definition": "function",
        "struct_specifier": "class",
        "union_specifier": "class",
    },
    "cpp": {
        "class_specifier": "class",
        "enum_specifier": "class",
        "function_definition": "function",
        "union_specifier": "class",
    },
    "csharp": {
        "class_declaration": "class",
        "constructor_declaration": "method",
        "conversion_operator_declaration": "method",
        "delegate_declaration": "class",
        "destructor_declaration": "method",
        "enum_declaration": "class",
        "interface_declaration": "class",
        "local_function_statement": "function",
        "method_declaration": "method",
        "operator_declaration": "method",
        "record_declaration": "class",
        "struct_declaration": "class",
    },
    "javascript": {
        "class_declaration": "class",
        "function_declaration": "function",
        "generator_function_declaration": "function",
        "method_definition": "method",
        "variable_declarator": "function",
    },
    "typescript": {
        "abstract_class_declaration": "class",
        "class_declaration": "class",
        "enum_declaration": "class",
        "function_declaration": "function",
        "generator_function_declaration": "function",
        "interface_declaration": "class",
        "method_definition": "method",
        "method_signature": "method",
        "type_alias_declaration": "class",
        "variable_declarator": "function",
    },
    "tsx": {
        "abstract_class_declaration": "class",
        "class_declaration": "class",
        "enum_declaration": "class",
        "function_declaration": "function",
        "generator_function_declaration": "function",
        "interface_declaration": "class",
        "method_definition": "method",
        "method_signature": "method",
        "type_alias_declaration": "class",
        "variable_declarator": "function",
    },
    "kotlin": {
        "class_declaration": "class",
        "function_declaration": "function",
        "object_declaration": "class",
        "secondary_constructor": "method",
    },
}

_TREE_SITTER_METHOD_CONTAINERS = {
    "class_body",
    "class_declaration",
    "class_specifier",
    "impl_item",
    "interface_body",
    "interface_declaration",
    "trait_item",
}
_TREE_SITTER_NAME_NODES = {
    "destructor_name",
    "field_identifier",
    "identifier",
    "operator_name",
    "type_identifier",
}


def _tree_sitter_declaration_ranges(
    path: str,
    content: str,
    cache_dir: Path | None,
    unavailable_languages: set[str],
) -> list[tuple[int, int, str, str]]:
    try:
        from tree_sitter_language_pack import PackConfig, configure, detect_language, get_parser

        language = detect_language(path)
        kinds = _TREE_SITTER_NODE_KINDS.get(language or "")
        if not language or kinds is None or language in unavailable_languages:
            return []
        if cache_dir is not None:
            configure(PackConfig(cache_dir=str(cache_dir)))
        encoded = content.encode()
        parser = get_parser(language)
        tree = parser.parse(encoded)
        root = tree.root_node
    except Exception:
        if "language" in locals() and language:
            unavailable_languages.add(language)
        return []

    ranges: list[tuple[int, int, str, str]] = []
    stack = [root]
    while stack:
        node = stack.pop()
        kind = kinds.get(node.type)
        if kind is not None and _tree_sitter_is_definition(node):
            name = _tree_sitter_definition_name(node, encoded)
            if name:
                if kind == "function" and _tree_sitter_has_method_container(node):
                    kind = "method"
                ranges.append(
                    (
                        _line_from_byte(encoded, node.start_byte),
                        _line_from_byte(encoded, node.end_byte),
                        kind,
                        name,
                    )
                )
        stack.extend(node.named_children)
    return ranges


_TREE_SITTER_CALL_TYPES = {
    "call",
    "call_expression",
    "invocation_expression",
    "method_invocation",
}


def _tree_sitter_relations(path: str, content: str, cache_dir: Path) -> list[CodeRelation]:
    if Path(path).suffix.casefold() == ".py":
        return _python_relations(path, content)
    try:
        from tree_sitter_language_pack import PackConfig, configure, detect_language, get_parser

        language = detect_language(path)
        kinds = _TREE_SITTER_NODE_KINDS.get(language or "")
        if not language or kinds is None:
            return []
        configure(PackConfig(cache_dir=str(cache_dir)))
        encoded = content.encode()
        parser = get_parser(language)
        tree = parser.parse(encoded)
        root = tree.root_node
    except Exception:
        return []

    relations: list[CodeRelation] = []

    def text(node: Any) -> str:
        return encoded[node.start_byte : node.end_byte].decode(errors="replace")

    def call_target(node: Any) -> str:
        if node.type == "method_invocation":
            name = node.child_by_field_name("name")
            owner = node.child_by_field_name("object")
            target = ".".join(text(item) for item in (owner, name) if item is not None)
        else:
            target_node = next(
                (
                    node.child_by_field_name(field)
                    for field in ("function", "callee", "name", "method")
                    if node.child_by_field_name(field) is not None
                ),
                None,
            )
            if target_node is None and node.named_children:
                target_node = node.named_children[0]
            target = text(target_node) if target_node is not None else ""
        target = " ".join(target.split())
        return target if 0 < len(target) <= 200 else ""

    def visit(node: Any, class_name: str = "", caller: str = "") -> None:
        kind = kinds.get(node.type)
        name = ""
        if kind is not None and _tree_sitter_is_definition(node):
            name = _tree_sitter_definition_name(node, encoded)
        if name:
            if kind == "class":
                class_name = f"{class_name}.{name}" if class_name else name
                relations.append(
                    CodeRelation(
                        path,
                        _line_from_byte(encoded, node.start_byte),
                        Path(path).name,
                        class_name,
                        "defines",
                    )
                )
                caller = ""
            elif kind in {"function", "method"}:
                owner = caller or class_name
                qualified = f"{owner}.{name}" if owner else name
                if class_name and not caller:
                    relations.append(
                        CodeRelation(
                            path,
                            _line_from_byte(encoded, node.start_byte),
                            class_name,
                            qualified,
                            "contains",
                        )
                    )
                elif not owner:
                    relations.append(
                        CodeRelation(
                            path,
                            _line_from_byte(encoded, node.start_byte),
                            Path(path).name,
                            qualified,
                            "defines",
                        )
                    )
                caller = qualified
        if caller and node.type in _TREE_SITTER_CALL_TYPES:
            target = call_target(node)
            if target:
                relations.append(
                    CodeRelation(
                        path,
                        _line_from_byte(encoded, node.start_byte),
                        caller,
                        target,
                        "calls",
                    )
                )
        for child in node.named_children:
            visit(child, class_name, caller)

    visit(root)
    return relations


def _line_from_byte(content: bytes, offset: int) -> int:
    return content.count(b"\n", 0, offset) + 1


def _python_relations(path: str, content: str) -> list[CodeRelation]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []
    relations: list[CodeRelation] = []

    class Visitor(ast.NodeVisitor):
        class_name = ""
        caller = ""

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            previous_class, previous_caller = self.class_name, self.caller
            self.class_name = f"{self.class_name}.{node.name}" if self.class_name else node.name
            relations.append(
                CodeRelation(path, node.lineno, Path(path).name, self.class_name, "defines")
            )
            self.caller = ""
            self.generic_visit(node)
            self.class_name, self.caller = previous_class, previous_caller

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_function(node)

        def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            previous_caller = self.caller
            owner = self.caller or self.class_name
            qualified = f"{owner}.{node.name}" if owner else node.name
            if self.class_name and not self.caller:
                relations.append(
                    CodeRelation(path, node.lineno, self.class_name, qualified, "contains")
                )
            elif not owner:
                relations.append(
                    CodeRelation(path, node.lineno, Path(path).name, qualified, "defines")
                )
            self.caller = qualified
            self.generic_visit(node)
            self.caller = previous_caller

        def visit_Call(self, node: ast.Call) -> None:
            if self.caller:
                target = " ".join(ast.unparse(node.func).split())
                if 0 < len(target) <= 200:
                    relations.append(
                        CodeRelation(path, node.lineno, self.caller, target, "calls")
                    )
            self.generic_visit(node)

    Visitor().visit(tree)
    return relations


def _tree_sitter_is_definition(node: Any) -> bool:
    if node.type != "variable_declarator":
        return True
    value = node.child_by_field_name("value")
    return value is not None and value.type in {"arrow_function", "function_expression"}


def _tree_sitter_definition_name(node: Any, encoded: bytes) -> str:
    name = node.child_by_field_name("name")
    if name is None and node.type == "impl_item":
        name = node.child_by_field_name("type")
    if name is None:
        declarator = node.child_by_field_name("declarator")
        name = _tree_sitter_descendant_name(declarator)
    if name is None and node.type == "impl_item":
        name = _tree_sitter_descendant_name(node)
    if name is None:
        return ""
    return encoded[name.start_byte : name.end_byte].decode(errors="replace")


def _tree_sitter_descendant_name(node: Any) -> Any:
    if node is None:
        return None
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type in _TREE_SITTER_NAME_NODES:
            return current
        stack.extend(reversed(current.named_children))
    return None


def _tree_sitter_has_method_container(node: Any) -> bool:
    parent = node.parent
    while parent is not None:
        if parent.type in _TREE_SITTER_METHOD_CONTAINERS:
            return True
        parent = parent.parent
    return False


def _declaration_ranges(path: str, content: str) -> list[tuple[int, int, str, str]]:
    suffix = Path(path).suffix.casefold()
    if suffix not in {
        ".java",
        ".kt",
        ".js",
        ".jsx",
        ".cjs",
        ".mjs",
        ".ts",
        ".tsx",
        ".cts",
        ".mts",
        ".go",
        ".rs",
        ".c",
        ".h",
        ".cc",
        ".cpp",
        ".cxx",
        ".hh",
        ".hpp",
        ".hxx",
        ".cs",
        ".rb",
        ".php",
        ".swift",
        ".scala",
    }:
        return []
    lines = content.splitlines()
    declarations: list[tuple[int, str, str]] = []
    for index, line in enumerate(lines, 1):
        match = _CLASS_DECLARATION.match(line)
        if match:
            declarations.append((index, "class", match.group(1)))
            continue
        match = _FUNCTION_DECLARATION.match(line) or _C_STYLE_METHOD.match(line)
        if match and match.group(1) not in {"if", "for", "while", "switch", "catch"}:
            declarations.append((index, "method", match.group(1)))
    result: list[tuple[int, int, str, str]] = []
    for position, (start, kind, name) in enumerate(declarations):
        next_start = declarations[position + 1][0] if position + 1 < len(declarations) else 0
        end = _brace_end(lines, start)
        if end is None:
            end = next_start - 1 if next_start else len(lines)
        result.append((start, end, kind, name))
    return result


def _brace_end(lines: list[str], start: int) -> int | None:
    balance = 0
    opened = False
    for index in range(start - 1, len(lines)):
        line = re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', "", lines[index])
        opens = line.count("{")
        closes = line.count("}")
        if opens:
            opened = True
        balance += opens - closes
        if opened and balance <= 0:
            return index + 1
    return None
