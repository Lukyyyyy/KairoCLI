import io
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import httpx
import pytest

import kairocli.rag as rag_module
from kairocli.rag import (
    CodeChunk,
    CodeChunker,
    CodeIndex,
    HashEmbeddingClient,
    HttpEmbeddingClient,
    VectorStore,
    _parse_embedding_vectors,
    embedding_client_from_environment,
)


class CountingEmbedding:
    def __init__(self, signature: str = "counting-v1") -> None:
        self.signature = signature
        self.calls: list[list[str]] = []
        self.fail = False

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.fail:
            raise RuntimeError("embedding unavailable")
        return [
            [1.0, float("payment" in text.casefold()), float(len(text) % 17)]
            for text in texts
        ]


def test_embedding_factory_supports_offline_ollama_and_zhipu() -> None:
    offline = embedding_client_from_environment({})
    assert isinstance(offline, HashEmbeddingClient)
    assert offline.dimensions == 384

    ollama = embedding_client_from_environment(
        {
            "KAIROCLI_EMBEDDING_PROVIDER": "ollama",
            "KAIROCLI_EMBEDDING_MODEL": "nomic-custom",
        }
    )
    assert isinstance(ollama, HttpEmbeddingClient)
    assert ollama.provider == "ollama"
    assert ollama.model == "nomic-custom"
    assert ollama.base_url == "http://localhost:11434"

    zhipu = embedding_client_from_environment(
        {"KAIROCLI_EMBEDDING_PROVIDER": "glm"}
    )
    assert isinstance(zhipu, HttpEmbeddingClient)
    assert zhipu.provider == "zhipu"
    assert zhipu.model == "embedding-3"


def test_embedding_configuration_and_response_validation() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        embedding_client_from_environment(
            {"KAIROCLI_EMBEDDING_PROVIDER": "unknown"}
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        embedding_client_from_environment(
            {"KAIROCLI_EMBEDDING_DIMENSIONS": "999999"}
        )
    with pytest.raises(ValueError, match="credentials"):
        HttpEmbeddingClient("https://user:pass@example.test", "model")

    assert _parse_embedding_vectors(
        {
            "data": [
                {"index": 1, "embedding": [3, 4]},
                {"index": 0, "embedding": [1, 2]},
            ]
        },
        2,
    ) == [[1.0, 2.0], [3.0, 4.0]]
    for payload, expected, match in (
        ({"embeddings": [[1.0]]}, 2, "for 2 texts"),
        ({"embeddings": [[float("nan")]]}, 1, "finite"),
        ({"embeddings": [[1.0], [1.0, 2.0]]}, 2, "consistent"),
        ({"data": [{"index": "0", "embedding": [1]}]}, 1, "indices"),
        (
            {
                "data": [
                    {"index": 0, "embedding": [1]},
                    {"index": 0, "embedding": [2]},
                ]
            },
            2,
            "uniquely cover",
        ),
        ({"data": [{"index": 3, "embedding": [1]}]}, 1, "uniquely cover"),
    ):
        with pytest.raises(ValueError, match=match):
            _parse_embedding_vectors(payload, expected)


async def test_http_embedding_client_batches_truncates_and_authenticates() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/embeddings"
        assert request.headers["Authorization"] == "Bearer private-key"
        payload = json.loads(request.content)
        assert payload["model"] == "embedding-test"
        assert [len(value) for value in payload["input"]] == [2_000, 5]
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [3, 4]},
                    {"index": 0, "embedding": [1, 2]},
                ]
            },
        )

    client = HttpEmbeddingClient(
        "https://embedding.example.test/v1",
        "embedding-test",
        "private-key",
        transport=httpx.MockTransport(handler),
    )

    vectors = await client.embed(["x" * 2_500, "short"])

    assert vectors == [[1.0, 2.0], [3.0, 4.0]]


async def test_http_embedding_client_rejects_dimension_drift() -> None:
    responses = iter(
        [
            {"data": [{"index": 0, "embedding": [1, 2]}]},
            {"data": [{"index": 0, "embedding": [1, 2, 3]}]},
        ]
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(responses))

    client = HttpEmbeddingClient(
        "https://embedding.example.test/v1",
        "embedding-test",
        transport=httpx.MockTransport(handler),
    )
    assert await client.embed(["first"]) == [[1.0, 2.0]]
    with pytest.raises(ValueError, match="dimensions changed"):
        await client.embed(["second"])


@pytest.mark.parametrize(
    "body",
    [
        b'{"data":[],"data":[{"index":0,"embedding":[1]}]}',
        b'{"data":[{"index":0,"embedding":[NaN]}]}',
        json.dumps({"extra": [[[[[[[[[[[[[[[[[0]]]]]]]]]]]]]]]]]}).encode(),
    ],
)
async def test_http_embedding_client_rejects_ambiguous_or_unbounded_json(
    body: bytes,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    client = HttpEmbeddingClient(
        "https://embedding.example.test/v1",
        "embedding-test",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ValueError, match="not valid JSON"):
        await client.embed(["input"])


async def test_code_index_respects_embedding_batch_contract(tmp_path: Path) -> None:
    embedding = CountingEmbedding()
    embedding.max_batch_size = 2  # type: ignore[attr-defined]
    index = CodeIndex(tmp_path, tmp_path / "index.db", embedding)

    vectors = await index._embed_all(["one", "two", "three", "four", "five"])

    assert [len(call) for call in embedding.calls] == [2, 2, 1]
    assert len(vectors) == 5


def test_chunker_emits_python_and_java_symbol_metadata() -> None:
    chunker = CodeChunker(max_lines=20, overlap=2)
    python_chunks = chunker.chunk(
        "service.py",
        "class Service:\n    def run(self):\n        return 1\n",
    )
    assert {(chunk.kind, chunk.name) for chunk in python_chunks} == {
        ("class", "Service"),
        ("method", "run"),
    }
    java_chunks = chunker.chunk(
        "Service.java",
        "public class Service {\n"
        "  public int run() {\n"
        "    return 1;\n"
        "  }\n"
        "}\n",
    )
    assert ("class", "Service") in {
        (chunk.kind, chunk.name) for chunk in java_chunks
    }
    assert ("method", "run") in {
        (chunk.kind, chunk.name) for chunk in java_chunks
    }


async def test_index_skips_unchanged_files_and_removes_deleted_paths(tmp_path: Path) -> None:
    source = tmp_path / "service.py"
    source.write_text("def charge():\n    return 'paid'\n", encoding="utf-8")
    embedding = CountingEmbedding()
    index = CodeIndex(tmp_path, tmp_path / ".kairocli" / "index.db", embedding)
    first = await index.index()
    assert first == {"files": 1, "chunks": 1}
    assert len(embedding.calls) == 1
    second = await index.index()
    assert second == first
    assert len(embedding.calls) == 1

    source.write_text("def refund():\n    return 'done'\n", encoding="utf-8")
    await index.index()
    assert len(embedding.calls) == 2
    with closing(sqlite3.connect(index.store.database)) as connection, connection:
        contents = [row[0] for row in connection.execute("SELECT content FROM chunks")]
    assert any("refund" in content for content in contents)
    assert all("charge" not in content for content in contents)

    source.unlink()
    assert await index.index() == {"files": 0, "chunks": 0}
    assert index.store.paths() == set()


async def test_index_reports_throttled_progress(tmp_path: Path) -> None:
    for number in range(30):
        (tmp_path / f"file-{number:02}.md").write_text(
            f"# File {number}\ncontent\n", encoding="utf-8"
        )
    index = CodeIndex(
        tmp_path, tmp_path / ".kairocli" / "index.db", CountingEmbedding()
    )
    updates: list[tuple[int, int, str]] = []

    async def progress(position: int, total: int, path: str) -> None:
        updates.append((position, total, path))

    result = await index.index(progress=progress)
    assert result["files"] == 30
    assert [item[0] for item in updates] == [1, 25, 30]
    assert all(item[1] == 30 for item in updates)

    def broken_progress(position: int, total: int, path: str) -> None:
        raise RuntimeError("status widget disappeared")

    assert await index.index(progress=broken_progress) == result


async def test_index_preserves_previous_path_when_embedding_fails(tmp_path: Path) -> None:
    source = tmp_path / "service.py"
    source.write_text("def stable():\n    return True\n", encoding="utf-8")
    embedding = CountingEmbedding()
    index = CodeIndex(tmp_path, tmp_path / ".kairocli" / "index.db", embedding)
    await index.index()
    source.write_text("def changed():\n    return False\n", encoding="utf-8")
    embedding.fail = True
    with pytest.raises(RuntimeError, match="unavailable"):
        await index.index()
    with closing(sqlite3.connect(index.store.database)) as connection, connection:
        content = str(connection.execute("SELECT content FROM chunks").fetchone()[0])
    assert "stable" in content and "changed" not in content


async def test_index_and_graph_bound_source_growth_after_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "service.py"
    source.write_text("def stable():\n    return True\n", encoding="utf-8")
    embedding = CountingEmbedding()
    index = CodeIndex(tmp_path, tmp_path / ".kairocli" / "index.db", embedding)
    await index.index()
    source.write_text("x", encoding="utf-8")
    requested: list[int] = []
    real_open = Path.open

    class GrowingReader(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            requested.append(size)
            return super().read(size)

    def growing_open(
        path: Path, mode: str = "r", *args: object, **kwargs: object
    ) -> Any:
        if path == source and mode == "rb":
            return GrowingReader(b"x" * 9)
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(index, "MAX_FILE_BYTES", 8)
    monkeypatch.setattr(Path, "open", growing_open)

    result = await index.index()
    graph = index.graph("stable")

    assert result == {"files": 1, "chunks": 0}
    assert index.store.file_state("service.py") == ("excluded:8", 0, embedding.signature)
    assert graph == []
    assert requested == [9, 9]


async def test_embedding_signature_change_invalidates_all_vectors(tmp_path: Path) -> None:
    source = tmp_path / "service.py"
    source.write_text("def work():\n    return 1\n", encoding="utf-8")
    database = tmp_path / ".kairocli" / "index.db"
    first = CountingEmbedding("model-v1")
    await CodeIndex(tmp_path, database, first).index()
    second = CountingEmbedding("model-v2")
    index = CodeIndex(tmp_path, database, second)
    await index.index()
    assert len(second.calls) == 1
    assert index.store.embedding_signature() == "model-v2"


async def test_partial_reindex_removes_deleted_files_only_inside_scope(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    stale = first_dir / "stale.py"
    stale.write_text("def stale(): pass\n", encoding="utf-8")
    (second_dir / "keep.py").write_text("def keep(): pass\n", encoding="utf-8")
    index = CodeIndex(
        tmp_path, tmp_path / ".kairocli" / "index.db", CountingEmbedding()
    )
    await index.index()
    stale.unlink()
    await index.index(first_dir)
    assert index.store.paths() == {"second/keep.py"}


@pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="symlinks unavailable")
async def test_index_refuses_workspace_symlink_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.py"
    outside.write_text("SECRET = 'outside'\n", encoding="utf-8")
    link = workspace / "linked.py"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is not permitted")
    index = CodeIndex(
        workspace, workspace / ".kairocli" / "index.db", CountingEmbedding()
    )
    assert await index.index() == {"files": 0, "chunks": 0}


def test_vector_store_rejects_inconsistent_or_nonfinite_vectors(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "vectors.db")
    chunks = [
        CodeChunk("a", "a.py", 1, 1, "a"),
        CodeChunk("b", "b.py", 1, 1, "b"),
    ]
    with pytest.raises(ValueError, match="dimension"):
        store.upsert(chunks, [[1.0], [1.0, 2.0]])
    with pytest.raises(ValueError, match="finite"):
        store.upsert(chunks[:1], [[float("nan")]])

    with pytest.raises(ValueError, match="numbers"):
        store.upsert(chunks[:1], [[True]])
    with pytest.raises(ValueError, match="cannot exceed"):
        HashEmbeddingClient(rag_module.MAX_EMBEDDING_DIMENSIONS + 1)


def test_vector_store_search_isolates_corrupt_rows_and_marks_paths_for_reindex(
    tmp_path: Path,
) -> None:
    database = tmp_path / "corrupt-vectors.db"
    store = VectorStore(database)
    paths = ["valid.py", "json.py", "nan.py", "shape.py", "lines.py", "large.py"]
    for index, path in enumerate(paths):
        store.replace_path(
            path,
            [CodeChunk(f"chunk-{index}", path, 1, 1, path)],
            [[1.0, 0.0]],
            f"hash-{index}",
            "model",
        )
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("UPDATE chunks SET vector='{' WHERE path='json.py'")
        connection.execute("UPDATE chunks SET vector='[NaN,0]' WHERE path='nan.py'")
        connection.execute("UPDATE chunks SET vector='[1]' WHERE path='shape.py'")
        connection.execute("UPDATE chunks SET start_line='bad' WHERE path='lines.py'")
        connection.execute(
            "UPDATE chunks SET vector=? WHERE path='large.py'",
            ("[" + " " * rag_module.MAX_STORED_VECTOR_BYTES + "]",),
        )

    results = store.search([1.0, 0.0], limit=10)

    assert [chunk.path for chunk, _ in results] == ["valid.py"]
    assert store.file_state("valid.py") is not None
    assert all(store.file_state(path) is None for path in paths[1:])


@pytest.mark.parametrize("query", [[], [float("inf")], [True]])
def test_vector_store_rejects_invalid_query_vectors(
    tmp_path: Path, query: list[float]
) -> None:
    store = VectorStore(tmp_path / "invalid-query.db")

    with pytest.raises(ValueError, match="Embedding vector"):
        store.search(query)


async def test_code_index_repairs_corrupt_vector_on_next_index(tmp_path: Path) -> None:
    source = tmp_path / "service.py"
    source.write_text("def payment():\n    return True\n", encoding="utf-8")
    embedding = CountingEmbedding()
    index = CodeIndex(tmp_path, tmp_path / ".kairocli" / "index.db", embedding)
    await index.index()
    with closing(sqlite3.connect(index.store.database)) as connection, connection:
        connection.execute("UPDATE chunks SET vector='not-json'")

    assert await index.search("payment") == []
    assert index.store.file_state("service.py") is None
    assert await index.index() == {"files": 1, "chunks": 1}
    assert len(embedding.calls) == 3
    assert (await index.search("payment"))[0]["path"] == "service.py"


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm"])
def test_vector_store_rechecks_database_symlinks(tmp_path: Path, suffix: str) -> None:
    database = tmp_path / "vectors.db"
    store = VectorStore(database)
    attacked = Path(str(database) + suffix)
    target = tmp_path / f"outside{suffix or '-db'}"
    if suffix:
        target.touch()
        attacked.unlink(missing_ok=True)
    else:
        database.replace(target)
    attacked.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        store.paths()


def test_vector_store_closes_connections_and_rolls_back_failed_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connections: list[sqlite3.Connection] = []
    closed: set[int] = set()
    original_connect = sqlite3.connect

    class TrackingConnection(sqlite3.Connection):
        def close(self) -> None:
            closed.add(id(self))
            super().close()

    def tracking_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        kwargs["factory"] = TrackingConnection
        connection = original_connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(rag_module.sqlite3, "connect", tracking_connect)
    store = VectorStore(tmp_path / "closed.db")
    original = CodeChunk("original", "service.py", 1, 1, "stable")
    store.replace_path("service.py", [original], [[1.0]], "hash", "model")
    duplicates = [
        CodeChunk("duplicate", "service.py", 1, 1, "first"),
        CodeChunk("duplicate", "service.py", 2, 2, "second"),
    ]
    with pytest.raises(sqlite3.IntegrityError):
        store.replace_path("service.py", duplicates, [[1.0], [1.0]], "new", "model")

    assert store.search([1.0])[0][0].content == "stable"
    assert connections and closed == {id(connection) for connection in connections}
