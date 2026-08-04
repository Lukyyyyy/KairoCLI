import io
import json
from pathlib import Path

import httpx
import pytest

import kairocli.tools.web as web_module
from kairocli.tools import ToolDefinition, ToolRegistry
from kairocli.web import MAX_FETCH_BYTES, NetworkDenied, NetworkPolicy, WebClient


def _nested_json_body(depth: int) -> bytes:
    value: object = 0
    for _ in range(depth):
        value = [value]
    return json.dumps({"extra": value}).encode()


async def test_network_policy_rejects_non_http() -> None:
    with pytest.raises(NetworkDenied):
        await NetworkPolicy().check("file:///etc/passwd")


def test_web_environment_ignores_symlinked_dotenv_and_preserves_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "work"
    (home / ".kairocli").mkdir(parents=True)
    workspace.mkdir()
    (home / ".kairocli" / ".env").write_text("GLM_API_KEY=user\n", encoding="utf-8")
    outside = tmp_path / "outside.env"
    outside.write_text("SERPAPI_API_KEY=external\n", encoding="utf-8")
    (workspace / ".env").symlink_to(outside)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("GLM_API_KEY", "process")
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    monkeypatch.delenv("SERPAPI_KEY", raising=False)

    environment = web_module._load_web_environment(workspace)

    assert environment["GLM_API_KEY"] == "process"
    assert "SERPAPI_API_KEY" not in environment


def test_web_environment_bounds_dotenv_growth_after_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "work"
    home.mkdir()
    workspace.mkdir()
    dotenv = workspace / ".env"
    dotenv.write_text("x", encoding="utf-8")
    requested: list[int] = []

    class GrowingReader(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            requested.append(size)
            return super().read(size)

    def growing_open(
        _path: Path, mode: str = "r", *args: object, **kwargs: object
    ) -> GrowingReader:
        assert mode == "rb"
        return GrowingReader(b"x" * 9)

    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(Path, "open", growing_open)
    monkeypatch.setattr(web_module, "MAX_WEB_DOTENV_BYTES", 8)
    for key in (
        "KAIROCLI_SEARCH_PROVIDER",
        "KAIROCLI_SEARXNG_URL",
        "KAIROCLI_ZHIPU_SEARCH_ENGINE",
        "GLM_API_KEY",
        "SERPAPI_API_KEY",
        "SERPAPI_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    assert web_module._load_web_environment(workspace) == {}
    assert requested == [9]


@pytest.mark.parametrize("url", ["http://127.0.0.1/a", "http://[::1]/a"])
async def test_network_policy_rejects_loopback(url: str) -> None:
    with pytest.raises(NetworkDenied):
        await NetworkPolicy().check(url)


async def test_network_policy_can_allow_configured_private_service() -> None:
    await NetworkPolicy(allow_private=True).check("http://127.0.0.1:8080/search")
    await NetworkPolicy(allow_private=True).check("http://localhost:8080/search")


async def test_network_policy_rejects_embedded_credentials() -> None:
    with pytest.raises(NetworkDenied, match="credentials"):
        await NetworkPolicy(allow_private=True).check("https://user:secret@example.com")


async def test_web_fetch_streams_with_download_and_content_budgets() -> None:
    body = b"<html><body>HEAD" + b"x" * (MAX_FETCH_BYTES + 100) + b"TAIL</body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, content=body)

    client = WebClient(NetworkPolicy(allow_private=True), transport=httpx.MockTransport(handler))
    result = await client.fetch("http://example.test/large", max_chars=1_000)
    assert result["downloaded_bytes"] == MAX_FETCH_BYTES
    assert result["response_truncated"] is True
    assert result["content_truncated"] is True
    assert result["partial"] is True
    assert len(result["content"]) == 1_000


async def test_web_fetch_rejects_binary_content_type() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "application/octet-stream"}, content=b"binary"
        )

    client = WebClient(NetworkPolicy(allow_private=True), transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="textual responses"):
        await client.fetch("http://example.test/file")


async def test_network_policy_rate_limit_resets_deterministically() -> None:
    now = [100.0]
    policy = NetworkPolicy(
        allow_private=True,
        max_requests=2,
        window_seconds=10,
        clock=lambda: now[0],
    )
    await policy.acquire()
    await policy.acquire()
    with pytest.raises(NetworkDenied, match="rate limit"):
        await policy.acquire()
    now[0] = 111.0
    await policy.acquire()


async def test_zhipu_search_contract_and_unsafe_result_url_filtering() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "search_result": [
                    {
                        "title": "Result",
                        "link": "javascript:alert(1)",
                        "content": "Evidence",
                    }
                ]
            },
        )

    client = WebClient(
        NetworkPolicy(allow_private=True),
        transport=httpx.MockTransport(handler),
        environment={"GLM_API_KEY": "secret"},
    )
    results = await client.search("current fact", 3)

    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert requests[0].headers["authorization"] == "Bearer secret"
    assert json.loads(requests[0].content)["search_query"] == "current fact"
    assert results[0].source == "zhipu"
    assert results[0].url == ""


async def test_auto_search_falls_back_but_explicit_provider_does_not() -> None:
    counts = {"zhipu": 0, "serpapi": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if "bigmodel" in request.url.host:
            counts["zhipu"] += 1
            return httpx.Response(503, text="temporary")
        counts["serpapi"] += 1
        return httpx.Response(
            200,
            json={
                "organic_results": [
                    {"title": "Fallback", "link": "https://example.com", "snippet": "ok"}
                ]
            },
        )

    environment = {"GLM_API_KEY": "glm", "SERPAPI_API_KEY": "serp"}
    client = WebClient(
        NetworkPolicy(allow_private=True),
        transport=httpx.MockTransport(handler),
        environment=environment,
    )
    results = await client.search("fallback")
    assert results[0].source == "serpapi"
    assert counts == {"zhipu": 2, "serpapi": 1}

    explicit = WebClient(
        NetworkPolicy(allow_private=True),
        transport=httpx.MockTransport(handler),
        environment={**environment, "KAIROCLI_SEARCH_PROVIDER": "zhipu"},
    )
    with pytest.raises(RuntimeError, match="zhipu"):
        await explicit.search("no fallback")
    assert counts["serpapi"] == 1


@pytest.mark.parametrize(
    "body",
    [
        b'{"search_result":[],"search_result":[{"title":"shadow"}]}',
        b'{"search_result":[{"title":"bad","score":NaN}]}',
        _nested_json_body(33),
        b'{"search_result":["\xff"]}',
    ],
)
async def test_search_provider_rejects_ambiguous_nonfinite_or_unbounded_json(
    body: bytes,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    client = WebClient(
        NetworkPolicy(allow_private=True),
        transport=httpx.MockTransport(handler),
        environment={"GLM_API_KEY": "secret", "KAIROCLI_SEARCH_PROVIDER": "zhipu"},
    )

    with pytest.raises(RuntimeError, match="invalid JSON"):
        await client._request_json("https://open.bigmodel.cn/search")


async def test_search_results_do_not_stringify_structured_fields_or_control_urls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "search_result": [
                    {
                        "title": {"nested": "not a title"},
                        "content": ["not", "a", "snippet"],
                        "link": "https://example.com/ignored",
                    },
                    {
                        "title": "Safe",
                        "content": "Evidence",
                        "link": "https://example.com/path\nforged",
                    },
                ]
            },
        )

    client = WebClient(
        NetworkPolicy(allow_private=True),
        transport=httpx.MockTransport(handler),
        environment={"GLM_API_KEY": "secret"},
    )

    results = await client.search("typed fields")

    assert [(result.title, result.snippet, result.url) for result in results] == [
        ("Safe", "Evidence", "")
    ]


async def test_web_fetch_extracts_main_content_and_reports_empty_spa() -> None:
    pages = {
        "/article": """
            <html><head><title>Article</title></head><body>
            <nav>navigation noise</nav><aside>sidebar noise</aside>
            <article><h1>Heading</h1><p>This is the main evidence paragraph with enough text
            to select the semantic article rather than the surrounding page shell and links.</p>
            <a href="/source">Source</a></article></body></html>
        """,
        "/spa": "<html><head><title>App</title></head><body><div id='app'></div></body></html>",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text=pages[request.url.path],
        )

    client = WebClient(NetworkPolicy(allow_private=True), transport=httpx.MockTransport(handler))
    article = await client.fetch("http://example.test/article")
    empty = await client.fetch("http://example.test/spa")

    assert article["title"] == "Article"
    assert "main evidence" in article["content"]
    assert "navigation noise" not in article["content"]
    assert "http://example.test/source" in article["content"]
    assert article["body_empty"] is False
    assert empty["body_empty"] is True
    assert "browser tool" in empty["hint"]


async def test_step_37_prefers_registered_search_mcp_with_schema_aliases(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(tmp_path)
    registry.current_provider = "step"
    registry.current_model = "step-3.7-flash"
    received: list[dict[str, object]] = []

    async def search(arguments: dict[str, object]) -> str:
        received.append(arguments)
        return "fresh step evidence"

    registry.register(
        ToolDefinition(
            "mcp__step-search__web_search",
            "search",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                },
            },
            search,
        )
    )

    result = json.loads(await registry.execute("web_search", {"query": "latest", "limit": 3}))

    assert received == [{"query": "latest", "top_k": 3}]
    assert result["provider"] == "step-mcp"
    assert result["results"][0]["snippet"] == "fresh step evidence"
    await registry.close()
