from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import socket
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

from ..trace import safe_redacted_text

MAX_FETCH_BYTES = 5 * 1024 * 1024
MAX_FETCH_CHARS = 100_000
MAX_SEARCH_RESULTS = 10
MAX_SEARCH_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_SEARCH_JSON_DEPTH = 32
MAX_SEARCH_JSON_NODES = 100_000
MAX_WEB_DOTENV_BYTES = 1024 * 1024
SEARCH_PROVIDERS = {"zhipu", "serpapi", "searxng"}


class NetworkDenied(PermissionError):
    pass


class NetworkPolicy:
    def __init__(
        self,
        allow_private: bool = False,
        *,
        max_requests: int = 30,
        window_seconds: float = 60,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.allow_private = allow_private
        self.max_requests = max(1, max_requests)
        self.window_seconds = max(0.01, window_seconds)
        self.clock = clock
        self._requests: deque[float] = deque()
        self._rate_lock = asyncio.Lock()

    async def check(self, url: str) -> None:
        if len(url) > 8_192:
            raise NetworkDenied("URL exceeds the 8192 character limit")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise NetworkDenied("Only HTTP(S) URLs with a host are allowed")
        if parsed.username is not None or parsed.password is not None:
            raise NetworkDenied("URLs containing embedded credentials are blocked")
        hostname = parsed.hostname.rstrip(".").casefold()
        if self.allow_private:
            return
        if hostname == "localhost" or hostname.endswith(".localhost"):
            raise NetworkDenied("Localhost targets are blocked")
        try:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
        except ValueError as exc:
            raise NetworkDenied("URL contains an invalid port") from exc
        try:
            addresses = await asyncio.to_thread(
                socket.getaddrinfo, parsed.hostname, port, type=socket.SOCK_STREAM
            )
        except socket.gaierror as exc:
            raise NetworkDenied(f"Cannot resolve host: {parsed.hostname}") from exc
        resolved_addresses = {item[4][0] for item in addresses}
        if not resolved_addresses:
            raise NetworkDenied(f"Cannot resolve host: {parsed.hostname}")
        for address in resolved_addresses:
            ip = ipaddress.ip_address(address)
            if not ip.is_global:
                raise NetworkDenied(
                    f"Private or local network target is blocked: {parsed.hostname}"
                )

    async def acquire(self) -> None:
        async with self._rate_lock:
            now = self.clock()
            cutoff = now - self.window_seconds
            while self._requests and self._requests[0] <= cutoff:
                self._requests.popleft()
            if len(self._requests) >= self.max_requests:
                retry_after = max(0.01, self.window_seconds - (now - self._requests[0]))
                raise NetworkDenied(f"Web request rate limit reached; retry in {retry_after:.1f}s")
            self._requests.append(now)

    def check_peer(self, address: str) -> None:
        if self.allow_private:
            return
        try:
            ip = ipaddress.ip_address(address)
        except ValueError as exc:
            raise NetworkDenied("Connected peer address is invalid") from exc
        if not ip.is_global:
            raise NetworkDenied("Connected peer resolved to a private or local address")


@dataclass(slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str = ""


class WebClient:
    def __init__(
        self,
        policy: NetworkPolicy | None = None,
        transport: Any = None,
        workspace: Path | None = None,
        environment: dict[str, str] | None = None,
    ) -> None:
        self.policy = policy or NetworkPolicy()
        self.transport = transport
        self.workspace = (workspace or Path.cwd()).resolve()
        self.environment = (
            environment if environment is not None else _load_web_environment(self.workspace)
        )

    async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Web search query cannot be empty")
        if len(normalized_query) > 2_000:
            raise ValueError("Web search query exceeds the 2000 character limit")
        maximum = min(max(int(limit), 1), MAX_SEARCH_RESULTS)
        providers, explicit = self._provider_order()
        if not providers:
            raise RuntimeError(
                "Configure GLM_API_KEY, SERPAPI_API_KEY, or KAIROCLI_SEARXNG_URL for web search"
            )
        errors: list[str] = []
        for provider in providers:
            try:
                results = await self._search_provider(provider, normalized_query, maximum)
            except Exception as exc:
                errors.append(f"{provider}: {_safe_error(exc)}")
                if explicit:
                    break
                continue
            if results:
                return results
            errors.append(f"{provider}: no results")
            if explicit:
                break
        raise RuntimeError("Web search providers failed: " + "; ".join(errors))

    def _provider_order(self) -> tuple[list[str], bool]:
        explicit = self.environment.get("KAIROCLI_SEARCH_PROVIDER", "").strip().casefold()
        if explicit:
            if explicit not in SEARCH_PROVIDERS:
                raise ValueError("KAIROCLI_SEARCH_PROVIDER must be zhipu, serpapi, or searxng")
            return [explicit], True
        providers: list[str] = []
        if self.environment.get("GLM_API_KEY", "").strip():
            providers.append("zhipu")
        if self._serp_key():
            providers.append("serpapi")
        if self.environment.get("KAIROCLI_SEARXNG_URL", "").strip():
            providers.append("searxng")
        return providers, False

    async def _search_provider(self, provider: str, query: str, limit: int) -> list[SearchResult]:
        if provider == "zhipu":
            key = self.environment.get("GLM_API_KEY", "").strip()
            if not key:
                raise RuntimeError("GLM_API_KEY is not configured")
            allowed_engines = {
                "search_std",
                "search_pro",
                "search_pro_sogou",
                "search_pro_quark",
            }
            engine = self.environment.get("KAIROCLI_ZHIPU_SEARCH_ENGINE", "search_std").strip()
            if engine not in allowed_engines:
                engine = "search_std"
            payload = await self._request_json(
                "https://open.bigmodel.cn/api/paas/v4/web_search",
                method="POST",
                json_body={
                    "search_engine": engine,
                    "search_query": query,
                    "count": limit,
                    "content_size": "medium",
                },
                headers={"Authorization": f"Bearer {key}"},
            )
            return _parse_results(payload.get("search_result"), "link", "content", "zhipu", limit)
        if provider == "serpapi":
            key = self._serp_key()
            if not key:
                raise RuntimeError("SERPAPI_API_KEY is not configured")
            payload = await self._request_json(
                "https://serpapi.com/search.json",
                params={"q": query, "api_key": key, "num": limit, "hl": "zh-cn"},
            )
            results = _parse_results(
                payload.get("organic_results"), "link", "snippet", "serpapi", limit
            )
            if not results and isinstance(payload.get("answer_box"), dict):
                answer_box = payload["answer_box"]
                snippet = _bounded_result_text(
                    answer_box.get("snippet") or answer_box.get("answer"), 5_000
                )
                if snippet:
                    results.append(SearchResult("Featured answer", "", snippet, "serpapi"))
            return results
        base = self.environment.get("KAIROCLI_SEARXNG_URL", "").strip().rstrip("/")
        if not base:
            raise RuntimeError("KAIROCLI_SEARXNG_URL is not configured")
        payload = await self._request_json(
            base + "/search",
            params={"q": query, "format": "json", "language": "zh"},
            allow_private=True,
        )
        return _parse_results(payload.get("results"), "url", "content", "searxng", limit)

    def _serp_key(self) -> str:
        return (
            self.environment.get("SERPAPI_API_KEY", "").strip()
            or self.environment.get("SERPAPI_KEY", "").strip()
        )

    async def fetch(self, url: str, max_chars: int = MAX_FETCH_CHARS) -> dict[str, Any]:
        import httpx

        current = url
        max_chars = min(max(int(max_chars), 1_000), MAX_FETCH_CHARS)
        async with httpx.AsyncClient(timeout=30, transport=self.transport) as client:
            for _ in range(6):
                await self.policy.check(current)
                await self.policy.acquire()
                async with client.stream(
                    "GET",
                    current,
                    headers={
                        "User-Agent": "Kairo-CLI/0.1",
                        "Accept": "text/html,application/xhtml+xml,text/plain,application/json",
                    },
                ) as response:
                    _check_response_peer(response, self.policy)
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise RuntimeError("Redirect response has no location")
                        current = urljoin(current, location)
                        continue
                    if response.status_code >= 400:
                        raise RuntimeError(f"web_fetch failed with HTTP {response.status_code}")
                    content_type = response.headers.get("content-type", "").casefold()
                    if content_type and not _supported_text_content_type(content_type):
                        raise RuntimeError(
                            f"web_fetch only supports textual responses, got {content_type}"
                        )
                    body = bytearray()
                    response_truncated = False
                    async for chunk in response.aiter_bytes():
                        remaining = MAX_FETCH_BYTES - len(body)
                        if len(chunk) > remaining:
                            body.extend(chunk[:remaining])
                            response_truncated = True
                            break
                        body.extend(chunk)
                    encoding = response.encoding or "utf-8"
                    try:
                        raw_text = bytes(body).decode(encoding, errors="replace")
                    except LookupError:
                        encoding = "utf-8"
                        raw_text = bytes(body).decode(encoding, errors="replace")
                    final_url = str(response.url)
                title, markdown = _extract_content(raw_text, content_type, final_url)
                original_chars = len(markdown)
                content_truncated = len(markdown) > max_chars
                body_empty = not markdown.strip()
                return {
                    "url": final_url,
                    "title": title,
                    "content": markdown[:max_chars],
                    "content_chars": original_chars,
                    "downloaded_bytes": len(body),
                    "charset": encoding,
                    "response_truncated": response_truncated,
                    "content_truncated": content_truncated,
                    "partial": response_truncated or content_truncated,
                    "body_empty": body_empty,
                    "hint": (
                        "No main text was extracted; the page may require JavaScript or login. "
                        "Use a browser tool instead of repeatedly fetching the same URL."
                        if body_empty
                        else ""
                    ),
                }
        raise RuntimeError("Too many redirects")

    async def _request_json(
        self,
        url: str,
        *,
        method: str = "GET",
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        allow_private: bool = False,
    ) -> dict[str, Any]:
        import httpx

        request_policy = NetworkPolicy(allow_private=True) if allow_private else self.policy
        for attempt in range(2):
            await request_policy.check(url)
            await self.policy.acquire()
            async with httpx.AsyncClient(timeout=30, transport=self.transport) as client:
                async with client.stream(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    headers={"Accept": "application/json", **(headers or {})},
                ) as response:
                    _check_response_peer(response, request_policy)
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        remaining = MAX_SEARCH_RESPONSE_BYTES - len(body)
                        if remaining <= 0 or len(chunk) > remaining:
                            raise RuntimeError("Search response exceeds the 2 MiB limit")
                        body.extend(chunk)
                    if response.status_code in {429, 500, 502, 503, 504} and attempt == 0:
                        await asyncio.sleep(0)
                        continue
                    if response.status_code >= 400:
                        detail = bytes(body[:200]).decode(errors="replace")
                        raise RuntimeError(
                            f"Search provider returned HTTP {response.status_code}: {detail}"
                        )
            try:
                parsed = _decode_search_json(bytes(body))
            except (RecursionError, TypeError, UnicodeError, ValueError) as exc:
                raise RuntimeError("Search provider returned invalid JSON") from exc
            if not isinstance(parsed, dict):
                raise RuntimeError("Search provider JSON must be an object")
            return parsed
        raise RuntimeError("Search provider retry exhausted")


def _supported_text_content_type(value: str) -> bool:
    media_type = value.split(";", 1)[0].strip()
    return media_type.startswith("text/") or media_type in {
        "application/json",
        "application/xml",
        "application/xhtml+xml",
    }


def _parse_results(
    raw: Any,
    url_key: str,
    snippet_key: str,
    source: str,
    limit: int,
) -> list[SearchResult]:
    if not isinstance(raw, list):
        return []
    results: list[SearchResult] = []
    for item in raw:
        if len(results) >= limit:
            break
        if not isinstance(item, dict):
            continue
        title = _bounded_result_text(item.get("title"), 500)
        snippet = _bounded_result_text(item.get(snippet_key), 5_000)
        if not title and not snippet:
            continue
        raw_url = item.get(url_key)
        results.append(
            SearchResult(
                title,
                _safe_search_url(raw_url if isinstance(raw_url, str) else ""),
                snippet,
                source,
            )
        )
    return results


def _safe_search_url(value: str) -> str:
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > 8_192
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in normalized)
    ):
        return ""
    parsed = urlparse(normalized)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return ""
    return normalized


def _extract_content(raw_text: str, content_type: str, base_url: str) -> tuple[str, str]:
    media_type = content_type.split(";", 1)[0].strip()
    if media_type == "application/json":
        try:
            parsed = _decode_search_json(raw_text)
        except (RecursionError, TypeError, UnicodeError, ValueError):
            return "", raw_text.strip()
        return "", json.dumps(parsed, ensure_ascii=False, indent=2, allow_nan=False)
    if media_type and media_type not in {"text/html", "application/xhtml+xml"}:
        return "", raw_text.strip()
    from bs4 import BeautifulSoup
    from markdownify import markdownify

    soup = BeautifulSoup(raw_text, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    for node in soup(
        [
            "script",
            "style",
            "noscript",
            "iframe",
            "nav",
            "aside",
            "header",
            "footer",
            "form",
            "svg",
            "canvas",
            "button",
        ]
    ):
        node.decompose()
    noise = {
        "ads",
        "advert",
        "banner",
        "popup",
        "modal",
        "subscribe",
        "newsletter",
        "related",
        "recommend",
        "comment",
        "share",
        "social",
        "breadcrumb",
        "sidebar",
        "promo",
        "cookie",
        "footer",
        "navigation",
    }
    for node in list(soup.select("[class], [id]")):
        classes = " ".join(str(value) for value in node.get_attribute_list("class"))
        marker = f"{classes} {node.get('id', '')}".casefold()
        if any(keyword in marker for keyword in noise):
            node.decompose()
    main = soup.select_one("article, main, [role=main]")
    if main is None or len(main.get_text(" ", strip=True)) <= 80:
        candidates = list(soup.select("div, section, article, main"))
        if soup.body is not None:
            candidates.append(soup.body)
        main = max(candidates, key=_content_score, default=None)
    if main is None:
        return title, ""
    for link in main.select("[href]"):
        link["href"] = urljoin(base_url, str(link.get("href", "")))
    markdown = markdownify(str(main), heading_style="ATX").strip()
    return title, _collapse_blank_lines(markdown)


def _content_score(node: Any) -> float:
    text = node.get_text(" ", strip=True)
    if len(text) < 80:
        return 0
    link_chars = sum(len(item.get_text(" ", strip=True)) for item in node.select("a"))
    return len(text) * (1 - min((link_chars / len(text)) * 2, 1))


def _collapse_blank_lines(value: str) -> str:
    import re

    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+\n", "\n", value)).strip()


def _check_response_peer(response: Any, policy: NetworkPolicy) -> None:
    stream = response.extensions.get("network_stream")
    getter = getattr(stream, "get_extra_info", None)
    if getter is None:
        return
    address = getter("server_addr") or getter("peername")
    if isinstance(address, tuple) and address:
        policy.check_peer(str(address[0]))
    elif isinstance(address, str):
        policy.check_peer(address)


def _safe_error(error: Exception) -> str:
    return safe_redacted_text(error, 500, "...[web error truncated]")


def _bounded_result_text(value: Any, maximum: int) -> str:
    return value.strip()[:maximum] if isinstance(value, str) else ""


def _decode_search_json(raw: str | bytes) -> Any:
    payload = json.loads(
        raw,
        object_pairs_hook=_web_object_without_duplicates,
        parse_constant=_reject_web_json_constant,
    )
    _validate_web_json_tree(payload)
    return payload


def _web_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate web response key: {key}")
        result[key] = value
    return result


def _reject_web_json_constant(value: str) -> Any:
    raise ValueError(f"Invalid web response JSON constant: {value}")


def _validate_web_json_tree(root: Any) -> None:
    nodes = 0
    stack: list[tuple[Any, int]] = [(root, 1)]
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > MAX_SEARCH_JSON_NODES:
            raise ValueError("Web response JSON is too complex")
        if depth > MAX_SEARCH_JSON_DEPTH:
            raise ValueError("Web response JSON is too deeply nested")
        if isinstance(value, dict):
            stack.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            stack.extend((item, depth + 1) for item in value)


def _load_web_environment(workspace: Path) -> dict[str, str]:
    relevant = {
        "KAIROCLI_SEARCH_PROVIDER",
        "KAIROCLI_SEARXNG_URL",
        "KAIROCLI_ZHIPU_SEARCH_ENGINE",
        "GLM_API_KEY",
        "SERPAPI_API_KEY",
        "SERPAPI_KEY",
    }
    values: dict[str, str] = {}
    for path in (Path.home() / ".kairocli" / ".env", workspace / ".env"):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            if path.stat().st_size > MAX_WEB_DOTENV_BYTES:
                continue
            with path.open("rb") as stream:
                encoded = stream.read(MAX_WEB_DOTENV_BYTES + 1)
            if len(encoded) > MAX_WEB_DOTENV_BYTES:
                continue
            lines = encoded.decode("utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            normalized = line.strip()
            if not normalized or normalized.startswith("#"):
                continue
            if normalized.startswith("export "):
                normalized = normalized[7:].lstrip()
            key, separator, value = normalized.partition("=")
            key = key.strip()
            if separator and key in relevant:
                values[key] = value.strip().strip("\"'")
    for key in relevant:
        if os.getenv(key):
            values[key] = str(os.environ[key]).strip()
    return values
