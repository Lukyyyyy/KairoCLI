from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .paths import reject_symlink_components
from .trace import safe_redacted_text

MAX_CDP_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_CDP_TABS = 1_000
MAX_CDP_JSON_DEPTH = 16
MAX_CDP_JSON_NODES = 50_000
MAX_SENSITIVE_RULE_BYTES = 128 * 1024
MAX_SENSITIVE_RULES = 1_000
MAX_SENSITIVE_PATTERN_CHARS = 1_024


class BrowserMode(StrEnum):
    ISOLATED = "isolated"
    SHARED = "shared"


@dataclass(frozen=True, slots=True)
class BrowserTab:
    id: str
    title: str
    url: str
    type: str = "page"


@dataclass(frozen=True, slots=True)
class SensitiveMatch:
    matched: bool
    pattern: str = ""


class SensitivePagePolicy:
    DEFAULT_PATTERNS = (
        "*://*.bank.*/*",
        "*://*.alipay.com/*",
        "*://*.paypal.com/*",
        "*://paypal.com/*",
        "*://*.stripe.com/*",
        "*://github.com/settings/*",
        "*://*.github.com/settings/*",
        "*://github.com/*/settings/*",
        "*://*.github.com/*/settings/*",
        "*://*.feishu.cn/admin/*",
        "*://*.larksuite.com/admin/*",
        "*://*.console.cloud.google.com/*",
        "*://*.console.aws.amazon.com/*",
        "*://*.portal.azure.com/*",
    )

    def __init__(self, user_rules_file: Path | None = None) -> None:
        patterns = list(self.DEFAULT_PATTERNS)
        if user_rules_file is not None:
            try:
                reject_symlink_components(user_rules_file, "Browser sensitive rules")
                if user_rules_file.is_file():
                    if user_rules_file.stat().st_size > MAX_SENSITIVE_RULE_BYTES:
                        raise ValueError("Browser sensitive rules exceed the size limit")
                    with user_rules_file.open("rb") as stream:
                        encoded = stream.read(MAX_SENSITIVE_RULE_BYTES + 1)
                    if len(encoded) > MAX_SENSITIVE_RULE_BYTES:
                        raise ValueError("Browser sensitive rules exceed the size limit")
                    user_patterns = []
                    for raw_line in encoded.decode("utf-8", errors="replace").splitlines():
                        line = raw_line.strip()
                        if (
                            not line
                            or line.startswith("#")
                            or len(line) > MAX_SENSITIVE_PATTERN_CHARS
                        ):
                            continue
                        user_patterns.append(line)
                        if len(user_patterns) >= MAX_SENSITIVE_RULES:
                            break
                    patterns.extend(user_patterns)
            except (OSError, ValueError):
                pass
        self.patterns = tuple(dict.fromkeys(item.casefold() for item in patterns))

    def match(self, url: str | None) -> SensitiveMatch:
        if not url:
            return SensitiveMatch(False)
        normalized = url.casefold()
        for pattern in self.patterns:
            if fnmatch.fnmatchcase(normalized, pattern):
                return SensitiveMatch(True, pattern)
        return SensitiveMatch(False)

    def is_sensitive(self, url: str) -> bool:
        return self.match(url).matched


class BrowserSession:
    def __init__(
        self,
        port: int = 9222,
        mode: BrowserMode = BrowserMode.ISOLATED,
        transport: Any = None,
    ) -> None:
        self.port = port
        self.mode = mode
        self.transport = transport
        self.connected = False
        self.browser_url: str | None = None
        self.last_navigated_url: str | None = None
        self.state_uncertain = False
        self.tabs: list[BrowserTab] = []
        self.agent_opened_tabs: set[str] = set()

    async def connect(self, port: int | None = None) -> list[BrowserTab]:
        target_port, base_url, tabs = await self.probe(port)
        self.switch_to_shared(base_url, tabs=tabs, port=target_port)
        return list(self.tabs)

    async def probe(self, port: int | None = None) -> tuple[int, str, list[BrowserTab]]:
        target_port = self.port if port is None else port
        if isinstance(target_port, bool) or not isinstance(target_port, int):
            raise ValueError("Chrome DevTools port must be between 1024 and 65535")
        if not 1024 <= target_port <= 65535:
            raise ValueError("Chrome DevTools port must be between 1024 and 65535")
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("Install Kairo CLI dependencies for browser sessions") from exc
        base_url = f"http://127.0.0.1:{target_port}"
        async with httpx.AsyncClient(timeout=3, transport=self.transport) as client:
            payload = await _read_cdp_json(client, base_url + "/json/version")
            if not _valid_cdp_version(payload):
                raise RuntimeError("Endpoint is not a Chrome DevTools target")
            raw_tabs = await _read_cdp_json(client, base_url + "/json/list")
        if not isinstance(raw_tabs, list):
            raise RuntimeError("Chrome DevTools /json/list did not return an array")
        if len(raw_tabs) > MAX_CDP_TABS:
            raise RuntimeError(f"Chrome DevTools returned more than {MAX_CDP_TABS} tabs")
        tabs = _parse_cdp_tabs(raw_tabs)
        return target_port, base_url, tabs

    def switch_to_shared(
        self,
        browser_url: str,
        *,
        tabs: list[BrowserTab] | None = None,
        port: int | None = None,
    ) -> None:
        if port is not None:
            self.port = port
        self.mode = BrowserMode.SHARED
        self.connected = True
        self.browser_url = browser_url
        self.last_navigated_url = None
        self.state_uncertain = False
        self.agent_opened_tabs.clear()
        self.tabs = list(tabs or [])

    def disconnect(self) -> None:
        self.switch_to_isolated()

    def switch_to_isolated(self) -> None:
        self.mode = BrowserMode.ISOLATED
        self.connected = False
        self.browser_url = None
        self.last_navigated_url = None
        self.state_uncertain = False
        self.tabs = []
        self.agent_opened_tabs.clear()

    def remember_navigation(self, url: str | None) -> None:
        if url and url.strip():
            self.last_navigated_url = url.strip()
            self.state_uncertain = False

    def mark_state_uncertain(self) -> None:
        self.state_uncertain = True

    def record_opened_tab(self, page_id: str | None) -> None:
        if page_id and page_id.strip():
            self.agent_opened_tabs.add(page_id.strip())

    def is_agent_opened_tab(self, page_id: str | None) -> bool:
        return bool(page_id) and page_id in self.agent_opened_tabs

    def status(self) -> str:
        state = "connected" if self.connected else "disconnected"
        tab_count = len(self.tabs)
        return (
            f"Chrome DevTools {state} on 127.0.0.1:{self.port} "
            f"({self.mode}; state={'uncertain' if self.state_uncertain else 'known'}; "
            f"tabs={tab_count}; agent_tabs={len(self.agent_opened_tabs)})"
        )


@dataclass(frozen=True, slots=True)
class BrowserCheck:
    blocked: bool = False
    requires_per_call_approval: bool = False
    reason: str = ""
    sensitive: bool = False
    url: str = ""
    pattern: str = ""


class BrowserGuard:
    PREFIX = "mcp__chrome-devtools__"
    WRITE_TOOLS = {
        "click",
        "drag",
        "fill",
        "fill_form",
        "handle_dialog",
        "hover",
        "press_key",
        "resize_page",
        "upload_file",
        "evaluate_script",
    }
    PAGE_ID = re.compile(r"(page[-_][A-Za-z0-9_-]+)")

    def __init__(self, session: BrowserSession, policy: SensitivePagePolicy) -> None:
        self.session = session
        self.policy = policy

    @classmethod
    def is_chrome_tool(cls, name: str) -> bool:
        return name.startswith(cls.PREFIX)

    def check(self, name: str, arguments: dict[str, Any]) -> BrowserCheck:
        if not self.is_chrome_tool(name):
            return BrowserCheck()
        local_name = name.removeprefix(self.PREFIX)
        target_url = (
            str(arguments.get("url", "")).strip()
            if local_name in {"navigate_page", "new_page"}
            else ""
        )
        effective_url = target_url or self.session.last_navigated_url or ""
        match = self.policy.match(effective_url)
        if (
            local_name == "close_page"
            and self.session.mode == BrowserMode.SHARED
            and not self.session.is_agent_opened_tab(_page_id(arguments))
        ):
            return BrowserCheck(
                blocked=True,
                reason=(
                    "Shared browser mode refuses to close a tab not opened by Kairo CLI; "
                    "close it manually in Chrome"
                ),
                sensitive=match.matched,
                url=effective_url,
                pattern=match.pattern,
            )
        if self.session.state_uncertain and local_name in self.WRITE_TOOLS:
            return BrowserCheck(
                requires_per_call_approval=True,
                reason=(
                    "Browser page state is uncertain after a canceled operation; this write "
                    "requires one-time approval until a navigation succeeds"
                ),
                sensitive=match.matched,
                url=effective_url,
                pattern=match.pattern,
            )
        if match.matched and local_name in self.WRITE_TOOLS:
            return BrowserCheck(
                requires_per_call_approval=True,
                reason=(
                    f"Sensitive page matched {match.pattern}; this browser write requires "
                    "one-time approval and cannot reuse approve-all"
                ),
                sensitive=True,
                url=effective_url,
                pattern=match.pattern,
            )
        return BrowserCheck(
            sensitive=match.matched,
            url=effective_url,
            pattern=match.pattern,
        )

    def apply_after_execution(self, name: str, arguments: dict[str, Any], result_text: str) -> None:
        if not self.is_chrome_tool(name):
            return
        local_name = name.removeprefix(self.PREFIX)
        if local_name in {"navigate_page", "new_page"}:
            self.session.remember_navigation(str(arguments.get("url", "")))
        if local_name == "new_page":
            page_id = _page_id(arguments)
            if not page_id:
                match = self.PAGE_ID.search(result_text)
                page_id = match.group(1) if match else ""
            self.session.record_opened_tab(page_id)
        elif local_name == "close_page":
            page_id = _page_id(arguments)
            if page_id:
                self.session.agent_opened_tabs.discard(page_id)

    def apply_after_cancellation(self, name: str) -> None:
        if not self.is_chrome_tool(name):
            return
        local_name = name.removeprefix(self.PREFIX)
        if local_name in self.WRITE_TOOLS or local_name in {
            "navigate_page",
            "new_page",
            "close_page",
            "select_page",
        }:
            self.session.mark_state_uncertain()


async def handle_browser_command(
    payload: str | None,
    browser: BrowserSession,
    *,
    mcp_manager: Any = None,
    approval_policy: Any = None,
    tools: Any = None,
) -> str:
    operation, _, argument = (payload or "status").strip().partition(" ")
    operation = operation.casefold() or "status"
    argument = argument.strip()
    if operation == "status":
        result = browser.status()
        if mcp_manager is not None:
            status = mcp_manager.status().get("chrome-devtools", "NOT CONFIGURED")
            result += f"\nchrome-devtools MCP: {status}"
        return result
    if operation == "connect":
        if argument:
            if not argument.isdigit():
                return "Usage: /browser connect [port]"
        if mcp_manager is not None:
            if "chrome-devtools" not in mcp_manager.configs:
                return "Browser connection failed: chrome-devtools MCP is not configured"

            async def connect_mcp() -> str:
                config = mcp_manager.configs["chrome-devtools"]
                if argument:
                    target_port, base_url, tabs = await browser.probe(int(argument))
                    mode_arg = f"--browser-url={base_url}"
                else:
                    target_port, base_url, tabs = browser.port, "autoConnect", []
                    mode_arg = "--autoConnect"

                def commit_connection() -> None:
                    browser.switch_to_shared(
                        base_url,
                        tabs=tabs,
                        port=target_port,
                    )
                    if approval_policy is not None:
                        approval_policy.clear_mcp_server_approvals("chrome-devtools")

                await mcp_manager.restart_with_args(
                    "chrome-devtools",
                    _browser_mode_args(config.args, mode_arg),
                    on_success=commit_connection,
                )
                if argument:
                    return f"Connected chrome-devtools MCP to {base_url}; {len(tabs)} tabs visible."
                return "Connected chrome-devtools MCP using Chrome autoConnect."

            try:
                return await connect_mcp()
            except Exception as exc:
                detail = safe_redacted_text(exc, 4_000, "...[browser error truncated]")
                return f"Browser connection failed: {type(exc).__name__}: {detail}"
        try:
            tabs = await browser.connect(int(argument) if argument else None)
        except Exception as exc:
            detail = safe_redacted_text(exc, 4_000, "...[browser error truncated]")
            return f"Browser connection failed: {type(exc).__name__}: {detail}"
        return f"Connected to Chrome DevTools; {len(tabs)} tabs visible."
    if operation == "disconnect":
        if mcp_manager is not None:

            async def disconnect_mcp() -> str:
                config = mcp_manager.configs.get("chrome-devtools")
                if config is None:
                    browser.disconnect()
                    if approval_policy is not None:
                        approval_policy.clear_mcp_server_approvals("chrome-devtools")
                    return "chrome-devtools MCP is not configured; local Browser state cleared."

                def commit_disconnection() -> None:
                    browser.disconnect()
                    if approval_policy is not None:
                        approval_policy.clear_mcp_server_approvals("chrome-devtools")

                await mcp_manager.restart_with_args(
                    "chrome-devtools",
                    _browser_mode_args(config.args, "--isolated=true"),
                    on_success=commit_disconnection,
                )
                return "Browser disconnected; chrome-devtools MCP is isolated."

            try:
                return await disconnect_mcp()
            except Exception as exc:
                detail = safe_redacted_text(exc, 4_000, "...[browser error truncated]")
                return f"Browser disconnect failed: {type(exc).__name__}: {detail}"
        browser.disconnect()
        return "Browser disconnected."
    if operation == "tabs":
        if tools is not None and browser.mode == BrowserMode.SHARED:
            return str(await tools.execute("mcp__chrome-devtools__list_pages", {}))
        if not browser.tabs:
            return "No browser tabs are currently visible."
        return "\n".join(f"{tab.id} {tab.title} {tab.url}" for tab in browser.tabs)
    return "Usage: /browser [status|connect [port]|disconnect|tabs]"


def _browser_mode_args(arguments: list[str], mode_argument: str) -> list[str]:
    result: list[str] = []
    skip_browser_url_value = False
    for argument in arguments:
        if skip_browser_url_value:
            skip_browser_url_value = False
            continue
        if argument == "--browser-url":
            skip_browser_url_value = True
            continue
        if argument in {"--autoConnect", "--isolated"} or argument.startswith(
            ("--browser-url=", "--isolated=")
        ):
            continue
        result.append(argument)
    result.append(mode_argument)
    return result


def _page_id(arguments: dict[str, Any]) -> str:
    return str(
        arguments.get("pageIdx") or arguments.get("pageId") or arguments.get("uid") or ""
    ).strip()


async def _read_cdp_json(client: Any, url: str) -> Any:
    body = bytearray()
    async with client.stream("GET", url) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            body.extend(chunk)
            if len(body) > MAX_CDP_RESPONSE_BYTES:
                raise RuntimeError("Chrome DevTools response exceeds the 2 MiB limit")
    try:
        payload = json.loads(
            bytes(body),
            object_pairs_hook=_cdp_object_without_duplicates,
            parse_constant=_reject_cdp_json_constant,
        )
        _validate_cdp_json_tree(payload)
        return payload
    except (RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise RuntimeError("Chrome DevTools returned invalid JSON") from exc


def _valid_cdp_version(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return any(
        isinstance(payload.get(key), str) and bool(payload[key].strip())
        for key in ("Browser", "webSocketDebuggerUrl")
    )


def _parse_cdp_tabs(raw_tabs: list[Any]) -> list[BrowserTab]:
    tabs: list[BrowserTab] = []
    seen: set[str] = set()
    for item in raw_tabs:
        if not isinstance(item, dict):
            continue
        identifier = _cdp_text(item.get("id"), 512)
        title = _cdp_text(item.get("title", ""), 4_096)
        url = _cdp_text(item.get("url", ""), 8_192)
        tab_type = _cdp_text(item.get("type", "page"), 128)
        if not identifier or title is None or url is None or tab_type is None:
            continue
        if identifier in seen:
            raise RuntimeError("Chrome DevTools returned duplicate tab IDs")
        seen.add(identifier)
        tabs.append(BrowserTab(identifier, title, url, tab_type))
    return tabs


def _cdp_text(value: Any, maximum: int) -> str | None:
    if not isinstance(value, str) or len(value) > maximum:
        return None
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        return None
    return value.strip()


def _cdp_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate Chrome DevTools response key: {key}")
        result[key] = value
    return result


def _reject_cdp_json_constant(value: str) -> Any:
    raise ValueError(f"Invalid Chrome DevTools JSON constant: {value}")


def _validate_cdp_json_tree(root: Any) -> None:
    nodes = 0
    stack: list[tuple[Any, int]] = [(root, 1)]
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > MAX_CDP_JSON_NODES:
            raise ValueError("Chrome DevTools JSON is too complex")
        if depth > MAX_CDP_JSON_DEPTH:
            raise ValueError("Chrome DevTools JSON is too deeply nested")
        if isinstance(value, dict):
            stack.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            stack.extend((item, depth + 1) for item in value)
