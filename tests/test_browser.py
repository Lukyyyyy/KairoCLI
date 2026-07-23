import asyncio
import io
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import kairocli.browser as browser_module
from kairocli.browser import (
    BrowserGuard,
    BrowserMode,
    BrowserSession,
    SensitivePagePolicy,
    handle_browser_command,
)
from kairocli.cancellation import AgentCanceled
from kairocli.cli import _register_browser_agent_tools
from kairocli.models import ToolOutput
from kairocli.policy import ApprovalPolicy, ApprovalResult
from kairocli.tools import ToolDefinition, ToolRegistry


def test_sensitive_page_policy_uses_defaults_and_user_globs(tmp_path: Path) -> None:
    rules = tmp_path / "sensitive_patterns.txt"
    rules.write_text("# custom\n*://example.com/admin/*\n", encoding="utf-8")
    policy = SensitivePagePolicy(rules)
    assert policy.is_sensitive("https://secure.bank.example/transfer")
    match = policy.match("https://example.com/admin/users")
    assert match.matched and match.pattern == "*://example.com/admin/*"
    assert not policy.is_sensitive("https://example.com/docs")


def test_sensitive_page_policy_ignores_symlinked_rule_container(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    outside = tmp_path / "outside-rules"
    outside.mkdir()
    rules = outside / "sensitive_patterns.txt"
    rules.write_text("*://example.com/private/*\n", encoding="utf-8")
    user_dir = home / ".kairocli"
    user_dir.symlink_to(outside, target_is_directory=True)

    policy = SensitivePagePolicy(user_dir / "sensitive_patterns.txt")

    assert not policy.is_sensitive("https://example.com/private/data")
    assert policy.is_sensitive("https://paypal.com/account")


def test_sensitive_page_policy_bounds_rule_growth_after_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rules = tmp_path / "sensitive_patterns.txt"
    rules.write_text("x", encoding="utf-8")
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

    monkeypatch.setattr(browser_module, "MAX_SENSITIVE_RULE_BYTES", 8)
    monkeypatch.setattr(Path, "open", growing_open)

    policy = SensitivePagePolicy(rules)

    assert requested == [9]
    assert not policy.is_sensitive("https://example.com/private/data")


def test_sensitive_page_policy_bounds_rule_count_and_pattern_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rules = tmp_path / "sensitive_patterns.txt"
    rules.write_text(
        "*://one.example/*\n" + "x" * 21 + "\n*://two.example/*\n*://three.example/*\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(browser_module, "MAX_SENSITIVE_RULES", 2)
    monkeypatch.setattr(browser_module, "MAX_SENSITIVE_PATTERN_CHARS", 20)

    policy = SensitivePagePolicy(rules)

    assert policy.is_sensitive("https://one.example/a")
    assert policy.is_sensitive("https://two.example/a")
    assert not policy.is_sensitive("https://three.example/a")


async def test_browser_connect_probes_version_and_switches_to_shared_mode() -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if request.url.path == "/json/version":
            return httpx.Response(200, json={"Browser": "Chrome/1"})
        return httpx.Response(
            200,
            json=[
                {"id": "existing", "title": "Docs", "url": "https://example.com"},
                {"title": "missing id"},
            ],
        )

    session = BrowserSession(9222, transport=httpx.MockTransport(handler))
    tabs = await session.connect()
    assert requests == ["/json/version", "/json/list"]
    assert session.connected and session.mode == BrowserMode.SHARED
    assert session.browser_url == "http://127.0.0.1:9222"
    assert [tab.id for tab in tabs] == ["existing"]
    session.disconnect()
    assert session.mode == BrowserMode.ISOLATED and not session.tabs


async def test_browser_connect_rejects_invalid_port_and_non_cdp_endpoint() -> None:
    session = BrowserSession(80)
    try:
        await session.connect()
    except ValueError as exc:
        assert "1024" in str(exc)
    else:
        raise AssertionError("invalid port was accepted")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"service": "not chrome"})

    session = BrowserSession(9222, transport=httpx.MockTransport(handler))
    try:
        await session.connect()
    except RuntimeError as exc:
        assert "Chrome DevTools" in str(exc)
    else:
        raise AssertionError("non-CDP endpoint was accepted")


async def test_browser_connect_bounds_cdp_responses_and_tab_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(browser_module, "MAX_CDP_RESPONSE_BYTES", 32)

    def oversized(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 33)

    session = BrowserSession(9222, transport=httpx.MockTransport(oversized))
    with pytest.raises(RuntimeError, match="exceeds"):
        await session.connect()
    assert session.mode == BrowserMode.ISOLATED and not session.connected

    monkeypatch.setattr(browser_module, "MAX_CDP_RESPONSE_BYTES", 2 * 1024 * 1024)
    monkeypatch.setattr(browser_module, "MAX_CDP_TABS", 1)

    def too_many_tabs(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/json/version":
            return httpx.Response(200, json={"Browser": "Chrome/1"})
        return httpx.Response(
            200,
            json=[{"id": "one"}, {"id": "two"}],
        )

    session = BrowserSession(9222, transport=httpx.MockTransport(too_many_tabs))
    with pytest.raises(RuntimeError, match="more than 1 tabs"):
        await session.connect()
    assert session.mode == BrowserMode.ISOLATED and not session.connected


@pytest.mark.parametrize(
    "version_body",
    [
        b'{"Browser":"Chrome/1","Browser":"forged"}',
        b'{"Browser":"Chrome/1","value":NaN}',
        b'{"Browser":"Chrome/1","extra":[[[[[[[[[[[[[[[[[0]]]]]]]]]]]]]]]]]}',
        b'{"Browser":"\xff"}',
    ],
)
async def test_browser_connect_rejects_ambiguous_or_unbounded_json_atomically(
    version_body: bytes,
) -> None:
    def valid_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/json/version":
            return httpx.Response(200, json={"Browser": "Chrome/1"})
        return httpx.Response(200, json=[{"id": "existing", "title": "Before"}])

    session = BrowserSession(9222, transport=httpx.MockTransport(valid_handler))
    await session.connect()
    session.transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=version_body)
    )

    with pytest.raises(RuntimeError, match="invalid JSON"):
        await session.connect(9333)

    assert session.port == 9222
    assert session.connected and session.mode == BrowserMode.SHARED
    assert [tab.id for tab in session.tabs] == ["existing"]


async def test_browser_connect_validates_tab_fields_and_duplicate_ids_atomically() -> None:
    response_tabs: list[object] = [
        {"id": True, "title": "boolean"},
        {"id": "unsafe\nid", "title": "control"},
        {"id": "valid", "title": "Docs", "url": "https://example.com"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/json/version":
            return httpx.Response(200, json={"Browser": "Chrome/1"})
        return httpx.Response(200, json=response_tabs)

    session = BrowserSession(9222, transport=httpx.MockTransport(handler))
    assert [tab.id for tab in await session.connect()] == ["valid"]
    response_tabs[:] = [{"id": "same"}, {"id": "same"}]

    with pytest.raises(RuntimeError, match="duplicate tab IDs"):
        await session.connect(9333)

    assert session.port == 9222
    assert [tab.id for tab in session.tabs] == ["valid"]


async def test_browser_command_is_shared_and_validates_port() -> None:
    session = BrowserSession()
    assert "disconnected" in (await handle_browser_command("status", session)).lower()
    assert await handle_browser_command("connect nope", session) == (
        "Usage: /browser connect [port]"
    )
    assert "failed" in (await handle_browser_command("connect 80", session)).lower()
    assert session.port == 9222
    assert await handle_browser_command("tabs", session) == (
        "No browser tabs are currently visible."
    )
    assert await handle_browser_command("disconnect", session) == "Browser disconnected."


async def test_agent_browser_tools_share_transactional_cli_controls(tmp_path: Path) -> None:
    class Manager:
        def __init__(self) -> None:
            self.configs = {
                "chrome-devtools": SimpleNamespace(
                    args=["-y", "chrome-devtools-mcp@latest", "--isolated=true"]
                )
            }
            self.calls: list[list[str]] = []
            self.fail = False

        def status(self) -> dict[str, str]:
            return {"chrome-devtools": "RUNNING"}

        async def restart_with_args(
            self, name: str, args: list[str], on_success: object = None
        ) -> None:
            self.calls.append(list(args))
            if self.fail:
                raise RuntimeError("restart exploded")
            self.configs[name].args = list(args)
            if callable(on_success):
                on_success()

    session = BrowserSession()
    registry = ToolRegistry(
        tmp_path,
        approval_policy=ApprovalPolicy(False),
        browser_guard=BrowserGuard(session, SensitivePagePolicy()),
    )
    manager = Manager()
    agent = SimpleNamespace(tools=registry)
    _register_browser_agent_tools(agent, session, manager)  # type: ignore[arg-type]

    names = {item["function"]["name"] for item in registry.schemas()}
    assert {"browser_connect", "browser_disconnect", "browser_status"} <= names
    assert "autoConnect" in await registry.execute("browser_connect", {})
    assert session.mode == BrowserMode.SHARED
    assert manager.calls[-1][-1] == "--autoConnect"
    assert "shared" in await registry.execute("browser_status", {})
    manager.fail = True
    failed = await registry.execute("browser_disconnect", {})
    assert "restart exploded" in failed
    assert session.mode == BrowserMode.SHARED
    manager.fail = False
    assert "isolated" in await registry.execute("browser_disconnect", {})
    assert session.mode == BrowserMode.ISOLATED


async def test_browser_command_transactionally_restarts_mcp_and_clears_approvals(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/json/version":
            return httpx.Response(200, json={"Browser": "Chrome/1"})
        return httpx.Response(
            200,
            json=[
                {
                    "id": "page-1",
                    "title": "Docs",
                    "url": "https://example.com/docs",
                    "type": "page",
                }
            ],
        )

    class Manager:
        def __init__(self) -> None:
            self.configs = {
                "chrome-devtools": type(
                    "Config",
                    (),
                    {"args": ["-y", "chrome-devtools-mcp@latest", "--isolated=true"]},
                )()
            }
            self.calls: list[list[str]] = []
            self.fail = False

        def status(self) -> dict[str, str]:
            return {"chrome-devtools": "RUNNING"}

        async def restart_with_args(
            self,
            name: str,
            args: list[str],
            on_success: object = None,
        ) -> None:
            self.calls.append(list(args))
            if self.fail:
                raise RuntimeError("restart failed")
            self.configs[name].args = list(args)
            if callable(on_success):
                on_success()

    session = BrowserSession(9222, transport=httpx.MockTransport(handler))
    manager = Manager()
    policy = ApprovalPolicy(True)
    policy.remember("mcp__chrome-devtools__click", ApprovalResult.approve_all())
    policy.remember(
        "mcp__chrome-devtools__take_snapshot",
        ApprovalResult.approve_all_by_server(),
    )
    tools = ToolRegistry(tmp_path)

    result = await handle_browser_command(
        "connect 9222",
        session,
        mcp_manager=manager,
        approval_policy=policy,
        tools=tools,
    )

    assert "Connected chrome-devtools MCP" in result
    assert session.mode == BrowserMode.SHARED
    assert session.browser_url == "http://127.0.0.1:9222"
    assert [tab.id for tab in session.tabs] == ["page-1"]
    assert manager.calls[-1] == [
        "-y",
        "chrome-devtools-mcp@latest",
        "--browser-url=http://127.0.0.1:9222",
    ]
    assert policy.needs_approval("mcp__chrome-devtools__click")
    assert policy.needs_approval("mcp__chrome-devtools__take_snapshot")

    policy.remember("mcp__chrome-devtools__click", ApprovalResult.approve_all_by_server())
    manager.fail = True
    failed = await handle_browser_command(
        "disconnect",
        session,
        mcp_manager=manager,
        approval_policy=policy,
        tools=tools,
    )
    assert "restart failed" in failed
    assert session.mode == BrowserMode.SHARED
    assert not policy.needs_approval("mcp__chrome-devtools__click")

    manager.fail = False
    disconnected = await handle_browser_command(
        "disconnect",
        session,
        mcp_manager=manager,
        approval_policy=policy,
        tools=tools,
    )
    assert "isolated" in disconnected
    assert session.mode == BrowserMode.ISOLATED
    assert manager.calls[-1][-1] == "--isolated=true"
    assert policy.needs_approval("mcp__chrome-devtools__click")

    auto_connected = await handle_browser_command(
        "connect",
        session,
        mcp_manager=manager,
        approval_policy=policy,
        tools=tools,
    )
    assert "autoConnect" in auto_connected
    assert session.mode == BrowserMode.SHARED
    assert session.browser_url == "autoConnect"
    assert manager.calls[-1][-1] == "--autoConnect"


def test_browser_guard_tracks_navigation_and_agent_owned_tabs(tmp_path: Path) -> None:
    session = BrowserSession(mode=BrowserMode.SHARED)
    guard = BrowserGuard(session, SensitivePagePolicy(tmp_path / "missing"))
    guard.apply_after_execution(
        "mcp__chrome-devtools__new_page",
        {"url": "https://example.com"},
        "created page-7",
    )
    assert session.last_navigated_url == "https://example.com"
    assert session.is_agent_opened_tab("page-7")
    assert not guard.check("mcp__chrome-devtools__close_page", {"pageIdx": "page-7"}).blocked
    blocked = guard.check("mcp__chrome-devtools__close_page", {"pageIdx": "existing"})
    assert blocked.blocked and "not opened by Kairo CLI" in blocked.reason


async def test_failed_browser_tools_do_not_commit_session_mutations(
    tmp_path: Path,
) -> None:
    session = BrowserSession(mode=BrowserMode.SHARED)
    session.remember_navigation("https://before.example")
    session.record_opened_tab("page-7")
    registry = ToolRegistry(
        tmp_path,
        browser_guard=BrowserGuard(session, SensitivePagePolicy(tmp_path / "missing")),
    )

    async def fail(arguments: dict[str, object]) -> ToolOutput:
        return ToolOutput(json.dumps({"error": "browser operation failed"}))

    registry.register(
        ToolDefinition(
            "mcp__chrome-devtools__navigate_page",
            "navigate",
            {"type": "object"},
            fail,
        )
    )
    registry.register(
        ToolDefinition(
            "mcp__chrome-devtools__close_page",
            "close",
            {"type": "object"},
            fail,
        )
    )

    await registry.execute("mcp__chrome-devtools__navigate_page", {"url": "https://after.example"})
    await registry.execute("mcp__chrome-devtools__close_page", {"pageIdx": "page-7"})

    assert session.last_navigated_url == "https://before.example"
    assert session.is_agent_opened_tab("page-7")


async def test_sensitive_browser_writes_ignore_approve_all_cache(tmp_path: Path) -> None:
    rules = tmp_path / "rules.txt"
    rules.write_text("*://example.com/admin/*\n", encoding="utf-8")
    session = BrowserSession(mode=BrowserMode.SHARED)
    session.remember_navigation("https://example.com/admin/users")
    guard = BrowserGuard(session, SensitivePagePolicy(rules))
    approvals = 0
    notices: list[str] = []

    async def approve(name: str, arguments: dict[str, object]) -> ApprovalResult:
        nonlocal approvals
        approvals += 1
        notice = arguments.get("_kairocli_approval_notice")
        if notice:
            notices.append(str(notice))
        return ApprovalResult.approve_all_by_server()

    registry = ToolRegistry(
        tmp_path,
        approval_policy=ApprovalPolicy(True),
        approver=approve,
        browser_guard=guard,
    )

    async def ok(arguments: dict[str, object]) -> str:
        return "ok"

    registry.register(
        ToolDefinition("mcp__chrome-devtools__take_snapshot", "read", {"type": "object"}, ok)
    )
    registry.register(
        ToolDefinition("mcp__chrome-devtools__click", "write", {"type": "object"}, ok)
    )
    await registry.execute("mcp__chrome-devtools__take_snapshot", {})
    await registry.execute("mcp__chrome-devtools__click", {"uid": "1"})
    await registry.execute("mcp__chrome-devtools__click", {"uid": "2"})
    assert approvals == 3
    assert len(notices) == 2
    assert all("Sensitive page" in notice for notice in notices)


async def test_parallel_navigation_commits_before_sensitive_write_approval(
    tmp_path: Path,
) -> None:
    rules = tmp_path / "rules.txt"
    rules.write_text("*://example.com/admin/*\n", encoding="utf-8")
    session = BrowserSession(mode=BrowserMode.SHARED)
    session.remember_navigation("https://example.com/docs")
    approvals: list[tuple[str, str]] = []
    navigation_entered = asyncio.Event()
    release_navigation = asyncio.Event()
    click_executed = asyncio.Event()

    async def approve(name: str, arguments: dict[str, object]) -> ApprovalResult:
        approvals.append((name, str(arguments.get("_kairocli_approval_notice", ""))))
        if name.endswith("navigate_page"):
            return ApprovalResult.approve_all_by_server()
        return ApprovalResult.approve()

    registry = ToolRegistry(
        tmp_path,
        approval_policy=ApprovalPolicy(True),
        approver=approve,
        browser_guard=BrowserGuard(session, SensitivePagePolicy(rules)),
    )

    async def navigate(arguments: dict[str, object]) -> str:
        navigation_entered.set()
        await release_navigation.wait()
        return "navigated"

    async def click(arguments: dict[str, object]) -> str:
        click_executed.set()
        return "clicked"

    registry.register(
        ToolDefinition(
            "mcp__chrome-devtools__navigate_page",
            "navigate",
            {"type": "object", "properties": {"url": {"type": "string"}}},
            navigate,
        )
    )
    registry.register(
        ToolDefinition(
            "mcp__chrome-devtools__click",
            "click",
            {"type": "object", "properties": {"uid": {"type": "string"}}},
            click,
        )
    )

    navigation_task = asyncio.create_task(
        registry.execute(
            "mcp__chrome-devtools__navigate_page",
            {"url": "https://example.com/admin/users"},
        )
    )
    await asyncio.wait_for(navigation_entered.wait(), timeout=1)
    click_task = asyncio.create_task(
        registry.execute("mcp__chrome-devtools__click", {"uid": "submit"})
    )
    await asyncio.sleep(0)

    assert not click_executed.is_set()
    assert [name for name, _ in approvals] == ["mcp__chrome-devtools__navigate_page"]

    release_navigation.set()
    await asyncio.gather(navigation_task, click_task)

    assert session.last_navigated_url == "https://example.com/admin/users"
    assert [name for name, _ in approvals] == [
        "mcp__chrome-devtools__navigate_page",
        "mcp__chrome-devtools__click",
    ]
    assert "Sensitive page" in approvals[1][1]
    assert click_executed.is_set()


async def test_waiting_for_browser_operation_is_cancelable(tmp_path: Path) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    second_executed = asyncio.Event()
    session = BrowserSession(mode=BrowserMode.SHARED)
    session.remember_navigation("https://example.com/docs")
    registry = ToolRegistry(
        tmp_path,
        browser_guard=BrowserGuard(session, SensitivePagePolicy(tmp_path / "missing-rules")),
    )

    async def first_handler(arguments: dict[str, object]) -> str:
        entered.set()
        await release.wait()
        return "first"

    async def second_handler(arguments: dict[str, object]) -> str:
        second_executed.set()
        return "second"

    registry.register(
        ToolDefinition(
            "mcp__chrome-devtools__take_snapshot",
            "first",
            {"type": "object", "properties": {}},
            first_handler,
        )
    )
    registry.register(
        ToolDefinition(
            "mcp__chrome-devtools__list_pages",
            "second",
            {"type": "object", "properties": {}},
            second_handler,
        )
    )

    first = asyncio.create_task(registry.execute("mcp__chrome-devtools__take_snapshot", {}))
    await asyncio.wait_for(entered.wait(), timeout=1)
    canceled = asyncio.Event()
    second = asyncio.create_task(registry.execute("mcp__chrome-devtools__list_pages", {}, canceled))
    await asyncio.sleep(0)
    canceled.set()

    with pytest.raises(AgentCanceled):
        await asyncio.wait_for(second, timeout=1)
    assert not second_executed.is_set()
    assert not session.state_uncertain

    release.set()
    await first


async def test_canceled_navigation_forces_fresh_browser_write_approval(
    tmp_path: Path,
) -> None:
    session = BrowserSession(mode=BrowserMode.SHARED)
    session.remember_navigation("https://example.com/docs")
    entered = asyncio.Event()
    approvals: list[tuple[str, str]] = []

    async def approve(name: str, arguments: dict[str, object]) -> ApprovalResult:
        approvals.append((name, str(arguments.get("_kairocli_approval_notice", ""))))
        if name.endswith("navigate_page"):
            return ApprovalResult.approve_all_by_server()
        return ApprovalResult.approve()

    registry = ToolRegistry(
        tmp_path,
        approval_policy=ApprovalPolicy(True),
        approver=approve,
        browser_guard=BrowserGuard(session, SensitivePagePolicy(tmp_path / "missing-rules")),
    )

    async def navigate(arguments: dict[str, object]) -> str:
        entered.set()
        await asyncio.Event().wait()
        return "unreachable"

    async def click(arguments: dict[str, object]) -> str:
        return "clicked"

    registry.register(
        ToolDefinition(
            "mcp__chrome-devtools__navigate_page",
            "navigate",
            {"type": "object", "properties": {"url": {"type": "string"}}},
            navigate,
        )
    )
    registry.register(
        ToolDefinition(
            "mcp__chrome-devtools__click",
            "click",
            {"type": "object", "properties": {"uid": {"type": "string"}}},
            click,
        )
    )

    canceled = asyncio.Event()
    navigation = asyncio.create_task(
        registry.execute(
            "mcp__chrome-devtools__navigate_page",
            {"url": "https://example.com/admin/users"},
            canceled,
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1)
    canceled.set()
    with pytest.raises(AgentCanceled):
        await asyncio.wait_for(navigation, timeout=1)

    assert session.last_navigated_url == "https://example.com/docs"
    assert session.state_uncertain

    await registry.execute("mcp__chrome-devtools__click", {"uid": "submit"})
    assert [name for name, _ in approvals] == [
        "mcp__chrome-devtools__navigate_page",
        "mcp__chrome-devtools__click",
    ]
    assert "state is uncertain" in approvals[1][1]


async def test_sensitive_browser_write_without_approver_is_denied(tmp_path: Path) -> None:
    rules = tmp_path / "rules.txt"
    rules.write_text("*://example.com/admin/*\n", encoding="utf-8")
    session = BrowserSession()
    session.remember_navigation("https://example.com/admin/users")
    registry = ToolRegistry(
        tmp_path,
        browser_guard=BrowserGuard(session, SensitivePagePolicy(rules)),
    )

    async def unreachable(arguments: dict[str, object]) -> str:
        raise AssertionError("sensitive tool should not execute")

    registry.register(
        ToolDefinition(
            "mcp__chrome-devtools__evaluate_script",
            "write",
            {"type": "object"},
            unreachable,
        )
    )
    result = json.loads(await registry.execute("mcp__chrome-devtools__evaluate_script", {}))
    assert result["approval_denied"] is True


async def test_browser_mcp_audit_strips_url_query(tmp_path: Path) -> None:
    class RecordingAudit:
        def __init__(self) -> None:
            self.entries: list[tuple[str, str, str]] = []

        def append(
            self,
            tool: str,
            decision: str,
            arguments: dict[str, object],
            detail: str = "",
        ) -> None:
            self.entries.append((tool, decision, detail))

    audit = RecordingAudit()
    session = BrowserSession()
    guard = BrowserGuard(session, SensitivePagePolicy(tmp_path / "missing"))
    registry = ToolRegistry(tmp_path, audit=audit, browser_guard=guard)  # type: ignore[arg-type]

    async def navigate(arguments: dict[str, object]) -> str:
        return "ok"

    registry.register(
        ToolDefinition(
            "mcp__chrome-devtools__navigate_page",
            "navigate",
            {"type": "object"},
            navigate,
        )
    )
    await registry.execute(
        "mcp__chrome-devtools__navigate_page",
        {"url": "https://example.com/path?token=secret#section"},
    )
    assert audit.entries[0][0] == "mcp__chrome-devtools__navigate_page"
    assert audit.entries[0][1] == "allowed"
    assert "https://example.com/path" in audit.entries[0][2]
    assert "secret" not in audit.entries[0][2]
