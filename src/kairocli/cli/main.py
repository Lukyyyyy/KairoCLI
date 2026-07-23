from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from ..brand import PRODUCT_NAME
from ..config import (
    AppConfig,
)
from ..diagnostics import configure_application_logging
from ..mcp import (
    McpServerManager,
    ensure_default_mcp_config,
)
from ..paths import KairoPaths
from ..policy import ApprovalPolicy
from ..sessions import SessionStore, apply_session
from ..skills import SkillRegistry
from ..tasks import DurableTaskManager, DurableTaskStore
from ..terminal_capabilities import detect_renderer_mode
from ..todos import SessionTodoController
from .bootstrap import _inject_mcp_resource_index, _register_browser_agent_tools, make_agent
from .interactive import (
    _close_components,
    _console,
    _print_status,
    _resume_session_hint,
    _safe_cli_error,
    interactive,
)
from .noninteractive import (
    _emit_noninteractive_error,
    _noninteractive_exception_text,
    noninteractive,
)
from .parser import build_parser
from .wechat_ui import QR_SCAN_PROMPT, workspace_prompt

log = logging.getLogger(__name__)
MAX_INTERACTIVE_ERROR_BYTES = 4_000


def run_server(paths: KairoPaths, config: AppConfig, provider: str | None, port: int) -> int:
    if type(port) is not int or not 1 <= port <= 65_535:
        raise ValueError("Runtime API port must be an integer from 1 to 65535")
    from ..runtime_api import create_app

    api_key = os.getenv("KAIROCLI_RUNTIME_API_KEY", "")
    app = create_app(
        lambda: make_agent(paths, config, provider),
        api_key,
        paths.runtime_dir / "runtime.db",
    )
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Install Kairo CLI dependencies to run the server") from exc
    uvicorn.run(app, host="127.0.0.1", port=port)
    return 0


async def handle_wechat(
    paths: KairoPaths,
    config: AppConfig,
    action: str,
    daemon_action: str | None,
) -> int:
    from datetime import UTC, datetime

    from ..wechat import (
        IlinkClient,
        WechatAccount,
        WechatAccountStore,
        WechatChannel,
        WechatPolicy,
        daemon_command,
    )

    store = WechatAccountStore(paths)
    if action == "daemon":
        print(daemon_command(paths, daemon_action or "status"))
        return 0
    if action == "status":
        account = store.load()
        if account is None:
            print("WeChat channel is not bound. Run `kairocli wechat setup`.")
        else:
            masked = _mask_secret(account.bound_user_id)
            print(f"WeChat channel bound\nAccount: {account.account_id}\nUser: {masked}")
            print(f"Workspace: {account.workspace}")
        return 0
    client = IlinkClient()
    if action == "setup":
        entered = await asyncio.to_thread(input, workspace_prompt(paths.workspace))
        workspace = await asyncio.to_thread(
            lambda: Path(entered.strip() or paths.workspace).expanduser().resolve()
        )
        if not await asyncio.to_thread(workspace.is_dir):
            raise ValueError(f"Workspace does not exist: {workspace}")
        login = await client.start_qr_login()
        print(QR_SCAN_PROMPT)
        try:
            import qrcode  # type: ignore[import-untyped]

            qr = qrcode.QRCode(border=1)
            qr.add_data(login.qrcode_url)
            qr.print_ascii(invert=True)
        except Exception:
            pass
        print(login.qrcode_url)
        deadline = asyncio.get_running_loop().time() + 300
        result = None
        while asyncio.get_running_loop().time() < deadline:
            result = await client.poll_qr_status(login.qrcode_id)
            if result.connected or result.expired:
                break
            await asyncio.sleep(3)
        if result is None or not result.connected:
            raise RuntimeError(
                f"WeChat binding did not complete: {result.message if result else 'timeout'}"
            )
        account = WechatAccount(
            result.token,
            result.account_id,
            result.base_url,
            result.user_id,
            str(workspace),
            created_at=datetime.now(UTC).isoformat(),
        )
        store.save(account)
        print(f"WeChat channel bound. Account: {account.account_id}\nWorkspace: {workspace}")
        return 0
    account = store.load()
    if account is None:
        raise RuntimeError("No WeChat account is bound; run `kairocli wechat setup` first")
    channel_paths = KairoPaths.discover(Path(account.workspace), paths.home)
    channel_config = AppConfig.load(channel_paths)
    policy = WechatPolicy()

    async def channel_approval(name: str, arguments: dict[str, Any]) -> bool:
        return policy.allow_tool(name, arguments)

    channel_agent = make_agent(
        channel_paths,
        channel_config,
        approval_policy=ApprovalPolicy(True),
        approver=channel_approval,
    )
    mcp_manager = McpServerManager(channel_paths, channel_agent.tools)
    await mcp_manager.start_all()
    _inject_mcp_resource_index(channel_agent, mcp_manager)
    try:
        await WechatChannel(client, store, account, channel_agent).run()
    finally:
        await _close_components(
            ("MCP", mcp_manager),
            ("Tool", channel_agent.tools),
        )
    return 0


def _mask_secret(value: str) -> str:
    if len(value) < 8:
        return "***"
    return value[:4] + "***" + value[-4:]


def _silence_broken_pipe(stream: Any) -> None:
    """Prevent interpreter shutdown from flushing a pipe that the reader closed."""
    try:
        descriptor = stream.fileno()
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, descriptor)
        finally:
            os.close(devnull)
    except (AttributeError, OSError, ValueError):
        return


def main(argv: list[str] | None = None) -> None:
    try:
        args = build_parser().parse_args(argv)
    except BrokenPipeError:
        _silence_broken_pipe(sys.stdout)
        raise SystemExit(0) from None
    batch_requested = args.print_prompt is not None
    try:
        paths = KairoPaths.discover()
        configure_application_logging(paths)
        log.info("cli_start argv_count=%d", len(argv or sys.argv[1:]))
        config = AppConfig.load(paths, {"default_provider": args.provider})
        terminal_size = shutil.get_terminal_size(fallback=(80, 24))
        renderer = detect_renderer_mode(
            config.renderer,
            sys.stdin,
            sys.stdout,
            os.environ,
            columns=terminal_size.columns,
            rows=terminal_size.lines,
        )
        batch_only_flags = bool(
            args.output_format != "text"
            or args.mode != "agent"
            or args.allow_tool
            or args.dangerously_skip_approvals
            or args.save_session
        )
        if batch_only_flags and not batch_requested:
            raise ValueError(
                "--output-format/--mode/--allow-tool/--dangerously-skip-approvals/"
                "--save-session require --print"
            )
        if args.allow_tool and args.dangerously_skip_approvals:
            raise ValueError("--allow-tool cannot be combined with --dangerously-skip-approvals")
        if args.subcommand and batch_requested:
            raise ValueError("--print cannot be combined with a subcommand")
        if args.subcommand and (args.resume or args.continue_session):
            raise ValueError(
                "--resume/--continue are available only in interactive or --print modes"
            )
        if batch_requested:
            code = asyncio.run(
                noninteractive(
                    paths,
                    config,
                    args.print_prompt,
                    provider=args.provider,
                    output_format=args.output_format,
                    mode=args.mode,
                    resume_id=args.resume,
                    continue_session=args.continue_session,
                    save_session=args.save_session,
                    allowed_tools=args.allow_tool,
                    bypass_approvals=args.dangerously_skip_approvals,
                )
            )
        elif args.subcommand == "serve":
            code = run_server(paths, config, args.provider, args.port)
        elif args.subcommand == "wechat":
            code = asyncio.run(handle_wechat(paths, config, args.action, args.daemon_action))
        elif renderer.mode == "tui":
            from ..tui import run_tui

            try:
                bootstrap_notice = ensure_default_mcp_config(paths).message
            except (OSError, ValueError) as exc:
                bootstrap_notice = "MCP bootstrap warning: " + _safe_cli_error(exc)

            store = SessionStore(paths.session_database)
            state = (
                store.load(args.resume, paths.workspace)
                if args.resume
                else store.latest(paths.workspace)
                if args.continue_session
                else None
            )
            if args.resume and state is None:
                raise ValueError("Session was not found in the current workspace")
            selected_provider = args.provider or (
                state.meta.provider if state is not None else config.default_provider
            )
            todo_controller = SessionTodoController(store, paths.workspace)
            tui_skills = SkillRegistry(paths)
            tui_skills.reload()
            tui_agent = make_agent(
                paths,
                config,
                selected_provider,
                skill_registry=tui_skills,
                todo_controller=todo_controller,
            )
            if state is None:
                state = store.create(paths.workspace, tui_agent.llm.provider, tui_agent.llm.model)
            else:
                apply_session(tui_agent, state)
            tui_mcp_manager = McpServerManager(paths, tui_agent.tools)
            tui_browser_guard = tui_agent.tools.browser_guard
            if tui_browser_guard is not None:
                _register_browser_agent_tools(tui_agent, tui_browser_guard.session, tui_mcp_manager)
            tui_task_store = DurableTaskStore(paths.task_database)
            tui_task_manager = DurableTaskManager(
                tui_task_store,
                lambda: make_agent(paths, config, selected_provider),
                config.task_workers,
            )
            run_tui(
                tui_agent,
                session_store=store,
                session_id=state.meta.id,
                workspace=paths.workspace,
                todo_controller=todo_controller,
                mcp_manager=tui_mcp_manager,
                task_store=tui_task_store,
                task_manager=tui_task_manager,
                skill_registry=tui_skills,
                app_config=config,
                startup_notice="\n".join(
                    item for item in (renderer.notice, bootstrap_notice) if item
                ),
            )
            _print_status(_console(), "\n" + _resume_session_hint(state.meta.id), "dim")
            code = 0
        else:
            try:
                bootstrap_notice = ensure_default_mcp_config(paths).message
            except (OSError, ValueError) as exc:
                bootstrap_notice = "MCP bootstrap warning: " + _safe_cli_error(exc)
            code = asyncio.run(
                interactive(
                    paths,
                    config,
                    args.provider,
                    args.resume,
                    args.continue_session,
                    "\n".join(item for item in (renderer.notice, bootstrap_notice) if item),
                    renderer.mode,
                )
            )
    except KeyboardInterrupt:
        log.info("cli_interrupted batch=%s", batch_requested)
        if batch_requested:
            code = _emit_noninteractive_error(
                args.output_format,
                "Canceled",
                "Interrupted",
                time.monotonic(),
                status="canceled",
            )
            code = 130
        else:
            print(f"{PRODUCT_NAME}: interrupted", file=sys.stderr)
            code = 130
    except BrokenPipeError:
        log.info("cli_broken_pipe")
        _silence_broken_pipe(sys.stdout)
        code = 0
    except (OSError, ValueError, RuntimeError) as exc:
        log.warning("cli_error type=%s detail=%s", type(exc).__name__, _safe_cli_error(exc))
        if batch_requested:
            code = _emit_noninteractive_error(
                args.output_format,
                type(exc).__name__,
                _noninteractive_exception_text(exc),
                time.monotonic(),
            )
            code = 2
        else:
            print(f"{PRODUCT_NAME}: {_safe_cli_error(exc)}", file=sys.stderr)
            code = 2
    except Exception as exc:
        log.error("cli_unexpected_error type=%s", type(exc).__name__)
        if batch_requested:
            code = _emit_noninteractive_error(
                args.output_format,
                type(exc).__name__,
                _noninteractive_exception_text(exc),
                time.monotonic(),
            )
        else:
            message = _safe_cli_error(exc) or "Unexpected runtime failure"
            print(f"{PRODUCT_NAME}: {message}", file=sys.stderr)
            code = 1
    raise SystemExit(code)
