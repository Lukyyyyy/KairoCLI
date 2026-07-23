"""Argument parsing for the Kairo CLI command-line interface."""

from __future__ import annotations

import argparse

from ..brand import VERSION
from ..config import PROVIDER_ALIASES, PROVIDER_DEFAULTS


def runtime_port(value: str) -> int:
    try:
        port = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer from 1 to 65535") from exc
    if not 1 <= port <= 65_535:
        raise argparse.ArgumentTypeError("port must be an integer from 1 to 65535")
    return port


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kairocli", description="Kairo CLI agent runtime")
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    parser.add_argument("--provider", choices=sorted({*PROVIDER_DEFAULTS, *PROVIDER_ALIASES}))
    session_group = parser.add_mutually_exclusive_group()
    session_group.add_argument(
        "--continue", dest="continue_session", action="store_true", help="resume latest session"
    )
    session_group.add_argument("--resume", metavar="SESSION_ID", help="resume a session by ID")
    parser.add_argument(
        "-p",
        "--print",
        dest="print_prompt",
        nargs="?",
        const="",
        metavar="PROMPT",
        help="run one noninteractive turn; omit PROMPT to read stdin",
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "json", "jsonl"],
        default="text",
        help="noninteractive stdout format",
    )
    parser.add_argument("--mode", choices=["agent", "plan", "team"], default="agent")
    parser.add_argument(
        "--allow-tool",
        action="append",
        default=[],
        metavar="NAME",
        help="allow one approval-gated tool in noninteractive mode; repeatable",
    )
    parser.add_argument(
        "--dangerously-skip-approvals",
        action="store_true",
        help="allow every approval-gated tool in noninteractive mode",
    )
    parser.add_argument(
        "--save-session",
        action="store_true",
        help="persist a new noninteractive session (resume flags already persist)",
    )
    subparsers = parser.add_subparsers(dest="subcommand")
    serve = subparsers.add_parser("serve", help="start the local Runtime API")
    serve.add_argument("--http", action="store_true", required=True)
    serve.add_argument("--port", type=runtime_port, default=8080)
    wechat = subparsers.add_parser("wechat", help="manage the WeChat channel")
    wechat.set_defaults(daemon_action=None)
    wechat_actions = wechat.add_subparsers(dest="action", required=True)
    wechat_actions.add_parser("setup", help="bind a WeChat account")
    wechat_actions.add_parser("start", help="run the WeChat channel in foreground")
    wechat_actions.add_parser("status", help="show the current WeChat binding")
    wechat_daemon = wechat_actions.add_parser("daemon", help="manage the background WeChat channel")
    wechat_daemon.add_argument(
        "daemon_action", nargs="?", choices=["start", "stop", "restart", "status", "logs"]
    )
    return parser
