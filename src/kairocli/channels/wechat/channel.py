from __future__ import annotations

import asyncio
import hashlib
import secrets
import threading
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Protocol

from ...agent import Agent
from ...cancellation import AgentCanceled
from ...channels.wechat.formatting import (
    format_wechat_text as format_wechat_text,
)
from ...channels.wechat.formatting import (
    split_message as split_message,
)
from ...runtime_api import RuntimeThreadStore
from ...trace import safe_redacted_text
from .accounts import WechatAccount, WechatMessage
from .client import IlinkClient

DEFAULT_BASE_URL = "https://ilinkai.weixin.qq.com"
MAX_WECHAT_ACCOUNT_BYTES = 128 * 1024
MAX_WECHAT_REQUEST_BYTES = 1024 * 1024
MAX_WECHAT_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_WECHAT_API_JSON_DEPTH = 32
MAX_WECHAT_API_JSON_NODES = 200_000
MAX_WECHAT_MESSAGE_CHARS = 100_000
MAX_WECHAT_MESSAGES_PER_UPDATE = 1_000
MAX_WECHAT_DAEMON_LOG_BYTES = 5 * 1024 * 1024
MAX_WECHAT_ACCOUNT_JSON_DEPTH = 16
MAX_WECHAT_ACCOUNT_JSON_NODES = 10_000
_WECHAT_ACCOUNT_THREAD_LOCK = threading.RLock()


@dataclass(slots=True)
class WechatPolicy:
    command_allowlist: tuple[str, ...] = ()
    mcp_allowlist: tuple[str, ...] = ()

    def allow_tool(self, name: str, arguments: dict[str, object]) -> bool:
        read_only = {
            "read_file",
            "list_dir",
            "glob_files",
            "grep_code",
            "search_code",
            "web_search",
            "web_fetch",
            "load_skill",
            "browser_status",
        }
        if name in read_only:
            return True
        if name in {"write_file", "create_project"}:
            return True
        if name in {"revert_turn", "browser_connect", "browser_disconnect"}:
            return False
        if name == "execute_command":
            return str(arguments.get("command", "")).strip() in self.command_allowlist
        if name.startswith("mcp__"):
            return any(
                name == allowed or name.startswith(f"mcp__{allowed}__")
                for allowed in self.mcp_allowlist
                if allowed.strip()
            )
        return False


class WechatApprovalHandler:
    TIMEOUT_SECONDS = 300.0

    def __init__(self) -> None:
        self.sender: Callable[[WechatMessage, str], Awaitable[None]] | None = None
        self.message: WechatMessage | None = None
        self._event: asyncio.Event | None = None
        self._code = ""
        self._approved = False

    async def __call__(self, name: str, arguments: dict[str, object]) -> bool:
        if WechatPolicy().allow_tool(name, arguments) and name not in {
            "write_file",
            "apply_patch",
            "create_project",
        }:
            return True
        if name not in {"write_file", "apply_patch", "create_project"}:
            return False
        if self.sender is None or self.message is None or self._event is not None:
            return False
        self._code = secrets.token_hex(2).upper()
        self._approved = False
        self._event = asyncio.Event()
        path = str(arguments.get("path", "")).strip()
        target = f" · {path[:500]}" if path else ""
        await self.sender(
            self.message,
            f"工具请求：{name}{target}\n"
            f"回复 /approve {self._code} 或 /deny {self._code}，5 分钟内有效。",
        )
        try:
            await asyncio.wait_for(self._event.wait(), timeout=self.TIMEOUT_SECONDS)
            return self._approved
        except TimeoutError:
            return False
        finally:
            self._event = None
            self._code = ""

    async def respond(self, message: WechatMessage) -> bool:
        command, _, code = message.text.strip().partition(" ")
        if command.casefold() not in {"/approve", "/deny"}:
            return False
        if self._event is None or not secrets.compare_digest(code.strip().upper(), self._code):
            if self.sender is not None:
                await self.sender(message, "没有匹配的待审批请求。")
            return True
        self._approved = command.casefold() == "/approve"
        self._event.set()
        if self.sender is not None:
            await self.sender(message, "已批准。" if self._approved else "已拒绝。")
        return True


class WechatSyncStore(Protocol):
    def update_sync_buf(self, expected: WechatAccount, sync_buf: str) -> bool: ...


class WechatChannel:
    def __init__(
        self,
        client: IlinkClient,
        store: WechatSyncStore,
        account: WechatAccount,
        agent: Agent,
        approval: WechatApprovalHandler | None = None,
        runtime_store: RuntimeThreadStore | None = None,
        runtime_thread_id: str = "",
        owner_user_id: str = "",
        workspace_switcher: Callable[[str], Awaitable[tuple[Agent | None, str, str, str]]]
        | None = None,
    ) -> None:
        self.client = client
        self.store = store
        self.account = account
        self.agent = agent
        self.approval = approval
        if approval is not None:
            approval.sender = self.send
        self.runtime_store = runtime_store
        self.runtime_thread_id = runtime_thread_id
        self.owner_user_id = owner_user_id
        self.workspace_switcher = workspace_switcher
        self.running = True
        self.paused = False
        self.seen: set[str] = set()
        self.seen_order: deque[str] = deque()
        self.queue: deque[tuple[WechatMessage, int]] = deque()
        self.max_queue = 100
        self.active_task: asyncio.Task[None] | None = None

    async def run(self) -> None:
        try:
            await self.client.notify(self.account, True)
        except Exception:
            pass
        timeout_ms = 35_000
        try:
            await self._drain_durable_messages()
            while self.running:
                await self._reap_active()
                poll_timeout = min(timeout_ms, 3_000) if self.active_task else timeout_ms
                try:
                    update = await self.client.get_updates(self.account, poll_timeout)
                except Exception:
                    await asyncio.sleep(2)
                    continue
                if update.code:
                    await asyncio.sleep(2)
                    continue
                if update.timeout_ms > 0:
                    timeout_ms = update.timeout_ms
                durable = getattr(self.store, "save_update", None)
                if durable is not None:
                    if not durable(self.account, update.sync_buf, update.messages):
                        self.running = False
                        break
                    self.account = replace(self.account, sync_buf=update.sync_buf)
                    await self._drain_durable_messages()
                elif update.sync_buf != self.account.sync_buf:
                    updated = replace(self.account, sync_buf=update.sync_buf)
                    if not self.store.update_sync_buf(self.account, update.sync_buf):
                        self.running = False
                        break
                    self.account = updated
                if durable is None:
                    for message in update.messages:
                        await self.handle(message)
                await self._reap_active()
                await self._start_next()
                await self._drain_durable_messages()
        finally:
            self.agent.cancel()
            if self.active_task is not None:
                self.active_task.cancel()
                await asyncio.gather(self.active_task, return_exceptions=True)
                self.active_task = None
            try:
                await self.client.notify(self.account, False)
            except Exception:
                pass

    async def _drain_durable_messages(self) -> None:
        claim = getattr(self.store, "claim_message", None)
        if claim is None:
            return
        while len(self.queue) < self.max_queue:
            item = claim()
            if item is None:
                return
            inbox_id, message = item
            queued = await self.handle(message, inbox_id)
            if not queued:
                finish = getattr(self.store, "finish_message", None)
                if finish is not None:
                    finish(inbox_id, "completed")

    async def handle(self, message: WechatMessage, inbox_id: int = 0) -> bool:
        if not message.from_user_id or message.from_user_id != self.account.bound_user_id:
            return False
        if message.message_id and message.message_id in self.seen:
            return False
        if message.message_id:
            self.seen.add(message.message_id)
            self.seen_order.append(message.message_id)
            while len(self.seen_order) > 2_000:
                self.seen.discard(self.seen_order.popleft())
        if self.approval is not None and await self.approval.respond(message):
            return False
        command, _, _argument = message.text.strip().partition(" ")
        command = command.casefold()
        if command == "/help":
            await self.send(
                message,
                "/help /status /clear /compact /model /cwd /send /pause /resume /stop\n"
                "/workspace [list|序号|名称|完整路径]\n"
                "execute_command 和 MCP 默认拒绝；写入仅限绑定 workspace。",
            )
            return False
        if command == "/status":
            await self.send(
                message,
                f"Kairo CLI channel: {'paused' if self.paused else 'running'}\n"
                f"Queue: {len(self.queue)}\n"
                f"Agent: {'running' if self.active_task else 'idle'}",
            )
            return False
        if command == "/pause":
            self.paused = True
            await self.send(message, "Kairo CLI channel paused; ordinary messages will queue.")
            return False
        if command == "/resume":
            self.paused = False
            await self.send(message, "Kairo CLI channel resumed.")
            await self._start_next()
            return False
        if command in {"/stop", "/cancel"}:
            self.agent.cancel()
            await self.send(message, "Cancellation requested.")
            return False
        queued_commands = {
            "/clear",
            "/compact",
            "/model",
            "/cwd",
            "/send",
            "/workspace",
        }
        if command.startswith("/") and command not in queued_commands:
            await self.send(message, f"Unknown WeChat command: {command}\nSend /help for commands.")
            return False
        if len(self.queue) >= self.max_queue:
            await self.send(message, "Kairo CLI channel queue is full; retry later.")
            return False
        self.queue.append((message, inbox_id))
        await self._start_next()
        return True

    async def _start_next(self) -> None:
        if self.paused or self.active_task is not None or not self.queue:
            return
        message, inbox_id = self.queue.popleft()
        self.active_task = asyncio.create_task(self._process(message, inbox_id))

    async def _reap_active(self) -> None:
        if self.active_task is None or not self.active_task.done():
            return
        await asyncio.gather(self.active_task, return_exceptions=True)
        self.active_task = None
        await self._start_next()

    async def _process(self, message: WechatMessage, inbox_id: int = 0) -> None:
        command, _, argument = message.text.strip().partition(" ")
        command = command.casefold()
        inbox_status = "completed"
        finish = getattr(self.store, "finish_message", None)

        def finish_command() -> None:
            if inbox_id and finish is not None:
                finish(inbox_id, "completed")

        if command == "/clear":
            if self.runtime_store is not None and self.runtime_thread_id:
                self.runtime_store.clear_thread(self.runtime_thread_id, self.owner_user_id)
            self.agent.clear()
            await self.send(message, "Conversation cleared; long-term memory retained.")
            finish_command()
            return
        if command == "/compact":
            compacted = await self.agent.compact()
            await self.send(
                message, "Conversation compacted." if compacted else "Nothing to compact."
            )
            finish_command()
            return
        if command == "/cwd":
            await self.send(
                message,
                "`/cwd` only operates inside the setup workspace; rerun setup to change it.",
            )
            finish_command()
            return
        if command == "/send":
            await self.send(
                message,
                "`/send` will be enabled after the encrypted media upload path is available.",
            )
            finish_command()
            return
        if command == "/model":
            await self.send(
                message,
                "`/model` switching is unavailable in the WeChat channel; "
                "using configured default.",
            )
            finish_command()
            return
        if command == "/workspace":
            if self.workspace_switcher is None:
                await self.send(message, "工作区切换仅在多用户 Web 模式可用。")
                finish_command()
                return
            new_agent, thread_id, workspace, reply = await self.workspace_switcher(argument.strip())
            if new_agent is not None:
                old_agent = self.agent
                self.agent = new_agent
                self.runtime_thread_id = thread_id
                self.account = replace(self.account, workspace=workspace)
                try:
                    await old_agent.tools.close()
                except Exception:
                    pass
            await self.send(message, reply)
            finish_command()
            return
        prompt = message.text
        if message.media_items:
            if not prompt.strip():
                prompt = f"[Media message: {len(message.media_items)} item(s)]"
            prompt += (
                "\n\nThe user sent media. Kairo CLI received bounded encrypted-media "
                "metadata only; CDN download/decryption is not available yet."
            )
        await self._typing(message, 1)
        typing_refresh = asyncio.create_task(self._refresh_typing(message))
        runtime_turn_id = ""
        runtime_store = self.runtime_store
        try:
            if runtime_store is not None and self.runtime_thread_id:
                idempotency_key = (
                    hashlib.sha256(message.message_id.encode()).hexdigest()
                    if message.message_id
                    else None
                )
                runtime_turn_id, status, created = runtime_store.reserve_turn(
                    self.runtime_thread_id,
                    prompt,
                    idempotency_key,
                    event_type="turn.started",
                    event_data={"source": "wechat"},
                )
                if not created:
                    result = runtime_store.turn_result(self.runtime_thread_id, runtime_turn_id)
                    if status == "completed" and result is not None:
                        await self.send(message, result[1])
                        return
                    runtime_turn_id, _status, _created = runtime_store.reserve_turn(
                        self.runtime_thread_id,
                        prompt,
                        None,
                        event_type="turn.started",
                        event_data={"source": "wechat", "retry_of": runtime_turn_id},
                    )
            if self.approval is not None:
                self.approval.message = message
            answer = await self.agent.run(prompt)
            if runtime_turn_id and runtime_store is not None:
                runtime_store.update_turn_status(
                    runtime_turn_id,
                    "completed",
                    response=answer,
                    event_type="turn.completed",
                    event_data={"turn_id": runtime_turn_id},
                )
            await self.send(message, answer)
        except AgentCanceled:
            inbox_status = "canceled"
            if runtime_turn_id and runtime_store is not None:
                runtime_store.update_turn_status(runtime_turn_id, "canceled")
            await self.send(message, "Task canceled.")
        except Exception as exc:
            inbox_status = "failed"
            error = safe_redacted_text(exc, 2_000, "...[WeChat error truncated]")
            if runtime_turn_id and runtime_store is not None:
                runtime_store.update_turn_status(runtime_turn_id, "failed", error=error)
            await self.send(message, f"Kairo CLI error: {error}")
        finally:
            if inbox_id and finish is not None:
                finish(inbox_id, inbox_status)
            if self.approval is not None:
                self.approval.message = None
            typing_refresh.cancel()
            await asyncio.gather(typing_refresh, return_exceptions=True)
            await self._typing(message, 2)

    async def _typing(self, message: WechatMessage, status: int) -> None:
        try:
            await self.client.send_typing(
                self.account, message.from_user_id, message.context_token, status
            )
        except Exception:
            pass

    async def _refresh_typing(self, message: WechatMessage) -> None:
        while True:
            await asyncio.sleep(5)
            await self._typing(message, 1)

    async def send(self, message: WechatMessage, text: str) -> None:
        for chunk in split_message(format_wechat_text(text)):
            await self.client.send_text(
                self.account, message.from_user_id, message.context_token, chunk
            )
