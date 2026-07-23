from __future__ import annotations

import asyncio
import threading
from collections import deque
from dataclasses import dataclass, replace

from ...agent import Agent
from ...cancellation import AgentCanceled
from ...channels.wechat.formatting import (
    format_wechat_text as format_wechat_text,
)
from ...channels.wechat.formatting import (
    split_message as split_message,
)
from ...trace import safe_redacted_text
from .accounts import WechatAccount, WechatAccountStore, WechatMessage
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


class WechatChannel:
    def __init__(
        self,
        client: IlinkClient,
        store: WechatAccountStore,
        account: WechatAccount,
        agent: Agent,
    ) -> None:
        self.client = client
        self.store = store
        self.account = account
        self.agent = agent
        self.running = True
        self.paused = False
        self.seen: set[str] = set()
        self.seen_order: deque[str] = deque()
        self.queue: deque[WechatMessage] = deque()
        self.max_queue = 100
        self.active_task: asyncio.Task[None] | None = None

    async def run(self) -> None:
        try:
            await self.client.notify(self.account, True)
        except Exception:
            pass
        timeout_ms = 35_000
        try:
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
                if update.sync_buf != self.account.sync_buf:
                    updated = replace(self.account, sync_buf=update.sync_buf)
                    if not self.store.update_sync_buf(self.account, update.sync_buf):
                        self.running = False
                        break
                    self.account = updated
                for message in update.messages:
                    await self.handle(message)
                await self._reap_active()
                await self._start_next()
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

    async def handle(self, message: WechatMessage) -> None:
        if not message.from_user_id or message.from_user_id != self.account.bound_user_id:
            return
        if message.message_id and message.message_id in self.seen:
            return
        if message.message_id:
            self.seen.add(message.message_id)
            self.seen_order.append(message.message_id)
            while len(self.seen_order) > 2_000:
                self.seen.discard(self.seen_order.popleft())
        command, _, _argument = message.text.strip().partition(" ")
        command = command.casefold()
        if command == "/help":
            await self.send(
                message,
                "/help /status /clear /compact /model /cwd /send /pause /resume /stop\n"
                "execute_command 和 MCP 默认拒绝；写入仅限绑定 workspace。",
            )
            return
        if command == "/status":
            await self.send(
                message,
                f"Kairo CLI channel: {'paused' if self.paused else 'running'}\n"
                f"Queue: {len(self.queue)}\n"
                f"Agent: {'running' if self.active_task else 'idle'}",
            )
            return
        if command == "/pause":
            self.paused = True
            await self.send(message, "Kairo CLI channel paused; ordinary messages will queue.")
            return
        if command == "/resume":
            self.paused = False
            await self.send(message, "Kairo CLI channel resumed.")
            await self._start_next()
            return
        if command in {"/stop", "/cancel"}:
            self.agent.cancel()
            await self.send(message, "Cancellation requested.")
            return
        queued_commands = {"/clear", "/compact", "/model", "/cwd", "/send"}
        if command.startswith("/") and command not in queued_commands:
            await self.send(message, f"Unknown WeChat command: {command}\nSend /help for commands.")
            return
        if len(self.queue) >= self.max_queue:
            await self.send(message, "Kairo CLI channel queue is full; retry later.")
            return
        self.queue.append(message)
        await self._start_next()

    async def _start_next(self) -> None:
        if self.paused or self.active_task is not None or not self.queue:
            return
        message = self.queue.popleft()
        self.active_task = asyncio.create_task(self._process(message))

    async def _reap_active(self) -> None:
        if self.active_task is None or not self.active_task.done():
            return
        await asyncio.gather(self.active_task, return_exceptions=True)
        self.active_task = None
        await self._start_next()

    async def _process(self, message: WechatMessage) -> None:
        command = message.text.strip().partition(" ")[0].casefold()
        if command == "/clear":
            self.agent.clear()
            await self.send(message, "Conversation cleared; long-term memory retained.")
            return
        if command == "/compact":
            compacted = await self.agent.compact()
            await self.send(
                message, "Conversation compacted." if compacted else "Nothing to compact."
            )
            return
        if command == "/cwd":
            await self.send(
                message,
                "`/cwd` only operates inside the setup workspace; rerun setup to change it.",
            )
            return
        if command == "/send":
            await self.send(
                message,
                "`/send` will be enabled after the encrypted media upload path is available.",
            )
            return
        if command == "/model":
            await self.send(
                message,
                "`/model` switching is unavailable in the WeChat channel; "
                "using configured default.",
            )
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
        try:
            answer = await self.agent.run(prompt)
            await self.send(message, answer)
        except AgentCanceled:
            await self.send(message, "Task canceled.")
        except Exception as exc:
            error = safe_redacted_text(exc, 2_000, "...[WeChat error truncated]")
            await self.send(message, f"Kairo CLI error: {error}")
        finally:
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
