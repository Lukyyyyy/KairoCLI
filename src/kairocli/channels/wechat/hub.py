from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

from ...runtime_api import RuntimeThreadStore
from ..store import ChannelBinding, ChannelStore
from .accounts import WechatAccount, WechatMediaItem, WechatMessage
from .channel import WechatApprovalHandler, WechatChannel
from .client import IlinkClient


class _BindingSyncStore:
    def __init__(self, store: ChannelStore, binding_id: str) -> None:
        self.store = store
        self.binding_id = binding_id

    def update_sync_buf(self, _expected: WechatAccount, sync_buf: str) -> bool:
        return self.store.update_sync_buf(self.binding_id, sync_buf)

    def save_update(
        self,
        expected: WechatAccount,
        sync_buf: str,
        messages: list[WechatMessage],
    ) -> bool:
        payloads = [
            {
                "message_id": message.message_id,
                "from_user_id": message.from_user_id,
                "context_token": message.context_token,
                "text": message.text,
                "media_items": [
                    {
                        "type": media.type,
                        "file_name": media.file_name,
                        "mime_type": media.mime_type,
                        "encrypt_query_param": media.encrypt_query_param,
                        "aes_key": media.aes_key,
                    }
                    for media in message.media_items
                ],
            }
            for message in messages
            if message.from_user_id == expected.bound_user_id
        ]
        self.store.enqueue_messages(self.binding_id, payloads, sync_buf)
        return True

    def claim_message(self) -> tuple[int, WechatMessage] | None:
        item = self.store.claim_message(self.binding_id)
        if item is None:
            return None
        inbox_id, payload = item
        media_items = tuple(
            WechatMediaItem(
                str(media.get("type", "")),
                str(media.get("file_name", "")),
                str(media.get("mime_type", "")),
                str(media.get("encrypt_query_param", "")),
                str(media.get("aes_key", "")),
            )
            for media in payload.get("media_items", [])
            if isinstance(media, dict)
        )
        return inbox_id, WechatMessage(
            str(payload.get("message_id", "")),
            str(payload.get("from_user_id", "")),
            str(payload.get("context_token", "")),
            str(payload.get("text", "")),
            media_items,
        )

    def finish_message(self, inbox_id: int, status: str) -> None:
        self.store.finish_message(inbox_id, status)


class WechatHub:
    def __init__(
        self,
        store: ChannelStore,
        agent_factory: Any,
        config_factory: Any,
        workspace_factory: Any,
        workspace_list_factory: Any,
        configure_agent: Any,
        runtime_store: RuntimeThreadStore,
    ) -> None:
        self.store = store
        self.agent_factory = agent_factory
        self.config_factory = config_factory
        self.workspace_factory = workspace_factory
        self.workspace_list_factory = workspace_list_factory
        self.configure_agent = configure_agent
        self.runtime_store = runtime_store
        self._active: dict[str, tuple[WechatChannel, asyncio.Task[None]]] = {}
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        for binding in self.store.list_bindings(enabled_only=True):
            if binding.channel_type == "wechat":
                await self.start_binding(binding)

    async def start_binding(self, binding: ChannelBinding) -> None:
        async with self._lock:
            if binding.id in self._active or not binding.enabled:
                return
            credentials = self.store.wechat_credentials(binding.id)
            if credentials is None:
                self.store.set_enabled(binding.id, False)
                return
            self.store.requeue_running(binding.id)
            workspace: Path = self.workspace_factory(binding.user_id, binding.active_workspace)
            approval = WechatApprovalHandler()
            agent, thread_id = self._agent_for(binding, workspace, approval)

            async def switch_workspace(
                argument: str,
            ) -> tuple[Any | None, str, str, str]:
                return await self._switch_workspace(binding.id, approval, argument)

            account = WechatAccount(
                credentials.token,
                binding.external_account_id,
                credentials.base_url,
                binding.external_user_id,
                str(workspace),
                sync_buf=credentials.sync_buf,
                created_at=binding.created_at,
            )
            channel = WechatChannel(
                IlinkClient(credentials.base_url),
                _BindingSyncStore(self.store, binding.id),
                account,
                agent,
                approval,
                self.runtime_store,
                thread_id,
                binding.user_id,
                switch_workspace,
            )
            task = asyncio.create_task(channel.run(), name=f"kairo-wechat-{binding.id}")
            self._active[binding.id] = (channel, task)

            def finished(done: asyncio.Task[None], binding_id: str = binding.id) -> None:
                self._finished(binding_id, done)

            task.add_done_callback(finished)

    def _agent_for(
        self,
        binding: ChannelBinding,
        workspace: Path,
        approval: WechatApprovalHandler,
    ) -> tuple[Any, str]:
        agent = self.agent_factory(
            approver=approval,
            config=self.config_factory(binding.user_id),
            workspace=workspace,
        )
        self.configure_agent(binding.user_id, agent)
        thread_id = self.store.thread_for_workspace(binding.id, str(workspace))
        if thread_id is None or not self.runtime_store.exists_for_user(thread_id, binding.user_id):
            thread_id = f"thread_{uuid.uuid4().hex[:12]}"
            self.runtime_store.create(
                thread_id,
                owner_user_id=binding.user_id,
                workspace=str(workspace),
                event_type="thread.created",
                event_data={"thread_id": thread_id, "source": "wechat"},
            )
            self.store.save_thread(binding.id, str(workspace), thread_id)
        agent.history = self.runtime_store.completed_messages(thread_id)
        return agent, thread_id

    async def _switch_workspace(
        self,
        binding_id: str,
        approval: WechatApprovalHandler,
        argument: str,
    ) -> tuple[Any | None, str, str, str]:
        binding = self.store.binding(binding_id)
        if binding is None:
            return None, "", "", "微信绑定已不存在。"
        workspaces: list[Path] = self.workspace_list_factory(binding.user_id)
        if not argument or argument.casefold() == "list":
            lines = ["可用工作区："]
            lines.extend(
                f"{index}. {'* ' if str(path) == binding.active_workspace else ''}"
                f"{path.name or path} · {path}"
                for index, path in enumerate(workspaces, 1)
            )
            lines.append("发送 /workspace 序号、名称或完整路径进行切换。")
            return None, "", "", "\n".join(lines)
        selected: Path | None = None
        if argument.isdecimal() and 1 <= int(argument) <= len(workspaces):
            selected = workspaces[int(argument) - 1]
        else:
            matches = [
                path
                for path in workspaces
                if str(path) == argument or path.name.casefold() == argument.casefold()
            ]
            if len(matches) == 1:
                selected = matches[0]
            elif len(matches) > 1:
                return None, "", "", "存在同名工作区，请使用序号或完整路径。"
        if selected is None:
            return None, "", "", "工作区不存在或未获管理员授权；发送 /workspace 查看列表。"
        if str(selected) == binding.active_workspace:
            return None, "", "", f"当前已在工作区：{selected.name or selected}"
        workspace: Path = self.workspace_factory(binding.user_id, str(selected))
        agent, thread_id = self._agent_for(binding, workspace, approval)
        self.store.set_workspace(binding.id, str(workspace))
        return (
            agent,
            thread_id,
            str(workspace),
            f"已切换到工作区：{workspace.name or workspace}\n已恢复该工作区的微信会话。",
        )

    async def refresh(self, binding_id: str) -> None:
        await self.stop_binding(binding_id)
        binding = self.store.binding(binding_id)
        if binding is not None and binding.enabled:
            await self.start_binding(binding)

    async def stop_binding(self, binding_id: str) -> None:
        async with self._lock:
            active = self._active.pop(binding_id, None)
        if active is None:
            return
        channel, task = active
        channel.running = False
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        try:
            await channel.agent.tools.close()
        except Exception:
            pass

    async def close(self) -> None:
        for binding_id in tuple(self._active):
            await self.stop_binding(binding_id)

    def _finished(self, binding_id: str, task: asyncio.Task[None]) -> None:
        active = self._active.get(binding_id)
        if active is not None and active[1] is task:
            self._active.pop(binding_id, None)
        try:
            task.exception()
        except (asyncio.CancelledError, Exception):
            pass
