from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

from ..cancellation import AgentCanceled, wait_with_cancellation
from ..llm import LlmClient
from ..models import LlmResponse, Message
from .context import estimate_message_tokens

MAX_SUMMARY_INPUT_CHARS = 60_000
MAX_SUMMARY_OUTPUT_CHARS = 20_000


class ConversationCompactor:
    def __init__(
        self,
        llm: LlmClient,
        retain_recent_rounds: int = 3,
        complete_handler: Callable[
            [list[Message], str, asyncio.Event | None], Awaitable[LlmResponse]
        ]
        | None = None,
    ) -> None:
        self.llm = llm
        self.retain_recent_rounds = max(1, retain_recent_rounds)
        self.complete_handler = complete_handler

    async def compact_if_needed(
        self,
        history: list[Message],
        current_tokens: int,
        trigger_tokens: int,
        cancel_event: asyncio.Event | None = None,
    ) -> bool:
        if current_tokens < trigger_tokens:
            return False
        return await self._compact(history, self.retain_recent_rounds, cancel_event)

    async def compact_now(
        self, history: list[Message], cancel_event: asyncio.Event | None = None
    ) -> bool:
        return await self._compact(history, 1, cancel_event)

    async def _compact(
        self,
        history: list[Message],
        retain_rounds: int,
        cancel_event: asyncio.Event | None,
    ) -> bool:
        snapshot = list(history)
        user_indices = [index for index, item in enumerate(snapshot) if item.role == "user"]
        if len(user_indices) <= retain_rounds:
            return False
        split_index = user_indices[-retain_rounds]
        if split_index <= 0:
            return False
        old_messages = snapshot[:split_index]
        transcript = self._transcript(old_messages)
        prompt = (
            "Compress the previous agent trajectory.\n\n"
            "You MUST preserve:\n\n"
            "1. User's original objective\n"
            "2. Current implementation status\n"
            "3. Important architectural decisions\n"
            "4. Files that were modified\n"
            "5. Important functions/classes\n"
            "6. Errors encountered\n"
            "7. Solutions already attempted\n"
            "8. Unresolved issues\n"
            "9. Current TODO list\n"
            "10. Constraints that must not be violated\n\n"
            "Remove:\n\n"
            "- redundant tool outputs\n"
            "- repeated explanations\n"
            "- obsolete intermediate reasoning\n"
            "- verbose logs\n\n"
            "=== conversation ===\n" + transcript + "\n=== end ==="
        )
        try:
            messages = [
                Message("system", "Return only a faithful conversation summary."),
                Message("user", prompt),
            ]
            if self.complete_handler is not None:
                response = await self.complete_handler(messages, "compaction", cancel_event)
            else:
                response = await wait_with_cancellation(self.llm.complete(messages), cancel_event)
        except AgentCanceled:
            raise
        except Exception:
            return False
        summary = response.content.strip()
        if not summary or len(summary) > MAX_SUMMARY_OUTPUT_CHARS:
            return False
        rebuilt = [
            Message("user", "[Compacted conversation summary]\n" + summary),
            Message("assistant", "Understood. I will continue from this context."),
            *snapshot[split_index:],
        ]
        if estimate_message_tokens(rebuilt) >= estimate_message_tokens(snapshot):
            return False
        if len(history) != len(snapshot) or any(
            current is not original for current, original in zip(history, snapshot, strict=True)
        ):
            return False
        history[:] = rebuilt
        return True

    @staticmethod
    def _transcript(messages: list[Message]) -> str:
        chunks: list[str] = []
        size = 0

        def append(value: str) -> bool:
            nonlocal size
            separator = 2 if chunks else 0
            remaining = MAX_SUMMARY_INPUT_CHARS - size - separator
            if remaining <= 0:
                return False
            selected = value[:remaining]
            chunks.append(selected)
            size += separator + len(selected)
            return len(selected) == len(value)

        for message in messages:
            content = (
                message.content
                if isinstance(message.content, str)
                else "\n".join(
                    str(part.get("text", ""))
                    for part in message.content
                    if isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text", ""), str)
                )
            )
            if not append(f"{message.role.upper()}: {content}"):
                break
            for call in message.tool_calls:
                try:
                    arguments = json.dumps(
                        call.arguments,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                except (RecursionError, TypeError, ValueError):
                    arguments = '{"_invalid_arguments":true}'
                if not append(f"  TOOL_CALL {call.name}: {arguments}"):
                    break
            if size >= MAX_SUMMARY_INPUT_CHARS:
                break
        if size >= MAX_SUMMARY_INPUT_CHARS:
            marker = "[Long history truncated before summarization]"
            if len(chunks[-1]) > len(marker):
                chunks[-1] = chunks[-1][: -len(marker)] + marker
        return "\n\n".join(chunks)
