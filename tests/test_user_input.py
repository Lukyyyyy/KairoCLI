from pathlib import Path

import pytest

from kairocli.agent import Agent
from kairocli.llm import LlmClient
from kairocli.models import LlmResponse, Message
from kairocli.tools import ToolRegistry
from kairocli.user_input import (
    UserInputError,
    normalize_interactive_submission,
    normalize_user_input,
)


class RecordingClient(LlmClient):
    provider = "test"
    model = "input"

    def __init__(self) -> None:
        self.prompt = ""

    async def complete(
        self, messages: list[Message], tools: list[dict[str, object]] | None = None
    ) -> LlmResponse:
        self.prompt = str(messages[-1].content)
        return LlmResponse(content="done")


def test_user_input_preserves_layout_and_normalizes_all_carriage_returns() -> None:
    assert normalize_user_input("first\rsecond\r\nthird\nfourth") == (
        "first\nsecond\nthird\nfourth"
    )
    assert normalize_user_input("  keep whitespace  ") == "  keep whitespace  "


def test_user_input_rejects_utf8_byte_overflow_and_surrogates() -> None:
    with pytest.raises(UserInputError, match="4-byte"):
        normalize_user_input("你好", max_bytes=4)
    with pytest.raises(UserInputError, match="surrogate"):
        normalize_user_input("bad\ud800")


def test_interactive_submission_defers_large_inline_image_budget_to_processor() -> None:
    image = "@image:data:image/png;base64," + "A" * (1024 * 1024)
    assert normalize_interactive_submission(image) == image
    with pytest.raises(UserInputError, match="1048576-byte"):
        normalize_interactive_submission("plain " + "A" * (1024 * 1024))


async def test_agent_normalizes_before_history_and_rejects_before_mutation(
    tmp_path: Path,
) -> None:
    client = RecordingClient()
    agent = Agent(client, ToolRegistry(tmp_path), "system")
    assert await agent.run("one\rtwo\r\nthree") == "done"
    assert agent.history[0].content == "one\ntwo\nthree"
    assert client.prompt == "one\ntwo\nthree"

    oversized = "中" * 400_000
    with pytest.raises(UserInputError, match="1048576-byte"):
        await agent.run(oversized)
    assert len(agent.history) == 2
