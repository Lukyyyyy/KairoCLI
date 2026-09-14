import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import kairocli.agent.compaction as compaction_module
from kairocli.agent import Agent, AgentBudget
from kairocli.compaction import ConversationCompactor
from kairocli.context import (
    ContextProfile,
    estimate_message_tokens,
    estimate_text_tokens,
    estimated_cost_cny,
)
from kairocli.llm import LlmClient
from kairocli.models import LlmResponse, Message, ToolCall, Usage
from kairocli.tools import ToolRegistry


class CapabilityClient(LlmClient):
    provider = "deepseek"
    model = "large"

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        return LlmResponse(content="ok")


def test_context_profile_is_derived_from_model_window() -> None:
    profile = ContextProfile.from_client(CapabilityClient())
    assert profile.max_context_window == 1_000_000
    assert profile.agent_token_budget == 800_000
    assert profile.compression_trigger_tokens == 967_000
    assert profile.short_term_memory_budget == 450_000
    assert profile.memory_context_tokens == 5_000
    assert profile.mcp_resource_index_enabled
    assert profile.prompt_cache_mode == "automatic-prefix-cache"


def test_token_estimator_counts_cjk_tools_and_images() -> None:
    message = Message(
        "assistant",
        [
            {"type": "text", "text": "中文abcd"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + "A" * 2_048},
            },
        ],
        tool_calls=[ToolCall("1", "read_file", {"path": "src/a.py"})],
    )
    assert estimate_text_tokens("中文abcd") == 3
    assert estimate_message_tokens([message]) >= 263


def test_deepseek_v4_flash_cost_uses_peak_and_off_peak_rates() -> None:
    peak = datetime(2026, 8, 17, 10, tzinfo=ZoneInfo("Asia/Shanghai"))
    off_peak = datetime(2026, 8, 17, 13, tzinfo=ZoneInfo("Asia/Shanghai"))

    assert (
        estimated_cost_cny(
            "deepseek",
            2_000_000,
            1_000_000,
            1_000_000,
            model="deepseek-v4-flash",
            at=peak,
        )
        == 12.1
    )
    assert (
        estimated_cost_cny(
            "deepseek",
            2_000_000,
            1_000_000,
            1_000_000,
            model="deepseek-v4-flash",
            at=off_peak,
        )
        == 6.05
    )


def test_deepseek_v4_pro_cost_uses_peak_and_off_peak_rates() -> None:
    peak = datetime(2026, 8, 17, 14, tzinfo=ZoneInfo("Asia/Shanghai"))
    off_peak = datetime(2026, 8, 17, 18, tzinfo=ZoneInfo("Asia/Shanghai"))

    assert (
        estimated_cost_cny(
            "deepseek",
            2_000_000,
            1_000_000,
            1_000_000,
            model="deepseek-v4-pro",
            at=peak,
        )
        == 36.3
    )
    assert (
        estimated_cost_cny(
            "deepseek",
            2_000_000,
            1_000_000,
            1_000_000,
            model="deepseek-v4-pro",
            at=off_peak,
        )
        == 18.15
    )


def test_deepseek_cost_keeps_old_rates_before_v4_pricing_effective_date() -> None:
    before_change = datetime(2026, 8, 16, 23, 59, tzinfo=ZoneInfo("Asia/Shanghai"))

    assert (
        estimated_cost_cny(
            "deepseek",
            1_000_000,
            1_000_000,
            1_000_000,
            model="deepseek-v4-flash",
            at=before_change,
        )
        == 8.5
    )


class SummaryClient(LlmClient):
    provider = "test"
    model = "test"

    def __init__(self, responses: list[LlmResponse]) -> None:
        self.responses = responses
        self.requests: list[list[Message]] = []

    def max_context_window(self) -> int:
        return 8_000

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        self.requests.append(messages)
        return self.responses.pop(0)


async def test_compactor_splits_at_user_boundary_and_preserves_tool_pair() -> None:
    client = SummaryClient([LlmResponse(content="verified summary")])
    compactor = ConversationCompactor(client, retain_recent_rounds=2)
    history = [
        Message("user", "old one " + "x" * 2_000),
        Message("assistant", "old answer"),
        Message("user", "old two"),
        Message("assistant", "old answer two"),
        Message("user", "recent one"),
        Message(
            "assistant",
            "",
            tool_calls=[ToolCall("call-1", "read_file", {"path": "a.py"})],
        ),
        Message("tool", "file content", tool_call_id="call-1"),
        Message("assistant", "done"),
        Message("user", "recent two"),
        Message("assistant", "answer two"),
    ]
    assert await compactor.compact_if_needed(history, 10_000, 100)
    assert history[0].role == "user" and "verified summary" in str(history[0].content)
    assert history[2].content == "recent one"
    assert history[3].tool_calls[0].id == "call-1"
    assert history[4].role == "tool"


async def test_compactor_failure_does_not_mutate_history() -> None:
    class FailingClient(CapabilityClient):
        async def complete(
            self, messages: list[Message], tools: list[dict[str, Any]] | None = None
        ) -> LlmResponse:
            raise RuntimeError("offline")

    history = [Message("user", f"q{i}") for i in range(4)]
    original = list(history)
    assert not await ConversationCompactor(FailingClient()).compact_now(history)
    assert history == original


async def test_compactor_does_not_overwrite_history_appended_during_summary() -> None:
    class BlockingSummaryClient(CapabilityClient):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def complete(
            self, messages: list[Message], tools: list[dict[str, Any]] | None = None
        ) -> LlmResponse:
            self.started.set()
            await self.release.wait()
            return LlmResponse(content="short summary")

    client = BlockingSummaryClient()
    history = [Message("user", f"question {index} " + "x" * 2_000) for index in range(4)]
    original = list(history)
    compacting = asyncio.create_task(ConversationCompactor(client).compact_now(history))
    await client.started.wait()
    appended = Message("assistant", "arrived while summarizing")
    history.append(appended)
    client.release.set()

    assert await compacting is False
    assert history == [*original, appended]


async def test_compactor_bounds_summary_and_degrades_recursive_tool_arguments() -> None:
    recursive: dict[str, Any] = {}
    recursive["self"] = recursive
    client = SummaryClient([LlmResponse(content="bounded summary")])
    history = [
        Message("user", "old " + "x" * 4_000),
        Message(
            "assistant",
            "",
            tool_calls=[ToolCall("call", "custom", recursive)],
        ),
        Message("tool", "result", tool_call_id="call"),
        Message("user", "recent"),
        Message("assistant", "answer"),
    ]

    assert await ConversationCompactor(client).compact_now(history) is True
    assert '"_invalid_arguments":true' in str(client.requests[0][-1].content)

    oversized = "x" * (compaction_module.MAX_SUMMARY_OUTPUT_CHARS + 1)
    rejecting = SummaryClient([LlmResponse(content=oversized)])
    untouched = [Message("user", f"q{index} " + "y" * 2_000) for index in range(4)]
    original = list(untouched)

    assert await ConversationCompactor(rejecting).compact_now(untouched) is False
    assert untouched == original


async def test_manual_compaction_recovers_from_stale_cancellation(tmp_path: Path) -> None:
    client = SummaryClient([LlmResponse(content="manual summary")])
    agent = Agent(client, ToolRegistry(tmp_path), "system")
    agent.history = [Message("user", f"question {index} " + "x" * 2_000) for index in range(4)]
    agent.cancel()

    assert await agent.compact() is True
    assert agent.compaction_count == 1


async def test_agent_auto_compacts_before_model_request(tmp_path: Path) -> None:
    client = SummaryClient(
        [
            LlmResponse(content="old work summarized"),
            LlmResponse(content="done", usage=Usage(100, 20, 30)),
        ]
    )
    agent = Agent(client, ToolRegistry(tmp_path), "system")
    agent.history = [
        item
        for index in range(4)
        for item in (
            Message("user", f"question {index} " + "x" * 6_000),
            Message("assistant", f"answer {index}"),
        )
    ]
    assert await agent.run("continue") == "done"
    assert agent.compaction_count == 1
    assert agent.total_cached_tokens == 30
    # Compaction is a real billed model call and must be included in usage accounting.
    assert agent.llm_call_count == 2
    assert "缓存输入：30 token" in agent.context_status()


def test_context_status_breaks_down_current_usage(tmp_path: Path) -> None:
    agent = Agent(CapabilityClient(), ToolRegistry(tmp_path), "system")
    agent.history = [Message("user", "hello"), Message("assistant", "world")]

    system = estimate_message_tokens([Message("system", agent.system_prompt)])
    conversation = estimate_message_tokens(agent.history)
    schema = agent.estimate_current_context_tokens() - system - conversation
    status = agent.context_status()

    assert "[█" in status
    assert f"当前占用  {system + schema + conversation:,} / 1,000,000 token" in status
    assert f"系统提示词：{system:,} token" in status
    assert f"工具定义：{schema:,} token" in status
    assert f"会话消息：{conversation:,} token" in status
    assert "2 条" in status
    assert "\n\n压缩与记忆\n" in status
    assert "\n\n累计用量\n" in status
    assert "\n\n运行设置\n" in status


def test_agent_budget_environment_overrides(monkeypatch: Any) -> None:
    monkeypatch.setenv("KAIROCLI_REACT_TOKEN_BUDGET", "1234")
    monkeypatch.setenv("KAIROCLI_REACT_HARD_MAX_ITERATIONS", "12")
    monkeypatch.setenv("KAIROCLI_REACT_STAGNATION_WINDOW", "4")
    budget = AgentBudget.from_environment()
    assert budget == AgentBudget(max_iterations=12, token_budget=1234, stagnation_window=4)
