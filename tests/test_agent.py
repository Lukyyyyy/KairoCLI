import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

import kairocli.agent as agent_module
import kairocli.agent.core as agent_core_module
import kairocli.cancellation as cancellation_module
from kairocli.agent import (
    Agent,
    AgentBudget,
    AgentCanceled,
    AgentOrchestrator,
    PlanExecuteAgent,
    markdown_fence_for,
)
from kairocli.llm import LlmClient, LlmError
from kairocli.models import LlmResponse, Message, ToolCall, ToolOutput, Usage
from kairocli.plan import ExecutionPlan
from kairocli.tools import ToolDefinition, ToolRegistry


class FakeClient(LlmClient):
    provider = "fake"
    model = "fake-model"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        self.calls += 1
        if self.calls == 1:
            return LlmResponse(
                tool_calls=[ToolCall("1", "write_file", {"path": "answer.txt", "content": "42"})]
            )
        return LlmResponse(content="done")


async def test_react_tool_loop(tmp_path: Path) -> None:
    client = FakeClient()
    agent = Agent(client, ToolRegistry(tmp_path), "system")
    assert await agent.run("work") == "done"
    assert (tmp_path / "answer.txt").read_text() == "42"
    assert [message.role for message in agent.history] == ["user", "assistant", "tool", "assistant"]
    assert "42" not in str(agent.history[2].content)


async def test_tool_call_observer_is_informational_and_failure_isolated(
    tmp_path: Path,
) -> None:
    observed: list[list[ToolCall]] = []
    result_events: list[tuple[list[ToolCall], list[ToolOutput]]] = []
    agent = Agent(FakeClient(), ToolRegistry(tmp_path), "system")

    def observer(calls: list[ToolCall]) -> None:
        observed.append(calls)
        raise RuntimeError("renderer failed")

    agent.on_tool_calls = observer
    agent.on_tool_results = lambda calls, results: result_events.append((calls, results))
    assert await agent.run("work") == "done"
    assert [call.name for call in observed[0]] == ["write_file"]
    assert (tmp_path / "answer.txt").read_text(encoding="utf-8") == "42"
    assert len(result_events) == 1
    assert result_events[0][1][0].elapsed_ms >= 0


async def test_reasoning_observer_receives_each_model_round(tmp_path: Path) -> None:
    class ReasoningClient(FakeClient):
        async def complete(
            self, messages: list[Message], tools: list[dict[str, Any]] | None = None
        ) -> LlmResponse:
            response = await super().complete(messages, tools)
            response.reasoning_content = f"reasoning-{self.calls}"
            return response

    observed: list[str] = []
    agent = Agent(ReasoningClient(), ToolRegistry(tmp_path), "system")
    agent.on_reasoning = observed.append

    assert await agent.run("work") == "done"
    assert observed == ["reasoning-1", "reasoning-2"]


async def test_reasoning_observer_receives_streaming_deltas_once(tmp_path: Path) -> None:
    class StreamingReasoningClient(LlmClient):
        provider = "test"
        model = "reasoning-stream"

        async def complete(
            self, messages: list[Message], tools: list[dict[str, Any]] | None = None
        ) -> LlmResponse:
            return LlmResponse(content="answer", reasoning_content="think")

        async def complete_streaming(
            self,
            messages: list[Message],
            tools: list[dict[str, Any]] | None = None,
            on_delta: Any = None,
        ) -> LlmResponse:
            assert self.on_reasoning_delta is not None
            self.on_reasoning_delta("think")
            on_delta("answer")
            return LlmResponse(
                content="answer",
                reasoning_content="think",
                streamed=True,
                reasoning_streamed=True,
            )

    observed: list[str] = []
    agent = Agent(StreamingReasoningClient(), ToolRegistry(tmp_path), "system")
    agent.on_reasoning_delta = observed.append

    assert await agent.run("work") == "answer"
    assert observed == ["think"]


async def test_tool_observers_cannot_mutate_execution_or_protocol_history(
    tmp_path: Path,
) -> None:
    agent = Agent(FakeClient(), ToolRegistry(tmp_path), "system")

    def mutate_calls(calls: list[ToolCall]) -> None:
        calls[0].id = "changed"
        calls[0].name = "read_file"
        calls[0].arguments["path"] = "hijacked.txt"
        calls.clear()

    def mutate_results(calls: list[ToolCall], results: list[ToolOutput]) -> None:
        calls.clear()
        results.clear()

    agent.on_tool_calls = mutate_calls
    agent.on_tool_results = mutate_results

    assert await agent.run("work") == "done"
    assert (tmp_path / "answer.txt").read_text(encoding="utf-8") == "42"
    assert not (tmp_path / "hijacked.txt").exists()
    assert [message.role for message in agent.history] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert agent.history[1].tool_calls[0].id == "1"
    assert agent.history[2].tool_call_id == "1"


async def test_async_tool_observer_is_time_bounded_and_cannot_cancel_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agent_core_module, "TOOL_OBSERVER_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(agent_core_module, "TOOL_OBSERVER_CANCEL_GRACE_SECONDS", 0.01)
    release = asyncio.Event()
    agent = Agent(FakeClient(), ToolRegistry(tmp_path), "system")

    async def stubborn(calls: list[ToolCall]) -> None:
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()

    agent.on_tool_calls = stubborn
    await asyncio.wait_for(
        agent._emit_tool_calls([ToolCall("1", "read_file", {"path": "safe"})]),
        0.2,
    )
    assert agent_module._DETACHED_TOOL_OBSERVERS

    async def self_cancel(calls: list[ToolCall]) -> None:
        raise asyncio.CancelledError

    agent.on_tool_calls = self_cancel
    await agent._emit_tool_calls([ToolCall("2", "read_file", {"path": "safe"})])
    assert not agent.cancel_event.is_set()

    release.set()
    for _ in range(50):
        if not agent_module._DETACHED_TOOL_OBSERVERS:
            break
        await asyncio.sleep(0.01)
    assert not agent_module._DETACHED_TOOL_OBSERVERS


async def test_content_observer_failure_falls_back_to_final_rendering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agent_core_module, "TOOL_OBSERVER_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(agent_core_module, "TOOL_OBSERVER_CANCEL_GRACE_SECONDS", 0.01)
    release = asyncio.Event()

    class StreamingClient(LlmClient):
        provider = "test"
        model = "stream-observer"

        async def complete(
            self,
            messages: list[Message],
            tools: list[dict[str, Any]] | None = None,
        ) -> LlmResponse:
            return LlmResponse(content="visible answer")

        async def complete_streaming(
            self,
            messages: list[Message],
            tools: list[dict[str, Any]] | None = None,
            on_delta: Any = None,
        ) -> LlmResponse:
            outcome = on_delta("visible answer")
            if outcome is not None:
                await outcome
            return LlmResponse(content="visible answer", streamed=True)

    async def stubborn(delta: str) -> None:
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()

    agent = Agent(StreamingClient(), ToolRegistry(tmp_path), "system")
    agent.on_content_delta = stubborn

    assert await asyncio.wait_for(agent.run("work"), 0.2) == "visible answer"
    assert agent.last_response_streamed is False
    assert agent_module._DETACHED_TOOL_OBSERVERS
    release.set()
    for _ in range(50):
        if not agent_module._DETACHED_TOOL_OBSERVERS:
            break
        await asyncio.sleep(0.01)
    assert not agent_module._DETACHED_TOOL_OBSERVERS

    def broken(delta: str) -> None:
        raise RuntimeError("renderer disappeared")

    agent.on_content_delta = broken
    assert await agent.run("again") == "visible answer"
    assert agent.last_response_streamed is False


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"max_iterations": 0}, "max_iterations"),
        ({"token_budget": 0}, "token_budget"),
        ({"stagnation_window": 1}, "stagnation_window"),
    ],
)
def test_agent_budget_rejects_invalid_direct_configuration(
    arguments: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        AgentBudget(**arguments)


class BudgetClient(LlmClient):
    provider = "budget"
    model = "budget"

    def __init__(self, response: LlmResponse) -> None:
        self.response = response
        self.messages: list[Message] = []

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        self.messages = messages
        return self.response


async def test_final_answer_is_not_discarded_when_it_reaches_token_budget(
    tmp_path: Path,
) -> None:
    response = LlmResponse(content="finished", usage=Usage(70, 30, 5))
    agent = Agent(
        BudgetClient(response),
        ToolRegistry(tmp_path),
        "system",
        AgentBudget(token_budget=100),
    )

    assert await agent.run("work") == "finished"
    assert [message.content for message in agent.history] == ["work", "finished"]
    assert agent.total_cached_tokens == 5


async def test_exhausted_budget_blocks_new_tool_side_effects_without_broken_history(
    tmp_path: Path,
) -> None:
    response = LlmResponse(
        tool_calls=[ToolCall("call-1", "write_file", {"path": "should-not-exist", "content": "x"})],
        usage=Usage(80, 20),
    )
    agent = Agent(
        BudgetClient(response),
        ToolRegistry(tmp_path),
        "system",
        AgentBudget(token_budget=100),
    )

    with pytest.raises(RuntimeError, match=r"100 / 100.*not executed"):
        await agent.run("work")

    assert not (tmp_path / "should-not-exist").exists()
    assert [message.role for message in agent.history] == ["user"]


class RepeatingToolClient(LlmClient):
    provider = "repeat"
    model = "repeat"

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        return LlmResponse(tool_calls=[ToolCall("same", "read_file", {"path": "x.txt"})])


async def test_stagnation_never_leaves_unpaired_tool_call_history(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("x", encoding="utf-8")
    agent = Agent(
        RepeatingToolClient(),
        ToolRegistry(tmp_path),
        "system",
        AgentBudget(stagnation_window=2),
    )

    with pytest.raises(RuntimeError, match="repeated identical"):
        await agent.run("repeat")

    assert [message.role for message in agent.history] == ["user", "assistant", "tool"]


async def test_tool_batch_cancellation_preserves_paired_protocol_result(
    tmp_path: Path,
) -> None:
    started = asyncio.Event()

    async def slow(arguments: dict[str, Any]) -> str:
        started.set()
        await asyncio.Event().wait()
        return "unreachable"

    tools = ToolRegistry(tmp_path)
    tools.register(ToolDefinition("read_file", "slow", {"type": "object"}, slow))
    agent = Agent(RepeatingToolClient(), tools, "system")
    running = asyncio.create_task(agent.run("cancel"))
    await started.wait()
    agent.cancel()

    with pytest.raises(AgentCanceled):
        await running

    assert [message.role for message in agent.history] == ["user", "assistant", "tool"]
    assert agent.history[-1].tool_call_id == "same"
    canceled = json.loads(str(agent.history[-1].content))
    assert canceled == {
        "error": "Tool batch canceled before a reliable result was available",
        "canceled": True,
        "side_effects_may_have_completed": False,
    }


async def test_canceled_mutation_history_discloses_possible_committed_side_effect(
    tmp_path: Path,
) -> None:
    started = asyncio.Event()

    class MutationClient(LlmClient):
        provider = "cancel"
        model = "mutation"

        async def complete(
            self, messages: list[Message], tools: list[dict[str, Any]] | None = None
        ) -> LlmResponse:
            return LlmResponse(
                tool_calls=[
                    ToolCall(
                        "write-1",
                        "write_file",
                        {"path": "committed.txt", "content": "committed"},
                    )
                ]
            )

    tools = ToolRegistry(tmp_path)
    schema = tools._tools["write_file"].parameters

    async def commit_then_wait(arguments: dict[str, Any]) -> str:
        await asyncio.to_thread(
            (tmp_path / str(arguments["path"])).write_text,
            str(arguments["content"]),
            encoding="utf-8",
        )
        started.set()
        await asyncio.Event().wait()
        return "unreachable"

    tools.register(ToolDefinition("write_file", "fixture", schema, commit_then_wait))
    agent = Agent(MutationClient(), tools, "system")
    running = asyncio.create_task(agent.run("mutate"))
    await started.wait()
    agent.cancel()

    with pytest.raises(AgentCanceled):
        await running

    assert (tmp_path / "committed.txt").read_text(encoding="utf-8") == "committed"
    assert [message.role for message in agent.history] == ["user", "assistant", "tool"]
    canceled = json.loads(str(agent.history[-1].content))
    assert canceled["canceled"] is True
    assert canceled["side_effects_may_have_completed"] is True


def test_clear_and_export(tmp_path: Path) -> None:
    agent = Agent(FakeClient(), ToolRegistry(tmp_path), "system instructions")
    agent.history = [Message("user", "hello")]
    assert "system instructions" in agent.export_markdown()
    agent.clear()
    assert agent.history == []


def test_export_preserves_reasoning_and_uses_collision_safe_fences(
    tmp_path: Path,
) -> None:
    agent = Agent(FakeClient(), ToolRegistry(tmp_path), "system prompt")
    agent.history = [
        Message("user", "inspect code"),
        Message(
            "assistant",
            "calling tool",
            [ToolCall("call_1", "write_file", {"content": "```python\npass\n```"})],
            reasoning_content="first\rsecond",
        ),
        Message("tool", "tool output\n```python\nclass A: pass\n```", tool_call_id="call_1"),
    ]

    exported = agent.export_markdown()

    assert "## System\n\nsystem prompt" in exported
    assert "> **Reasoning**:\n>\n> first\n> second" in exported
    assert "````json" in exported
    assert "````\ntool output\n```python\nclass A: pass\n```\n````" in exported
    assert "**Exported at**:" in exported
    assert markdown_fence_for("before ````` after") == "``````"


def test_export_bounds_large_tool_results(tmp_path: Path) -> None:
    agent = Agent(FakeClient(), ToolRegistry(tmp_path), "system")
    agent.history = [Message("tool", "x" * 9_000, tool_call_id="call_1")]

    exported = agent.export_markdown()

    assert "original length 9000 characters" in exported
    assert "x" * 8_001 not in exported


class PlanClient(LlmClient):
    provider = "plan"
    model = "plan"

    def __init__(self) -> None:
        self.responses = [
            LlmResponse(
                content=(
                    '{"tasks":[{"id":"inspect","description":"inspect files","dependencies":[]}]}'
                )
            ),
            LlmResponse(content="inspection complete"),
            LlmResponse(content="final reviewed answer"),
        ]

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        return self.responses.pop(0)


async def test_plan_agent_builds_executes_and_reviews(tmp_path: Path) -> None:
    base = Agent(PlanClient(), ToolRegistry(tmp_path), "system")
    result = await PlanExecuteAgent(base).run("understand project")
    assert result == "final reviewed answer"
    assert base.llm_call_count == 3


class PlanToolClient(LlmClient):
    provider = "plan"
    model = "plan-tool"

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        prompt = str(messages[-1].content)
        if "Divide" in prompt or "planner" in str(messages[0].content):
            return LlmResponse(
                content='{"tasks":[{"id":"inspect","description":"inspect","dependencies":[]}]}'
            )
        if any(message.role == "tool" for message in messages):
            return LlmResponse(content="worker done")
        if "Assigned plan step" in prompt:
            return LlmResponse(tool_calls=[ToolCall("read-1", "read_file", {"path": "README.md"})])
        return LlmResponse(content="reviewed")


async def test_plan_workers_inherit_tool_progress_but_not_content_observer(
    tmp_path: Path,
) -> None:
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    base = Agent(PlanToolClient(), ToolRegistry(tmp_path), "system")
    calls: list[list[ToolCall]] = []
    results: list[list[ToolOutput]] = []
    deltas: list[str] = []
    base.on_tool_calls = calls.append
    base.on_tool_results = lambda _calls, outputs: results.append(outputs)
    base.on_content_delta = deltas.append

    assert await PlanExecuteAgent(base).run("inspect project") == "reviewed"
    assert [[call.name for call in event] for event in calls] == [["read_file"]]
    assert len(results) == 1
    assert deltas == []


class ReplanClient(LlmClient):
    provider = "plan"
    model = "plan"

    def __init__(self) -> None:
        self.planner_calls = 0

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        system = str(messages[0].content)
        prompt = str(messages[-1].content)
        if "Kairo CLI planner" in system:
            self.planner_calls += 1
            description = "fail first" if self.planner_calls == 1 else "recover"
            return LlmResponse(
                content=(
                    '{"tasks":[{"id":"a","description":"' + description + '","dependencies":[]}]}'
                )
            )
        if "fail first" in prompt:
            raise RuntimeError("transient failure")
        if "Assigned plan step: recover" in prompt:
            return LlmResponse(content="recovered")
        return LlmResponse(content="final after recovery")


async def test_plan_agent_replans_once_after_early_failure(tmp_path: Path) -> None:
    client = ReplanClient()
    base = Agent(client, ToolRegistry(tmp_path), "system")
    result = await PlanExecuteAgent(base).run("complete goal")
    assert result == "final after recovery"
    assert client.planner_calls == 2


class UnprintableAgentError(RuntimeError):
    def __str__(self) -> str:
        raise KeyboardInterrupt


class UnprintableWorkerClient(LlmClient):
    provider = "plan"
    model = "unprintable-worker"

    def __init__(self) -> None:
        self.calls = 0
        self.final_prompt = ""

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        self.calls += 1
        if self.calls == 1:
            return LlmResponse(
                content='{"tasks":[{"id":"a","description":"work","dependencies":[]}]}'
            )
        if self.calls == 2:
            raise UnprintableAgentError
        self.final_prompt = str(messages[-1].content)
        return LlmResponse(content="recovered final answer")


async def test_plan_terminalizes_unprintable_worker_failure(tmp_path: Path) -> None:
    client = UnprintableWorkerClient()
    base = Agent(client, ToolRegistry(tmp_path), "system")

    result = await PlanExecuteAgent(base, max_replans=0).run("complete goal")

    assert result == "recovered final answer"
    assert "UnprintableAgentError message unavailable" in client.final_prompt


async def test_team_terminalizes_unprintable_worker_failure(tmp_path: Path) -> None:
    client = UnprintableWorkerClient()
    base = Agent(client, ToolRegistry(tmp_path), "system")

    result = await AgentOrchestrator(base, max_retries_per_step=0).run("complete goal")

    assert result == "recovered final answer"
    assert "UnprintableAgentError message unavailable" in client.final_prompt


def test_orchestration_text_repairs_surrogates_within_byte_budget() -> None:
    value = agent_module._truncate_orchestration_text("a\ud800界", 5)

    assert value == "a?界"
    assert len(value.encode("utf-8")) == 5


@pytest.mark.parametrize(
    "planner_payload",
    [
        '{"tasks":[{"id":"bad\\nid","description":"step","dependencies":[]}]}',
        (
            '{"tasks":[{"id":"shadow","description":"wrong","dependencies":[]}],'
            '"tasks":[{"id":"real","description":"ambiguous","dependencies":[]}]}'
        ),
        ('{"tasks":[{"id":"a","description":"step","dependencies":[]}],"score":NaN}'),
        json.dumps(
            {
                "tasks": [{"id": "a", "description": "step", "dependencies": []}],
                "extra": [[[[[[[[[[[[[[[[[0]]]]]]]]]]]]]]]]],
            }
        ),
        json.dumps(
            {
                "tasks": [
                    {
                        "id": "a",
                        "description": "界" * 6_000,
                        "dependencies": [],
                    }
                ]
            }
        ),
        json.dumps(
            {
                "tasks": [
                    {
                        "id": "a",
                        "description": "step",
                        "dependencies": [f"dep-{index}" for index in range(65)],
                    }
                ]
            }
        ),
    ],
)
async def test_plan_agent_falls_back_from_oversized_or_unsafe_plan_schema(
    tmp_path: Path, planner_payload: str
) -> None:
    client = BudgetClient(LlmResponse(content=planner_payload))
    planner = PlanExecuteAgent(Agent(client, ToolRegistry(tmp_path), "system"))

    plan = await planner.create_plan("safe fallback")

    assert list(plan.tasks) == ["task-1"]
    assert plan.tasks["task-1"].description == "执行用户请求的工作并验证结果"


async def test_plan_fallback_does_not_echo_an_oversized_user_request(
    tmp_path: Path,
) -> None:
    client = BudgetClient(LlmResponse(content="not JSON"))
    planner = PlanExecuteAgent(Agent(client, ToolRegistry(tmp_path), "system"))

    plan = await planner.create_plan("界" * 100_000)
    description = plan.tasks["task-1"].description

    assert description == "执行用户请求的工作并验证结果"
    assert "界" not in description


async def test_plan_rewrites_a_verbatim_question_as_an_action(tmp_path: Path) -> None:
    question = "深圳今天天气怎么样？"
    client = BudgetClient(
        LlmResponse(
            content=json.dumps(
                {"tasks": [{"id": "weather", "description": question, "dependencies": []}]},
                ensure_ascii=False,
            )
        )
    )
    planner = PlanExecuteAgent(Agent(client, ToolRegistry(tmp_path), "system"))

    plan = await planner.create_plan(question)

    assert plan.tasks["weather"].description == "执行用户请求的工作并验证结果"


async def test_plan_prompt_requires_chinese_step_descriptions(tmp_path: Path) -> None:
    client = BudgetClient(
        LlmResponse(
            content=(
                '{"tasks":[{"id":"inspect","description":"检查项目",'
                '"dependencies":[]}]}'
            )
        )
    )
    planner = PlanExecuteAgent(Agent(client, ToolRegistry(tmp_path), "system"))

    await planner.create_plan("inspect project", "keep it minimal")

    assert "description 必须使用简体中文" in str(client.messages[0].content)
    assert client.messages[1].content == "任务：inspect project\n审阅反馈：keep it minimal"


def test_team_review_rejects_ambiguous_or_structured_approval_fields() -> None:
    ambiguous = agent_module._parse_review('{"approved":true,"approved":false,"issues":[]}')
    structured = agent_module._parse_review('{"approved": true, "issues": [{"text": "hidden"}]}')

    assert ambiguous[0] is False
    assert structured[0] is False


class LargePlanContextClient(LlmClient):
    provider = "plan"
    model = "large-context"

    def __init__(self) -> None:
        self.dependency_prompt = ""
        self.synthesis_prompt = ""

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        system = str(messages[0].content)
        prompt = str(messages[-1].content)
        if "Kairo CLI planner" in system:
            return LlmResponse(
                content=(
                    '{"tasks":['
                    '{"id":"a","description":"produce","dependencies":[]},'
                    '{"id":"b","description":"consume","dependencies":["a"]}'
                    "]}"
                )
            )
        if "Assigned plan step: produce" in prompt:
            return LlmResponse(content="界" * 300_000)
        if "Assigned plan step: consume" in prompt:
            self.dependency_prompt = prompt
            return LlmResponse(content="consumed")
        self.synthesis_prompt = prompt
        return LlmResponse(content="bounded final")


async def test_plan_agent_bounds_large_dependency_and_synthesis_context(
    tmp_path: Path,
) -> None:
    client = LargePlanContextClient()
    result = await PlanExecuteAgent(Agent(client, ToolRegistry(tmp_path), "system")).run(
        "process large result"
    )

    assert result == "bounded final"
    assert "[orchestration context truncated]" in client.dependency_prompt
    assert "[orchestration context truncated]" in client.synthesis_prompt
    assert len(client.dependency_prompt.encode("utf-8")) < 1024 * 1024
    assert len(client.synthesis_prompt.encode("utf-8")) < 1024 * 1024


class SharedPlanBudgetClient(LlmClient):
    provider = "plan"
    model = "shared-budget"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        self.calls += 1
        if self.calls == 1:
            return LlmResponse(
                content='{"tasks":[{"id":"a","description":"write","dependencies":[]}]}',
                usage=Usage(input_tokens=30),
            )
        if self.calls == 2:
            return LlmResponse(
                tool_calls=[
                    ToolCall(
                        "write-1",
                        "write_file",
                        {"path": "over-budget.txt", "content": "must not run"},
                    )
                ],
                usage=Usage(input_tokens=70),
            )
        raise AssertionError("No model request may start after the shared budget is exhausted")


async def test_plan_shares_budget_and_blocks_over_budget_worker_side_effects(
    tmp_path: Path,
) -> None:
    client = SharedPlanBudgetClient()
    base = Agent(
        client,
        ToolRegistry(tmp_path),
        "system",
        AgentBudget(token_budget=100),
    )

    with pytest.raises(RuntimeError, match=r"budget exhausted \(100 / 100\)"):
        await PlanExecuteAgent(base, max_replans=0).run("write safely")

    assert client.calls == 2
    assert base.total_input_tokens == 100
    assert not (tmp_path / "over-budget.txt").exists()


class BlockingClient(LlmClient):
    provider = "blocking"
    model = "blocking"

    def __init__(self, first_response: LlmResponse | None = None) -> None:
        self.first_response = first_response
        self.started = asyncio.Event()
        self.was_canceled = False

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        if self.first_response is not None:
            response, self.first_response = self.first_response, None
            return response
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.was_canceled = True
            raise
        raise AssertionError("unreachable")


async def test_cancel_interrupts_active_llm_request(tmp_path: Path) -> None:
    client = BlockingClient()
    agent = Agent(client, ToolRegistry(tmp_path), "system")
    run = asyncio.create_task(agent.run("wait"))
    await client.started.wait()
    agent.cancel()
    with pytest.raises(AgentCanceled):
        await asyncio.wait_for(run, 1)
    assert client.was_canceled


async def test_cancel_is_bounded_when_llm_suppresses_cancellation(
    tmp_path: Path,
) -> None:
    release = asyncio.Event()

    class StubbornClient(LlmClient):
        provider = "stubborn"
        model = "stubborn"

        def __init__(self) -> None:
            self.started = asyncio.Event()

        async def complete(
            self, messages: list[Message], tools: list[dict[str, Any]] | None = None
        ) -> LlmResponse:
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()
            return LlmResponse(content="late answer")

    client = StubbornClient()
    agent = Agent(client, ToolRegistry(tmp_path), "system")
    running = asyncio.create_task(agent.run("wait"))
    await client.started.wait()
    agent.cancel()

    with pytest.raises(AgentCanceled):
        await asyncio.wait_for(running, 0.5)
    assert cancellation_module._DETACHED_CANCELLATIONS
    assert [message.role for message in agent.history] == ["user"]

    release.set()
    for _ in range(50):
        if not cancellation_module._DETACHED_CANCELLATIONS:
            break
        await asyncio.sleep(0.01)
    assert not cancellation_module._DETACHED_CANCELLATIONS
    assert [message.role for message in agent.history] == ["user"]


async def test_next_run_fences_late_streaming_deltas_from_canceled_request(
    tmp_path: Path,
) -> None:
    release = asyncio.Event()

    class LateDeltaClient(LlmClient):
        provider = "late-delta"
        model = "late-delta"

        def __init__(self) -> None:
            self.calls = 0
            self.started = asyncio.Event()

        async def complete(
            self,
            messages: list[Message],
            tools: list[dict[str, Any]] | None = None,
        ) -> LlmResponse:
            raise AssertionError("streaming path expected")

        async def complete_streaming(
            self,
            messages: list[Message],
            tools: list[dict[str, Any]] | None = None,
            on_delta: Any = None,
        ) -> LlmResponse:
            self.calls += 1
            if self.calls == 1:
                on_delta("first")
                self.started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    await release.wait()
                on_delta("stale")
                return LlmResponse(content="firststale", streamed=True)
            on_delta("fresh")
            return LlmResponse(content="fresh", streamed=True)

    client = LateDeltaClient()
    agent = Agent(client, ToolRegistry(tmp_path), "system")
    deltas: list[str] = []
    agent.on_content_delta = deltas.append
    first = asyncio.create_task(agent.run("first turn"))
    await client.started.wait()
    agent.cancel()

    with pytest.raises(AgentCanceled):
        await asyncio.wait_for(first, 0.5)
    assert await agent.run("second turn") == "fresh"
    release.set()
    for _ in range(50):
        if not cancellation_module._DETACHED_CANCELLATIONS:
            break
        await asyncio.sleep(0.01)

    assert not cancellation_module._DETACHED_CANCELLATIONS
    assert deltas == ["first", "fresh"]
    assert [message.content for message in agent.history] == [
        "first turn",
        "second turn",
        "fresh",
    ]


async def test_agent_rejects_concurrent_run_then_recovers_after_cancel(
    tmp_path: Path,
) -> None:
    class SerializedClient(LlmClient):
        provider = "serialized"
        model = "serialized"

        def __init__(self) -> None:
            self.calls = 0
            self.started = asyncio.Event()

        async def complete(
            self,
            messages: list[Message],
            tools: list[dict[str, Any]] | None = None,
        ) -> LlmResponse:
            self.calls += 1
            if self.calls == 1:
                self.started.set()
                await asyncio.Event().wait()
            return LlmResponse(content="recovered")

    client = SerializedClient()
    agent = Agent(client, ToolRegistry(tmp_path), "system")
    first = asyncio.create_task(agent.run("first turn"))
    await client.started.wait()

    with pytest.raises(RuntimeError, match="already has an active run"):
        await agent.run("must not enter history")
    with pytest.raises(RuntimeError, match="already has an active run"):
        await agent.compact()
    with pytest.raises(RuntimeError, match="already has an active run"):
        agent.clear()
    with pytest.raises(RuntimeError, match="already has an active run"):
        agent.add_system_context("must not race")
    agent.cancel()
    with pytest.raises(AgentCanceled):
        await asyncio.wait_for(first, 1)

    assert await agent.run("next turn") == "recovered"
    assert client.calls == 2
    assert [message.content for message in agent.history] == [
        "first turn",
        "next turn",
        "recovered",
    ]


class RetryClient(LlmClient):
    provider = "retry"
    model = "retry"

    def __init__(self, failures: int, *, emit_before_failure: bool = False) -> None:
        self.failures = failures
        self.emit_before_failure = emit_before_failure
        self.calls = 0
        self.failed = asyncio.Event()

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        raise AssertionError("streaming path expected")

    async def complete_streaming(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        on_delta: Any = None,
    ) -> LlmResponse:
        self.calls += 1
        if self.calls <= self.failures:
            if self.emit_before_failure and on_delta is not None:
                callback = on_delta("partial")
                if asyncio.iscoroutine(callback):
                    await callback
            self.failed.set()
            raise LlmError("temporary", retryable=True)
        return LlmResponse(content="recovered")


async def test_retryable_model_failure_recovers_without_duplicate_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAIROCLI_LLM_RETRY_BASE_SECONDS", "0")
    client = RetryClient(2)
    agent = Agent(client, ToolRegistry(tmp_path), "system")

    assert await agent.run("work") == "recovered"
    assert client.calls == 3
    assert [message.content for message in agent.history] == ["work", "recovered"]
    assert agent.last_response_streamed is False


async def test_model_retry_stops_after_any_streamed_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAIROCLI_LLM_RETRY_BASE_SECONDS", "0")
    client = RetryClient(2, emit_before_failure=True)
    agent = Agent(client, ToolRegistry(tmp_path), "system")
    deltas: list[str] = []
    agent.on_content_delta = deltas.append

    with pytest.raises(LlmError, match="temporary"):
        await agent.run("work")

    assert client.calls == 1
    assert deltas == ["partial"]
    assert [message.role for message in agent.history] == ["user"]


async def test_cancel_interrupts_model_retry_backoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAIROCLI_LLM_RETRY_BASE_SECONDS", "10")
    client = RetryClient(2)
    agent = Agent(client, ToolRegistry(tmp_path), "system")
    running = asyncio.create_task(agent.run("work"))
    await client.failed.wait()
    await asyncio.sleep(0)
    agent.cancel()

    with pytest.raises(AgentCanceled):
        await asyncio.wait_for(running, 1)
    assert client.calls == 1


class AuxiliaryRetryClient(LlmClient):
    provider = "retry"
    model = "retry"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        self.calls += 1
        if self.calls == 1:
            raise LlmError("temporary auxiliary failure", retryable=True)
        return LlmResponse(content="auxiliary recovered", usage=Usage(3, 2))


async def test_auxiliary_planner_and_compaction_requests_share_retry_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAIROCLI_LLM_RETRY_BASE_SECONDS", "0")
    client = AuxiliaryRetryClient()
    agent = Agent(client, ToolRegistry(tmp_path), "system")

    response = await agent.complete_auxiliary(
        [Message("user", "plan")], "planner", agent.cancel_event
    )

    assert response.content == "auxiliary recovered"
    assert client.calls == 2
    assert agent.llm_call_count == 1
    assert agent.total_input_tokens == 3


async def test_plan_workers_share_parent_cancellation(tmp_path: Path) -> None:
    client = BlockingClient(
        LlmResponse(content='{"tasks":[{"id":"slow","description":"wait","dependencies":[]}]}')
    )
    base = Agent(client, ToolRegistry(tmp_path), "system")
    run = asyncio.create_task(PlanExecuteAgent(base).run("slow plan"))
    await client.started.wait()
    base.cancel()
    with pytest.raises(AgentCanceled):
        await asyncio.wait_for(run, 1)
    assert client.was_canceled


async def test_plan_review_wait_shares_parent_cancellation(tmp_path: Path) -> None:
    client = BlockingClient(
        LlmResponse(content='{"tasks":[{"id":"review","description":"wait","dependencies":[]}]}')
    )
    base = Agent(client, ToolRegistry(tmp_path), "system")
    review_started = asyncio.Event()
    review_canceled = asyncio.Event()

    async def review(_plan: ExecutionPlan) -> bool:
        review_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            review_canceled.set()
            raise
        return True

    run = asyncio.create_task(PlanExecuteAgent(base, review_handler=review).run("slow review"))
    await review_started.wait()
    base.cancel()

    with pytest.raises(AgentCanceled):
        await asyncio.wait_for(run, 1)
    assert review_canceled.is_set()


class ImageToolClient(LlmClient):
    provider = "glm"
    model = "vision"

    def __init__(self) -> None:
        self.calls = 0
        self.second_messages: list[Message] = []

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        self.calls += 1
        if self.calls == 1:
            return LlmResponse(tool_calls=[ToolCall("image-1", "screenshot", {})])
        self.second_messages = messages
        return LlmResponse(content="image analyzed")


async def test_tool_images_are_added_after_all_tool_results(tmp_path: Path) -> None:
    client = ImageToolClient()
    registry = ToolRegistry(tmp_path)

    async def screenshot(arguments: dict[str, object]) -> ToolOutput:
        return ToolOutput("screenshot captured", ("data:image/png;base64,aGVsbG8=",))

    registry.register(ToolDefinition("screenshot", "test", {"type": "object"}, screenshot))
    agent = Agent(client, registry, "system")
    assert await agent.run("inspect") == "image analyzed"
    roles = [message.role for message in agent.history]
    assert roles == ["user", "assistant", "tool", "user", "assistant"]
    image_message = client.second_messages[-1]
    assert isinstance(image_message.content, list)
    assert image_message.content[1]["image_url"]["url"].startswith("data:image/png")


async def test_new_turn_prunes_historical_image_payloads(tmp_path: Path) -> None:
    client = ImageToolClient()
    client.calls = 1
    agent = Agent(client, ToolRegistry(tmp_path), "system")
    agent.history.append(
        Message(
            "user",
            [
                {"type": "text", "text": "old screenshot"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,old"}},
            ],
        )
    )
    await agent.run("new task")
    old_content = agent.history[0].content
    assert isinstance(old_content, list)
    assert all(part["type"] != "image_url" for part in old_content)
    assert "Omitted 1 historical" in old_content[-1]["text"]


async def test_agent_parses_image_references_outside_cli_path(tmp_path: Path) -> None:
    class RecordingImageClient(LlmClient):
        provider = "glm"
        model = "vision"

        def __init__(self) -> None:
            self.messages: list[Message] = []

        async def complete(
            self, messages: list[Message], tools: list[dict[str, Any]] | None = None
        ) -> LlmResponse:
            self.messages = messages
            return LlmResponse(content="seen")

    image = tmp_path / "shot.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    client = RecordingImageClient()
    agent = Agent(client, ToolRegistry(tmp_path), "system")

    assert await agent.run("inspect @image:shot.png") == "seen"
    content = client.messages[-1].content
    assert isinstance(content, list)
    assert content[-1]["type"] == "image_url"
    assert "Inspect them directly" in content[0]["text"]
