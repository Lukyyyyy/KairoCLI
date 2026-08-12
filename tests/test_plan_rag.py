import asyncio
from pathlib import Path
from typing import Any

import pytest

from kairocli.agent import Agent, AgentBudget, AgentOrchestrator
from kairocli.llm import LlmClient
from kairocli.models import LlmResponse, Message, Usage
from kairocli.plan import (
    ExecutionPlan,
    PlanReviewDecisionType,
    PlanTask,
    TaskStatus,
    parse_plan_review_input,
)
from kairocli.rag import CodeChunker, CodeIndex, VectorStore
from kairocli.tools import ToolRegistry


def test_plan_dependency_order() -> None:
    first = PlanTask("a", "first")
    second = PlanTask("b", "second", {"a"})
    plan = ExecutionPlan([first, second])
    assert [task.id for task in plan.ready()] == ["a"]
    first.status = TaskStatus.COMPLETED
    assert [task.id for task in plan.ready()] == ["b"]


def test_plan_rejects_cycle() -> None:
    with pytest.raises(ValueError, match="cycle"):
        ExecutionPlan([PlanTask("a", "a", {"b"}), PlanTask("b", "b", {"a"})])


def test_plan_review_input_matches_reference_semantics() -> None:
    for value in (None, "", "y", "YES", "run", "/run"):
        assert parse_plan_review_input(value).type == PlanReviewDecisionType.EXECUTE
    for value in ("\x1b", "n", "no", "cancel", "esc", "/cancel"):
        assert parse_plan_review_input(value).type == PlanReviewDecisionType.CANCEL
    decision = parse_plan_review_input("add integration tests")
    assert decision.type == PlanReviewDecisionType.SUPPLEMENT
    assert decision.feedback == "add integration tests"
    assert parse_plan_review_input("i improve docs").feedback == "improve docs"


def test_plan_propagates_blocking_independent_of_task_order() -> None:
    plan = ExecutionPlan(
        [
            PlanTask("c", "third", {"b"}),
            PlanTask("b", "second", {"a"}),
            PlanTask("a", "first", status=TaskStatus.FAILED),
        ]
    )
    assert plan.ready() == []
    assert plan.tasks["b"].status == TaskStatus.BLOCKED
    assert plan.tasks["c"].status == TaskStatus.BLOCKED
    assert plan.complete


class OrchestratorClient(LlmClient):
    provider = "test"
    model = "test"

    def __init__(self) -> None:
        self.worker_prompts: list[str] = []

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        system = str(messages[0].content)
        prompt = str(messages[-1].content)
        if "Kairo CLI planner" in system:
            return LlmResponse(
                content=(
                    '{"tasks":['
                    '{"id":"a","description":"collect","dependencies":[]},'
                    '{"id":"b","description":"use collection","dependencies":["a"]}'
                    "]}"
                )
            )
        if "strict Kairo CLI reviewer" in system:
            return LlmResponse(content='{"approved":true,"issues":[]}')
        if "You are the worker responsible for:" in prompt:
            self.worker_prompts.append(prompt)
            if "use collection" in prompt:
                assert "result:collect" in prompt
                return LlmResponse(content="result:use collection")
            return LlmResponse(content="result:collect")
        return LlmResponse(content="final answer")


async def test_orchestrator_respects_dependencies_and_reviews_steps(tmp_path: Path) -> None:
    client = OrchestratorClient()
    orchestrator = AgentOrchestrator(
        Agent(client, ToolRegistry(tmp_path), "system"), max_concurrency=2
    )
    events: list[tuple[str, str, bool | None]] = []
    orchestrator.on_plan_created = lambda plan: events.append(
        ("plan", ",".join(plan.tasks), None)
    )
    orchestrator.on_task_started = lambda task: events.append(("started", task.id, None))
    orchestrator.on_task_completed = lambda task, success: events.append(
        ("completed", task.id, success)
    )

    assert await orchestrator.run("build") == "final answer"
    assert len(client.worker_prompts) == 2
    assert "(none)" in client.worker_prompts[0]
    assert "[a] result:collect" in client.worker_prompts[1]
    assert orchestrator.agent.llm_call_count == 6
    assert events == [
        ("plan", "a,b", None),
        ("started", "a", None),
        ("completed", "a", True),
        ("started", "b", None),
        ("completed", "b", True),
    ]


class RetryOrchestratorClient(LlmClient):
    provider = "test"
    model = "test"

    def __init__(self) -> None:
        self.worker_calls = 0
        self.review_calls = 0

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        system = str(messages[0].content)
        prompt = str(messages[-1].content)
        if "Kairo CLI planner" in system:
            return LlmResponse(
                content='{"tasks":[{"id":"a","description":"fix","dependencies":[]}]}'
            )
        if "strict Kairo CLI reviewer" in system:
            self.review_calls += 1
            if self.review_calls == 1:
                return LlmResponse(content='{"approved":false,"issues":["add verification"]}')
            return LlmResponse(content='{"approved":true,"issues":[]}')
        if "You are the worker responsible for:" in prompt:
            self.worker_calls += 1
            if self.worker_calls == 2:
                assert "add verification" in prompt
                return LlmResponse(content="corrected")
            return LlmResponse(content="draft")
        return LlmResponse(content="final")


async def test_orchestrator_retries_rejected_step_with_feedback(tmp_path: Path) -> None:
    client = RetryOrchestratorClient()
    orchestrator = AgentOrchestrator(Agent(client, ToolRegistry(tmp_path), "system"))
    assert await orchestrator.run("fix it") == "final"
    assert client.worker_calls == 2
    assert client.review_calls == 2


class LargeTeamContextClient(LlmClient):
    provider = "test"
    model = "large-team-context"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        system = str(messages[0].content)
        prompt = str(messages[-1].content)
        self.prompts.append(prompt)
        if "Kairo CLI planner" in system:
            return LlmResponse(
                content=(
                    '{"tasks":['
                    '{"id":"a","description":"produce","dependencies":[]},'
                    '{"id":"b","description":"consume","dependencies":["a"]}'
                    "]}"
                )
            )
        if "strict Kairo CLI reviewer" in system:
            return LlmResponse(content='{"approved":true,"issues":[]}')
        if "responsible for: produce" in prompt:
            return LlmResponse(content="界" * 300_000)
        if "responsible for: consume" in prompt:
            assert "[orchestration context truncated]" in prompt
            return LlmResponse(content="consumed")
        assert "[orchestration context truncated]" in prompt
        return LlmResponse(content="team final")


async def test_orchestrator_bounds_worker_review_dependency_and_final_context(
    tmp_path: Path,
) -> None:
    client = LargeTeamContextClient()
    result = await AgentOrchestrator(Agent(client, ToolRegistry(tmp_path), "system")).run(
        "process large result"
    )

    assert result == "team final"
    assert all(len(prompt.encode("utf-8")) < 1024 * 1024 for prompt in client.prompts)


class ConcurrentTeamBudgetClient(LlmClient):
    provider = "test"
    model = "shared-team-budget"

    def __init__(self) -> None:
        self.worker_calls = 0
        self.in_flight = 0
        self.peak_in_flight = 0

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        system = str(messages[0].content)
        prompt = str(messages[-1].content)
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0.01)
            if "Kairo CLI planner" in system:
                return LlmResponse(
                    content=(
                        '{"tasks":['
                        '{"id":"a","description":"first","dependencies":[]},'
                        '{"id":"b","description":"second","dependencies":[]}'
                        "]}"
                    ),
                    usage=Usage(input_tokens=10),
                )
            if "You are the worker responsible for:" in prompt:
                self.worker_calls += 1
                return LlmResponse(content="worker result", usage=Usage(input_tokens=60))
            raise AssertionError("Reviewer/final request must not start after exhaustion")
        finally:
            self.in_flight -= 1


async def test_team_hard_budget_serializes_parallel_model_requests(
    tmp_path: Path,
) -> None:
    client = ConcurrentTeamBudgetClient()
    base = Agent(
        client,
        ToolRegistry(tmp_path),
        "system",
        AgentBudget(token_budget=70),
    )

    with pytest.raises(RuntimeError, match=r"budget exhausted \(70 / 70\)"):
        await AgentOrchestrator(base).run("parallel work")

    assert client.worker_calls == 1
    assert client.peak_in_flight == 1
    assert base.total_input_tokens == 70


def test_chunker_and_vector_search(tmp_path: Path) -> None:
    chunks = CodeChunker(max_lines=3, overlap=1).chunk("a.py", "a\nb\nc\nd")
    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [(1, 3), (3, 4)]
    store = VectorStore(tmp_path / "index.db")
    store.upsert(chunks, [[1.0, 0.0], [0.0, 1.0]])
    assert store.search([1.0, 0.0], 1)[0][0].id == chunks[0].id


async def test_code_index_search_and_graph(tmp_path: Path) -> None:
    (tmp_path / "service.py").write_text(
        "class PaymentService:\n    def charge(self):\n        return 'paid'\n", encoding="utf-8"
    )
    index = CodeIndex(tmp_path, tmp_path / ".kairocli" / "index.db")
    summary = await index.index()
    assert summary == {"files": 1, "chunks": 2}
    assert (await index.search("payment charge", 1))[0]["path"] == "service.py"
    assert index.graph("PaymentService")[0]["kind"] == "definition"
