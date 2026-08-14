from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from typing import Any

from ..cancellation import AgentCanceled as AgentCanceled
from ..models import Message
from ..plan import PlanTask, TaskStatus
from ..text_safety import bound_utf8, safe_text
from .core import Agent
from .planning import PlanExecuteAgent, _extract_json

MAX_PLAN_TASKS = 64
MAX_PLAN_TASK_ID_CHARS = 128
MAX_PLAN_TASK_DESCRIPTION_BYTES = 16 * 1024
MAX_PLAN_DEPENDENCIES = 64
MAX_PLAN_JSON_BYTES = 1024 * 1024
MAX_PLAN_JSON_DEPTH = 16
MAX_PLAN_JSON_NODES = 10_000
MAX_REVIEW_ISSUES = 64
MAX_REVIEW_ISSUE_BYTES = 2_000
MAX_ORCHESTRATION_GOAL_BYTES = 256 * 1024
MAX_ORCHESTRATION_DEPENDENCY_BYTES = 256 * 1024
MAX_ORCHESTRATION_RESULTS_BYTES = 512 * 1024
MAX_ORCHESTRATION_FAILURE_BYTES = 64 * 1024
MAX_ORCHESTRATION_REVIEW_BYTES = 256 * 1024
TOOL_OBSERVER_TIMEOUT_SECONDS = 2.0
TOOL_OBSERVER_CANCEL_GRACE_SECONDS = 0.1
_PLAN_TASK_ID = re.compile(r"[A-Za-z0-9_.:-]+\Z")
_DETACHED_TOOL_OBSERVERS: set[asyncio.Future[Any]] = set()


class AgentOrchestrator:
    def __init__(
        self,
        agent: Agent,
        max_concurrency: int = 2,
        max_retries_per_step: int = 2,
    ) -> None:
        self.agent = agent
        self.max_concurrency = max(1, max_concurrency)
        self.max_retries_per_step = max(0, max_retries_per_step)
        self.on_plan_created: Callable[[Any], Any] | None = None
        self.on_task_started: Callable[[PlanTask], Any] | None = None
        self.on_task_completed: Callable[[PlanTask, bool], Any] | None = None

    async def run(self, task: str, image_urls: list[str] | None = None) -> str:
        self.agent.cancel_event.clear()
        self.agent.reset_run_budget()
        planner = PlanExecuteAgent(self.agent)
        plan = await planner.create_plan(
            "Divide this goal into independently executable specialist assignments where possible: "
            + task,
            cancel_event=self.agent.cancel_event,
        )
        if self.on_plan_created is not None:
            self.on_plan_created(plan)
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def worker(assignment: PlanTask) -> tuple[PlanTask, str, Exception | None]:
            dependency_context = _bounded_orchestration_entries(
                [
                    f"[{dependency}] {plan.tasks[dependency].result}"
                    for dependency in assignment.dependencies
                    if plan.tasks[dependency].result
                ],
                MAX_ORCHESTRATION_DEPENDENCY_BYTES,
            )
            feedback = ""
            accepted_result = ""
            try:
                async with semaphore:
                    for attempt in range(self.max_retries_per_step + 1):
                        role_agent = Agent(
                            self.agent.llm,
                            self.agent.tools,
                            self.agent.base_system_prompt,
                            self.agent.budget,
                            self.agent.cancel_event,
                            self.agent.memory_store,
                            self.agent.image_cache_dir,
                            self.agent.trace_logger,
                            "team-worker",
                            self.agent._shared_token_budget,
                            pricing=self.agent.pricing,
                        )
                        self.agent.attach_tool_observers(role_agent)
                        prompt = (
                            f"You are the worker responsible for: {assignment.description}\n"
                            "Overall goal: "
                            f"{_truncate_orchestration_text(task, MAX_ORCHESTRATION_GOAL_BYTES)}\n"
                            f"Completed dependency results:\n{dependency_context or '(none)'}\n"
                            "Complete and verify your assignment."
                        )
                        if feedback:
                            prompt += (
                                "\n\nThe reviewer rejected the previous result. Correct these "
                                "issues:\n"
                                + _truncate_orchestration_text(
                                    feedback, MAX_ORCHESTRATION_FAILURE_BYTES
                                )
                            )
                        try:
                            accepted_result = await role_agent.run(
                                prompt,
                                image_urls,
                                reset_cancellation=False,
                            )
                        finally:
                            self.agent.absorb_usage(role_agent)
                        if not accepted_result.strip():
                            raise RuntimeError("Worker returned an empty result")
                        try:
                            review = await self._review_step(assignment, accepted_result)
                        except AgentCanceled:
                            raise
                        except Exception:
                            # A reviewer outage must not discard otherwise valid work.
                            return assignment, accepted_result, None
                        approved, feedback = _parse_review(review)
                        feedback = _truncate_orchestration_text(
                            feedback, MAX_ORCHESTRATION_FAILURE_BYTES
                        )
                        if approved or attempt == self.max_retries_per_step:
                            return assignment, accepted_result, None
                return assignment, accepted_result, None
            except AgentCanceled:
                raise
            except Exception as exc:
                return assignment, "", exc

        while not plan.complete:
            ready = plan.ready()
            if not ready:
                break
            for assignment in ready:
                assignment.status = TaskStatus.RUNNING
                if self.on_task_started is not None:
                    self.on_task_started(assignment)
            batch = await asyncio.gather(*(worker(item) for item in ready))
            for assignment, result, error in batch:
                if error is not None:
                    assignment.status = TaskStatus.FAILED
                    assignment.result = safe_text(
                        error,
                        fallback=f"{type(error).__name__} message unavailable",
                    )
                    if self.on_task_completed is not None:
                        self.on_task_completed(assignment, False)
                else:
                    assignment.status = TaskStatus.COMPLETED
                    assignment.result = result
                    if self.on_task_completed is not None:
                        self.on_task_completed(assignment, True)

        assignments = list(plan.tasks.values())
        assignment_results = [
            f"[{assignment.id} — {assignment.description} — {assignment.status}]\n"
            f"{assignment.result}"
            for assignment in assignments
        ]
        review_prompt = (
            "You are the final reviewer. Original goal: "
            + _truncate_orchestration_text(task, MAX_ORCHESTRATION_GOAL_BYTES)
            + "\n\nWorker results:\n"
            + _bounded_orchestration_entries(assignment_results, MAX_ORCHESTRATION_RESULTS_BYTES)
            + "\n\nReconcile conflicts, check completeness, fix remaining issues with tools "
            "if needed, "
            "and return one final answer."
        )
        return await self.agent.run(review_prompt, reset_cancellation=False)

    async def _review_step(self, assignment: PlanTask, result: str) -> str:
        response = await self.agent.complete_auxiliary(
            [
                Message(
                    "system",
                    "You are a strict Kairo CLI reviewer. Return only JSON with boolean "
                    "approved and an issues array of concise correction requests.",
                ),
                Message(
                    "user",
                    f"Step: {assignment.description}\n\nWorker result:\n"
                    + _truncate_orchestration_text(result, MAX_ORCHESTRATION_REVIEW_BYTES),
                ),
            ],
            "team-reviewer",
            self.agent.cancel_event,
        )
        return response.content


def _parse_review(content: str) -> tuple[bool, str]:
    """Parse reviewer output conservatively, matching the Java approval policy."""
    try:
        payload = _extract_json(content)
        approved = payload.get("approved") is True
        issues_value = payload.get("issues", payload.get("suggestions", []))
        if (
            not isinstance(issues_value, list)
            or len(issues_value) > MAX_REVIEW_ISSUES
            or any(
                not isinstance(item, str) or len(item.encode("utf-8")) > MAX_REVIEW_ISSUE_BYTES
                for item in issues_value
            )
        ):
            raise ValueError("Reviewer issues must be a bounded string array")
        issues = "\n".join(f"- {item.strip()}" for item in issues_value if item.strip())
        return approved, issues or "Reviewer did not approve the result."
    except (RecursionError, TypeError, ValueError):
        normalized = content.casefold()
        if "{" in normalized and "}" in normalized:
            return False, content.strip() or "Reviewer returned invalid JSON."
        negative = any(
            marker in normalized
            for marker in ("未通过", "不通过", "不合格", "有问题", '"approved": false')
        )
        positive = any(marker in normalized for marker in ("通过", "合格", '"approved": true'))
        return positive and not negative, content.strip() or "Reviewer returned no feedback."


def _truncate_orchestration_text(value: str, max_bytes: int) -> str:
    return bound_utf8(value, max_bytes, "\n...[orchestration context truncated]")


def _bounded_orchestration_entries(entries: list[str], max_bytes: int) -> str:
    if not entries:
        return ""
    separator = "\n\n"
    separator_bytes = len(separator.encode("utf-8")) * (len(entries) - 1)
    available = max(1, max_bytes - separator_bytes)
    per_entry = max(1, available // len(entries))
    bounded = [_truncate_orchestration_text(entry, per_entry) for entry in entries]
    return separator.join(bounded)
