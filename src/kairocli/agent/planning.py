from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from typing import Any

from ..cancellation import AgentCanceled as AgentCanceled
from ..cancellation import wait_with_cancellation
from ..json_boundary import decode_strict_json
from ..models import Message
from ..plan import ExecutionPlan, PlanTask, TaskStatus
from ..text_safety import bound_utf8, safe_text
from .core import Agent

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


class PlanExecuteAgent:
    def __init__(
        self,
        agent: Agent,
        review_handler: Callable[[ExecutionPlan], Any] | None = None,
        max_concurrency: int = 4,
        max_replans: int = 1,
    ) -> None:
        self.agent = agent
        self.review_handler = review_handler
        self.max_concurrency = max(1, max_concurrency)
        self.max_replans = max(0, max_replans)
        self.on_plan_created: Callable[[ExecutionPlan], Any] | None = None
        self.on_task_started: Callable[[PlanTask], Any] | None = None
        self.on_task_completed: Callable[[PlanTask, bool], Any] | None = None

    async def run(self, task: str, image_urls: list[str] | None = None) -> str:
        self.agent.cancel_event.clear()
        self.agent.reset_run_budget()
        plan = await self.create_plan(task, cancel_event=self.agent.cancel_event)
        reviewed = await self._review_plan(task, plan)
        if reviewed is None:
            return "Plan canceled."
        plan = reviewed
        results: list[str] = []
        replans_remaining = self.max_replans
        while not plan.complete:
            ready = plan.ready()
            if not ready:
                break
            semaphore = asyncio.Semaphore(self.max_concurrency)

            async def execute(
                plan_task: PlanTask,
                batch_semaphore: asyncio.Semaphore = semaphore,
                current_plan: ExecutionPlan = plan,
            ) -> tuple[PlanTask, str, Exception | None]:
                plan_task.status = TaskStatus.RUNNING
                if self.on_task_started is not None:
                    self.on_task_started(plan_task)
                dependency_results = [
                    f"[{item}] {current_plan.tasks[item].result}"
                    for item in plan_task.dependencies
                    if current_plan.tasks[item].result
                ]
                context = _bounded_orchestration_entries(
                    dependency_results, MAX_ORCHESTRATION_DEPENDENCY_BYTES
                )
                bounded_goal = _truncate_orchestration_text(task, MAX_ORCHESTRATION_GOAL_BYTES)
                prompt = (
                    f"Overall goal: {bounded_goal}"
                    f"\n\nAssigned plan step: {plan_task.description}\n"
                    f"Dependency results:\n{context or '(none)'}\n\n"
                    "Execute this step and verify its result."
                )
                worker = Agent(
                    self.agent.llm,
                    self.agent.tools,
                    self.agent.base_system_prompt,
                    self.agent.budget,
                    self.agent.cancel_event,
                    self.agent.memory_store,
                    self.agent.image_cache_dir,
                    self.agent.trace_logger,
                    "plan-worker",
                    self.agent._shared_token_budget,
                    pricing=self.agent.pricing,
                )
                self.agent.attach_tool_observers(worker)
                async with batch_semaphore:
                    try:
                        result = await worker.run(prompt, image_urls, reset_cancellation=False)
                        return plan_task, result, None
                    except AgentCanceled:
                        raise
                    except Exception as exc:
                        return plan_task, "", exc
                    finally:
                        self.agent.absorb_usage(worker)

            batch = await asyncio.gather(*(execute(item) for item in ready))
            batch_errors: list[str] = []
            for plan_task, result, error in batch:
                if error:
                    plan_task.status = TaskStatus.FAILED
                    error_text = safe_text(
                        error,
                        fallback=f"{type(error).__name__} message unavailable",
                    )
                    plan_task.result = error_text
                    batch_errors.append(f"{plan_task.id}: {error_text}")
                    if self.on_task_completed is not None:
                        self.on_task_completed(plan_task, False)
                else:
                    plan_task.status = TaskStatus.COMPLETED
                    plan_task.result = result
                    results.append(f"[{plan_task.id}] {result}")
                    if self.on_task_completed is not None:
                        self.on_task_completed(plan_task, True)
            completed = sum(item.status == TaskStatus.COMPLETED for item in plan.tasks.values())
            if batch_errors and replans_remaining > 0 and completed / len(plan.tasks) < 0.5:
                feedback = (
                    "The previous plan failed early. Avoid the failure and preserve useful "
                    "completed work.\nFailures:\n- "
                    + _bounded_orchestration_entries(batch_errors, MAX_ORCHESTRATION_FAILURE_BYTES)
                    + "\nCompleted results:\n"
                    + (
                        _bounded_orchestration_entries(results, MAX_ORCHESTRATION_RESULTS_BYTES)
                        or "(none)"
                    )
                )
                replanned = await self.create_plan(task, feedback, self.agent.cancel_event)
                reviewed = await self._review_plan(task, replanned)
                if reviewed is None:
                    return "Plan canceled."
                plan = reviewed
                results = []
                replans_remaining -= 1
        failures = [item for item in plan.tasks.values() if item.status != TaskStatus.COMPLETED]
        synthesis = (
            "Original goal: "
            + _truncate_orchestration_text(task, MAX_ORCHESTRATION_GOAL_BYTES)
            + "\n\nPlan execution results:\n"
            + (_bounded_orchestration_entries(results, MAX_ORCHESTRATION_RESULTS_BYTES) or "(none)")
        )
        if failures:
            failure_entries = [f"- {item.id}: {item.status}: {item.result}" for item in failures]
            synthesis += "\n\nFailed or blocked steps:\n" + _bounded_orchestration_entries(
                failure_entries, MAX_ORCHESTRATION_FAILURE_BYTES
            )
        synthesis += (
            "\n\nReview the work, resolve any remaining issue if possible, and answer the user."
        )
        return await self.agent.run(synthesis, reset_cancellation=False)

    async def _review_plan(self, task: str, plan: ExecutionPlan) -> ExecutionPlan | None:
        while self.review_handler is not None:
            review = self.review_handler(plan)
            if asyncio.iscoroutine(review):
                review = await wait_with_cancellation(review, self.agent.cancel_event)
            if review is False:
                return None
            if not isinstance(review, str) or not review.strip():
                break
            plan = await self.create_plan(task, review.strip(), self.agent.cancel_event)
        if self.on_plan_created is not None:
            self.on_plan_created(plan)
        return plan

    async def create_plan(
        self,
        task: str,
        feedback: str = "",
        cancel_event: asyncio.Event | None = None,
    ) -> ExecutionPlan:
        system = Message(
            "system",
            "你是 Kairo CLI planner（规划器）。只返回包含 tasks 数组的 JSON。每个任务必须包含 "
            "id、description 和 dependencies（任务 ID 数组）。"
            "任务应当可执行、依赖关系明确且数量精简。"
            "每个 description 必须使用简体中文，以祈使句说明要执行的操作；"
            "不得直接复制用户的问题作为步骤描述。",
        )
        bounded_task = _truncate_orchestration_text(task, MAX_ORCHESTRATION_GOAL_BYTES)
        prompt = (
            bounded_task
            if not feedback
            else "任务："
            + bounded_task
            + "\n审阅反馈："
            + _truncate_orchestration_text(feedback, MAX_ORCHESTRATION_REVIEW_BYTES)
        )
        response = await self.agent.complete_auxiliary(
            [system, Message("user", prompt)], "planner", cancel_event
        )
        try:
            payload = _extract_json(response.content)
            raw_tasks = payload["tasks"]
            if not isinstance(raw_tasks, list) or not 1 <= len(raw_tasks) <= MAX_PLAN_TASKS:
                raise ValueError(f"Planner must return between 1 and {MAX_PLAN_TASKS} tasks")
            tasks = [
                PlanTask(
                    _required_plan_text(item, "id"),
                    _required_plan_text(item, "description"),
                    _plan_dependencies(item),
                )
                for item in raw_tasks
            ]
            if len(tasks) == 1 and tasks[0].description.strip() == task.strip():
                tasks[0].description = _fallback_plan_description()
            return ExecutionPlan(tasks)
        except (KeyError, RecursionError, TypeError, ValueError):
            return ExecutionPlan([PlanTask("task-1", _fallback_plan_description())])


def _fallback_plan_description() -> str:
    return "执行用户请求的工作并验证结果"


def _extract_json(content: str) -> dict[str, Any]:
    stripped = content.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.S)
    if fenced:
        stripped = fenced.group(1)
    else:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end >= start:
            stripped = stripped[start : end + 1]
    payload = decode_strict_json(
        stripped,
        max_bytes=MAX_PLAN_JSON_BYTES,
        max_depth=MAX_PLAN_JSON_DEPTH,
        max_nodes=MAX_PLAN_JSON_NODES,
    )
    if not isinstance(payload, dict):
        raise TypeError("Planner response must be an object")
    return payload


def _required_plan_text(item: Any, field: str) -> str:
    if not isinstance(item, dict):
        raise TypeError("Each planner task must be an object")
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Planner task {field} must be a non-empty string")
    normalized = value.strip()
    if field == "id":
        if len(normalized) > MAX_PLAN_TASK_ID_CHARS or not _PLAN_TASK_ID.fullmatch(normalized):
            raise ValueError("Planner task id contains invalid characters or is too long")
    elif len(normalized.encode("utf-8")) > MAX_PLAN_TASK_DESCRIPTION_BYTES:
        raise ValueError(
            f"Planner task description exceeds {MAX_PLAN_TASK_DESCRIPTION_BYTES} bytes"
        )
    return normalized


def _plan_dependencies(item: Any) -> set[str]:
    if not isinstance(item, dict):
        raise TypeError("Each planner task must be an object")
    dependencies = item.get("dependencies", [])
    if (
        not isinstance(dependencies, list)
        or len(dependencies) > MAX_PLAN_DEPENDENCIES
        or not all(
            isinstance(value, str)
            and 0 < len(value.strip()) <= MAX_PLAN_TASK_ID_CHARS
            and _PLAN_TASK_ID.fullmatch(value.strip())
            for value in dependencies
        )
    ):
        raise ValueError("Planner task dependencies must be an array of task IDs")
    return {value.strip() for value in dependencies}


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
