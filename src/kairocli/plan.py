from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, StrEnum


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class PlanReviewDecisionType(Enum):
    EXECUTE = "execute"
    SUPPLEMENT = "supplement"
    CANCEL = "cancel"


@dataclass(frozen=True, slots=True)
class PlanReviewDecision:
    type: PlanReviewDecisionType
    feedback: str | None = None


def parse_plan_review_input(value: str | None) -> PlanReviewDecision:
    if value == "\x1b":
        return PlanReviewDecision(PlanReviewDecisionType.CANCEL)
    normalized = (value or "").strip()
    if normalized.casefold() in {"", "y", "yes", "run", "/run"}:
        return PlanReviewDecision(PlanReviewDecisionType.EXECUTE)
    if normalized.casefold() in {"n", "no", "cancel", "esc", "/cancel"}:
        return PlanReviewDecision(PlanReviewDecisionType.CANCEL)
    if normalized.casefold().startswith("i "):
        normalized = normalized[2:].strip()
    return PlanReviewDecision(PlanReviewDecisionType.SUPPLEMENT, normalized)


@dataclass(slots=True)
class PlanTask:
    id: str
    description: str
    dependencies: set[str] = field(default_factory=set)
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""


class ExecutionPlan:
    def __init__(self, tasks: list[PlanTask]) -> None:
        if not tasks:
            raise ValueError("Plan must contain at least one task")
        self.tasks = {task.id: task for task in tasks}
        if len(self.tasks) != len(tasks):
            raise ValueError("Task IDs must be unique")
        self._validate()

    def _validate(self) -> None:
        for task in self.tasks.values():
            missing = task.dependencies - self.tasks.keys()
            if missing:
                raise ValueError(f"Task {task.id} has unknown dependencies: {sorted(missing)}")
            if task.id in task.dependencies:
                raise ValueError(f"Task {task.id} cannot depend on itself")
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError("Plan contains a dependency cycle")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in self.tasks[task_id].dependencies:
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in self.tasks:
            visit(task_id)

    def ready(self) -> list[PlanTask]:
        # Propagate blocking to a fixed point. A planner may emit tasks in any
        # order, so a single insertion-order pass can otherwise leave an
        # indirect dependent pending forever.
        changed = True
        while changed:
            changed = False
            for task in self.tasks.values():
                if task.status != TaskStatus.PENDING:
                    continue
                dependency_states = [self.tasks[item].status for item in task.dependencies]
                if any(
                    state in {TaskStatus.FAILED, TaskStatus.BLOCKED} for state in dependency_states
                ):
                    task.status = TaskStatus.BLOCKED
                    changed = True

        return [
            task
            for task in self.tasks.values()
            if task.status == TaskStatus.PENDING
            and all(self.tasks[item].status == TaskStatus.COMPLETED for item in task.dependencies)
        ]

    @property
    def complete(self) -> bool:
        return all(
            task.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.BLOCKED}
            for task in self.tasks.values()
        )
