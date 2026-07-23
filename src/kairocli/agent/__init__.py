"""Agent runtime, planning, and orchestration for Kairo CLI."""

from typing import TYPE_CHECKING, Any

from ..cancellation import AgentCanceled as AgentCanceled

if TYPE_CHECKING:
    from .core import Agent, AgentBudget
    from .orchestration import AgentOrchestrator
    from .planning import PlanExecuteAgent

__all__ = [
    "Agent",
    "AgentBudget",
    "AgentCanceled",
    "AgentOrchestrator",
    "PlanExecuteAgent",
    "markdown_fence_for",
]


def __getattr__(name: str) -> Any:
    if name in {
        "Agent",
        "AgentBudget",
        "_DETACHED_TOOL_OBSERVERS",
        "markdown_fence_for",
    }:
        from . import core

        return getattr(core, name)
    if name == "PlanExecuteAgent":
        from .planning import PlanExecuteAgent

        return PlanExecuteAgent
    if name in {
        "AgentOrchestrator",
        "_parse_review",
        "_truncate_orchestration_text",
    }:
        from . import orchestration

        return getattr(orchestration, name)
    raise AttributeError(name)
