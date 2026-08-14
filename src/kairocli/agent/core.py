from __future__ import annotations

import asyncio
import copy
import inspect
import json
import os
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..cancellation import AgentCanceled as AgentCanceled
from ..cancellation import raise_if_canceled, wait_with_cancellation
from ..image import prepare_image_input
from ..llm import LlmClient, LlmError
from ..memory import MemoryStore, browser_login_fact
from ..models import LlmResponse, Message, ToolCall, ToolOutput
from ..pricing import PricingConfig
from ..tools import SERIALIZED_WORKSPACE_MUTATION_TOOLS, ToolRegistry
from ..trace import LlmTraceLogger
from ..user_input import normalize_user_input
from .compaction import ConversationCompactor
from .context import (
    ContextProfile,
    estimate_message_tokens,
    estimate_schema_tokens,
    estimated_cost_cny,
)

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


@dataclass(slots=True)
class AgentBudget:
    max_iterations: int = 50
    token_budget: int | None = None
    stagnation_window: int = 3

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.token_budget is not None and self.token_budget <= 0:
            raise ValueError("token_budget must be positive when configured")
        if self.stagnation_window < 2:
            raise ValueError("stagnation_window must be at least 2")

    @classmethod
    def from_environment(cls) -> AgentBudget:
        token = _positive_int(os.getenv("KAIROCLI_REACT_TOKEN_BUDGET"))
        return cls(
            max_iterations=_positive_int(os.getenv("KAIROCLI_REACT_HARD_MAX_ITERATIONS"), 50) or 50,
            token_budget=token,
            stagnation_window=max(
                2,
                _positive_int(os.getenv("KAIROCLI_REACT_STAGNATION_WINDOW"), 3) or 3,
            ),
        )


@dataclass(slots=True)
class _SharedTokenBudget:
    limit: int
    used: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def charge(self, response: LlmResponse) -> None:
        self.used += max(0, response.usage.input_tokens)
        self.used += max(0, response.usage.output_tokens)


class Agent:
    def __init__(
        self,
        llm: LlmClient,
        tools: ToolRegistry,
        system_prompt: str,
        budget: AgentBudget | None = None,
        cancel_event: asyncio.Event | None = None,
        memory_store: MemoryStore | None = None,
        image_cache_dir: Path | None = None,
        trace_logger: LlmTraceLogger | None = None,
        trace_scope: str = "agent",
        shared_token_budget: _SharedTokenBudget | None = None,
        pricing: PricingConfig | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.base_system_prompt = system_prompt
        self.memory_store = memory_store
        self.image_cache_dir = image_cache_dir or (
            tools.workspace / ".kairocli" / "cache" / "clipboard"
        )
        self.budget = budget or AgentBudget.from_environment()
        self._shared_token_budget = shared_token_budget
        if self._shared_token_budget is None and self.budget.token_budget is not None:
            self._shared_token_budget = _SharedTokenBudget(self.budget.token_budget)
        self.context_profile = ContextProfile.from_client(llm)
        self.trace_logger = trace_logger
        self.trace_scope = trace_scope
        self.pricing = pricing or PricingConfig.default()
        self.compactor = ConversationCompactor(llm, complete_handler=self.complete_auxiliary)
        self.history: list[Message] = []
        self.cancel_event = cancel_event or asyncio.Event()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_tokens = 0
        self.llm_call_count = 0
        self.compaction_count = 0
        self.last_context_tokens = 0
        self.on_content_delta: Callable[[str], Any] | None = None
        self.on_reasoning: Callable[[str], Any] | None = None
        self.on_reasoning_delta: Callable[[str], Any] | None = None
        self.on_tool_calls: Callable[[list[ToolCall]], Any] | None = None
        self.on_tool_results: Callable[[list[ToolCall], list[ToolOutput]], Any] | None = None
        self.last_response_streamed = False
        self._run_generation = 0
        self._run_lock = asyncio.Lock()
        self.max_llm_retries = min(_nonnegative_int(os.getenv("KAIROCLI_LLM_MAX_RETRIES"), 2), 5)
        self.llm_retry_base_seconds = _bounded_float(
            os.getenv("KAIROCLI_LLM_RETRY_BASE_SECONDS"), 0.5, 0.0, 10.0
        )

    async def run(
        self,
        prompt: str,
        image_urls: list[str] | None = None,
        *,
        reset_cancellation: bool = True,
    ) -> str:
        if self._run_lock.locked():
            raise RuntimeError("Agent already has an active run")
        async with self._run_lock:
            return await self._run_once(
                prompt,
                image_urls,
                reset_cancellation=reset_cancellation,
            )

    async def _run_once(
        self,
        prompt: str,
        image_urls: list[str] | None = None,
        *,
        reset_cancellation: bool = True,
    ) -> str:
        prompt = normalize_user_input(prompt, max_bytes=None)
        if not prompt.strip():
            return ""
        self._run_generation += 1
        run_generation = self._run_generation
        if reset_cancellation:
            self.cancel_event.clear()
            self.reset_run_budget()
            self._strip_historical_images()
        if image_urls is None:
            prepared = await prepare_image_input(
                prompt,
                self.tools.workspace,
                self.image_cache_dir,
            )
            prompt = prepared.text
            image_urls = list(prepared.image_urls)
        prompt = normalize_user_input(prompt)
        self._refresh_memory_context(prompt)
        user_content: Any = prompt
        if image_urls:
            user_content = [
                {"type": "text", "text": prompt},
                *[
                    {"type": "image_url", "image_url": {"url": image_url}}
                    for image_url in image_urls
                ],
            ]
        self.history.append(Message("user", user_content))
        session_id: str | None = None
        session_start = time.monotonic()
        if self.trace_logger is not None:
            session_id = self.trace_logger.open_session(
                scope=self.trace_scope,
                provider=self.llm.provider,
                model=self.llm.model,
            )
        fingerprints: list[str] = []
        run_input_start = self.total_input_tokens
        run_output_start = self.total_output_tokens
        run_cached_start = self.total_cached_tokens
        turn = 0
        run_error: BaseException | None = None
        try:
            for _ in range(self.budget.max_iterations):
                raise_if_canceled(self.cancel_event)
                await self._compact_if_needed()
                messages = [Message("system", self.system_prompt), *self.history]
                tool_schemas = self.tools.schemas()
                turn += 1
                if self.trace_logger is not None:
                    await self.trace_logger.record_llm_request(
                        session_id=session_id,
                        turn=turn,
                        messages=messages,
                        tools=tool_schemas,
                    )
                request_started = time.monotonic()
                try:
                    async with self._budget_request() as shared_budget:
                        response = await self._complete_streaming_with_retry(
                            messages,
                            tool_schemas,
                            self.cancel_event,
                            run_generation,
                        )
                        raise_if_canceled(self.cancel_event)
                        self.record_usage(response)
                        if shared_budget is not None:
                            shared_budget.charge(response)
                except Exception as exc:
                    if self.trace_logger is not None:
                        await self.trace_logger.record_llm_error(
                            session_id=session_id,
                            turn=turn,
                            error=exc,
                            duration_ms=int((time.monotonic() - request_started) * 1_000),
                            msg_count=len(messages),
                            tool_count=len(tool_schemas),
                        )
                    raise
                if self.trace_logger is not None:
                    await self.trace_logger.record_llm_turn(
                        session_id=session_id,
                        turn=turn,
                        response=response,
                        duration_ms=int((time.monotonic() - request_started) * 1_000),
                        msg_count=len(messages),
                        tool_count=len(tool_schemas),
                    )
                self.last_response_streamed = response.streamed
                if not response.reasoning_streamed:
                    await self._emit_reasoning(response.reasoning_content)
                assistant = Message(
                    "assistant",
                    response.content,
                    tool_calls=response.tool_calls,
                    reasoning_content=response.reasoning_content,
                )
                if not response.tool_calls:
                    self.history.append(assistant)
                    return response.content
                run_tokens = self._current_run_tokens(run_input_start, run_output_start)
                if self.budget.token_budget is not None and run_tokens >= self.budget.token_budget:
                    raise RuntimeError(
                        "Agent token budget exceeded "
                        f"({run_tokens} / {self.budget.token_budget}); "
                        "additional tool calls were not executed"
                    )
                fingerprint = json.dumps(
                    [(call.name, call.arguments) for call in response.tool_calls], sort_keys=True
                )
                fingerprints.append(fingerprint)
                if (
                    len(fingerprints) >= self.budget.stagnation_window
                    and len(set(fingerprints[-self.budget.stagnation_window :])) == 1
                ):
                    raise RuntimeError(
                        "Agent stopped after "
                        f"{self.budget.stagnation_window} repeated identical tool-call rounds"
                    )
                tool_round_start = len(self.history)
                self.history.append(assistant)
                try:
                    await self._emit_tool_calls(response.tool_calls)
                    results = await self.tools.execute_many_outputs(
                        [(call.name, call.arguments) for call in response.tool_calls],
                        cancel_event=self.cancel_event,
                    )
                    raise_if_canceled(self.cancel_event)
                    await self._emit_tool_results(response.tool_calls, results)
                    raise_if_canceled(self.cancel_event)
                except (AgentCanceled, asyncio.CancelledError):
                    # A mutating handler may already have committed before cancellation.
                    # Preserve a valid provider protocol and make that ambiguity explicit
                    # instead of erasing evidence that the tool call ever happened.
                    canceled_results = [_canceled_tool_result(call) for call in response.tool_calls]
                    for call, canceled_result in zip(
                        response.tool_calls, canceled_results, strict=True
                    ):
                        self.history.append(
                            Message(
                                "tool",
                                canceled_result.text,
                                tool_call_id=call.id,
                            )
                        )
                    raise
                except BaseException:
                    # Unknown failures before a complete result batch retain the old
                    # rollback behavior; unlike cancellation, no stable outcome exists.
                    del self.history[tool_round_start:]
                    raise
                if self.trace_logger is not None:
                    await self.trace_logger.record_tool_results(
                        session_id=session_id,
                        calls=response.tool_calls,
                        results=results,
                    )
                for call, result in zip(response.tool_calls, results, strict=True):
                    self.history.append(Message("tool", result.text, tool_call_id=call.id))
                for call, result in zip(response.tool_calls, results, strict=True):
                    if result.has_images:
                        self.history.append(
                            Message(
                                "user",
                                [
                                    {
                                        "type": "text",
                                        "text": (
                                            f"Tool {call.name} returned image content. Analyze the "
                                            "attached image together with its tool result text."
                                        ),
                                    },
                                    *[
                                        {"type": "image_url", "image_url": {"url": image_url}}
                                        for image_url in result.image_urls
                                    ],
                                ],
                            )
                        )
            raise RuntimeError(f"Agent iteration limit exceeded ({self.budget.max_iterations})")
        except BaseException as exc:
            run_error = exc
            raise
        finally:
            if self.trace_logger is not None:
                await self.trace_logger.close_session(
                    session_id=session_id,
                    duration_ms=int((time.monotonic() - session_start) * 1_000),
                    total_calls=turn,
                    total_input=self.total_input_tokens - run_input_start,
                    total_output=self.total_output_tokens - run_output_start,
                    total_cached=self.total_cached_tokens - run_cached_start,
                    error=run_error,
                )

    async def _emit_tool_calls(self, calls: list[ToolCall]) -> None:
        callback = self.on_tool_calls
        if callback is None:
            return
        try:
            outcome = callback(_tool_call_snapshot(calls))
            if inspect.isawaitable(outcome):
                await _await_tool_observer(outcome)
        except Exception:
            # A presentation observer must never change Agent execution semantics.
            return

    async def _emit_reasoning(self, reasoning: str | None) -> None:
        callback = self.on_reasoning
        if callback is None or not reasoning or not reasoning.strip():
            return
        try:
            outcome = callback(reasoning)
            if inspect.isawaitable(outcome):
                await _await_tool_observer(outcome)
        except Exception:
            return

    async def _emit_tool_results(self, calls: list[ToolCall], results: list[ToolOutput]) -> None:
        callback = self.on_tool_results
        if callback is None:
            return
        try:
            outcome = callback(_tool_call_snapshot(calls), list(results))
            if inspect.isawaitable(outcome):
                await _await_tool_observer(outcome)
        except Exception:
            return

    def attach_tool_observers(self, child: Agent) -> None:
        """Share progress observers without interleaving child model text."""

        child.on_tool_calls = self.on_tool_calls
        child.on_tool_results = self.on_tool_results
        child.on_reasoning = self.on_reasoning
        child.on_reasoning_delta = self.on_reasoning_delta

    async def complete_auxiliary(
        self,
        messages: list[Message],
        scope: str,
        cancel_event: asyncio.Event | None = None,
    ) -> LlmResponse:
        event = cancel_event or self.cancel_event
        try:
            async with self._budget_request() as shared_budget:
                response = await self._retry_llm_request(
                    lambda: self.llm.complete(messages),
                    event,
                )
                raise_if_canceled(event)
                self.record_usage(response)
                if shared_budget is not None:
                    shared_budget.charge(response)
        except Exception:
            raise
        return response

    @asynccontextmanager
    async def _budget_request(self) -> AsyncIterator[_SharedTokenBudget | None]:
        shared = self._shared_token_budget
        if shared is None:
            yield None
            return
        async with shared.lock:
            if shared.used >= shared.limit:
                raise RuntimeError(
                    "Agent token budget exhausted "
                    f"({shared.used} / {shared.limit}); no new model request was started"
                )
            yield shared

    def reset_run_budget(self) -> None:
        if self.budget.token_budget is None:
            self._shared_token_budget = None
        else:
            self._shared_token_budget = _SharedTokenBudget(self.budget.token_budget)

    def _current_run_tokens(self, input_start: int, output_start: int) -> int:
        if self._shared_token_budget is not None:
            return self._shared_token_budget.used
        return (self.total_input_tokens - input_start) + (self.total_output_tokens - output_start)

    async def _complete_streaming_with_retry(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        cancel_event: asyncio.Event,
        run_generation: int,
    ) -> LlmResponse:
        emitted = False
        observer_failed = False

        def observe(delta: str) -> Any:
            nonlocal emitted, observer_failed
            if cancel_event.is_set() or run_generation != self._run_generation:
                return None
            emitted = True
            if self.on_content_delta is not None:
                try:
                    outcome = self.on_content_delta(delta)
                except (asyncio.CancelledError, Exception):
                    observer_failed = True
                    return None
                if inspect.isawaitable(outcome):

                    async def await_observer() -> None:
                        nonlocal observer_failed
                        if not await _await_tool_observer(outcome):
                            observer_failed = True

                    return await_observer()
            return None

        def observe_reasoning(delta: str) -> Any:
            nonlocal emitted, observer_failed
            if cancel_event.is_set() or run_generation != self._run_generation:
                return None
            emitted = True
            observer = self.on_reasoning_delta or self.on_reasoning
            if observer is not None:
                try:
                    outcome = observer(delta)
                except (asyncio.CancelledError, Exception):
                    observer_failed = True
                    return None
                if inspect.isawaitable(outcome):

                    async def await_observer() -> None:
                        nonlocal observer_failed
                        if not await _await_tool_observer(outcome):
                            observer_failed = True

                    return await_observer()
            return None

        previous_reasoning_observer = self.llm.on_reasoning_delta
        self.llm.on_reasoning_delta = observe_reasoning
        try:
            response = await self._retry_llm_request(
                lambda: self.llm.complete_streaming(messages, tools, observe),
                cancel_event,
                has_partial_output=lambda: emitted,
            )
        finally:
            self.llm.on_reasoning_delta = previous_reasoning_observer
        if self.on_content_delta is None or observer_failed:
            response.streamed = False
        if self.on_reasoning_delta is None and self.on_reasoning is None or observer_failed:
            response.reasoning_streamed = False
        return response

    async def _retry_llm_request(
        self,
        request: Callable[[], Awaitable[LlmResponse]],
        cancel_event: asyncio.Event,
        *,
        has_partial_output: Callable[[], bool] = lambda: False,
    ) -> LlmResponse:
        for attempt in range(self.max_llm_retries + 1):
            try:
                return await wait_with_cancellation(request(), cancel_event)
            except LlmError as exc:
                if not exc.retryable or has_partial_output() or attempt >= self.max_llm_retries:
                    raise
                delay = (
                    exc.retry_after
                    if exc.retry_after is not None
                    else self.llm_retry_base_seconds * (2**attempt)
                )
                await wait_with_cancellation(
                    asyncio.sleep(min(max(delay, 0.0), 10.0)), cancel_event
                )
        raise AssertionError("unreachable retry loop")

    def cancel(self) -> None:
        self.cancel_event.set()

    def add_system_context(self, value: str) -> None:
        if self._run_lock.locked():
            raise RuntimeError("Agent already has an active run")
        if not value.strip():
            return
        self.base_system_prompt += "\n\n" + value.strip()
        self.system_prompt = self.base_system_prompt

    def _refresh_memory_context(self, query: str) -> None:
        self.system_prompt = self.base_system_prompt
        if self.memory_store is None:
            return
        recent_texts = [
            message.content
            for message in self.history
            if isinstance(message.content, str) and message.content.strip()
        ][-10:]
        fact = browser_login_fact(query, recent_texts)
        if fact:
            self.memory_store.save(fact, "global")
        context = self.memory_store.context_for_query(
            query, self.context_profile.memory_context_tokens
        )
        if context:
            self.system_prompt += (
                "\n\n<relevant_long_term_memory>\n" + context + "\n</relevant_long_term_memory>"
            )

    def clear(self) -> None:
        if self._run_lock.locked():
            raise RuntimeError("Agent already has an active run")
        self.history.clear()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_tokens = 0
        self.llm_call_count = 0
        self.compaction_count = 0
        self.last_context_tokens = 0
        self.reset_run_budget()

    def _strip_historical_images(self) -> None:
        for message in self.history:
            if not isinstance(message.content, list):
                continue
            retained = [part for part in message.content if part.get("type") != "image_url"]
            omitted = len(message.content) - len(retained)
            if omitted:
                retained.append(
                    {
                        "type": "text",
                        "text": (
                            f"[Omitted {omitted} historical image attachment(s). Re-run the "
                            "related tool or attach the image again if it is still needed.]"
                        ),
                    }
                )
                message.content = retained

    async def compact(self) -> bool:
        if self._run_lock.locked():
            raise RuntimeError("Agent already has an active run")
        async with self._run_lock:
            self.cancel_event.clear()
            compacted = await self.compactor.compact_now(self.history, self.cancel_event)
            if compacted:
                self.compaction_count += 1
                self.last_context_tokens = self.estimate_current_context_tokens()
            return compacted

    async def _compact_if_needed(self) -> None:
        current = self.estimate_current_context_tokens()
        self.last_context_tokens = current
        compacted = await self.compactor.compact_if_needed(
            self.history,
            current,
            self.context_profile.compression_trigger_tokens,
            self.cancel_event,
        )
        if compacted:
            self.compaction_count += 1
            self.last_context_tokens = self.estimate_current_context_tokens()

    def estimate_current_context_tokens(self) -> int:
        return estimate_message_tokens(
            [Message("system", self.system_prompt), *self.history]
        ) + estimate_schema_tokens(self.tools.schemas())

    def record_usage(self, response: Any) -> None:
        self.total_input_tokens += max(0, response.usage.input_tokens)
        self.total_output_tokens += max(0, response.usage.output_tokens)
        self.total_cached_tokens += max(0, response.usage.cache_tokens)
        self.llm_call_count += 1

    def absorb_usage(self, child: Agent) -> None:
        self.total_input_tokens += child.total_input_tokens
        self.total_output_tokens += child.total_output_tokens
        self.total_cached_tokens += child.total_cached_tokens
        self.llm_call_count += child.llm_call_count
        self.compaction_count += child.compaction_count

    def context_status(self) -> str:
        current = self.estimate_current_context_tokens()
        profile = self.context_profile
        ratio = current / profile.max_context_window
        remaining = max(0, profile.compression_trigger_tokens - current)
        cost = estimated_cost_cny(
            self.llm.provider,
            self.total_input_tokens,
            self.total_output_tokens,
            self.total_cached_tokens,
            model=self.llm.model,
            pricing=self.pricing,
        )
        hard_budget = self.budget.token_budget or "unlimited"
        memory_status = (
            self.memory_store.status_summary()
            if self.memory_store is not None
            else "Long-term memory: unavailable"
        )
        pricing_status = f"\n{self.pricing.warning}" if self.pricing.warning else ""
        return (
            f"Model: {self.llm.model} ({self.llm.provider})\n"
            f"Context: {current} / {profile.max_context_window} tokens ({ratio:.1%})\n"
            f"Auto-compact: {profile.compression_trigger_tokens} tokens; "
            f"remaining {remaining}; runs {self.compaction_count}\n"
            f"Conversation budget: {profile.short_term_memory_budget}; "
            f"memory injection: {profile.memory_context_tokens}\n"
            f"MCP resource index: {'on' if profile.mcp_resource_index_enabled else 'off'}; "
            f"prompt cache: {profile.prompt_cache_mode}\n"
            f"Usage: calls {self.llm_call_count}; input {self.total_input_tokens}; "
            f"output {self.total_output_tokens}; cached {self.total_cached_tokens}; "
            f"estimated cost ¥{cost:.4f}\n"
            f"Safety limits: hard iterations {self.budget.max_iterations}; "
            f"token budget {hard_budget}; model retries {self.max_llm_retries}\n"
            f"{memory_status}{pricing_status}"
        )

    def status_line(self) -> str:
        current = self.estimate_current_context_tokens()
        cost = estimated_cost_cny(
            self.llm.provider,
            self.total_input_tokens,
            self.total_output_tokens,
            self.total_cached_tokens,
            model=self.llm.model,
            pricing=self.pricing,
        )
        window = _format_tokens(self.context_profile.max_context_window)
        return (
            f"{self.llm.provider}/{self.llm.model} · "
            f"ctx {_format_tokens(current)}/{window} "
            f"· in/out/cache {self.total_input_tokens}/{self.total_output_tokens}/"
            f"{self.total_cached_tokens} · ¥{cost:.4f}"
        )

    def export_markdown(self) -> str:
        exported_at = datetime.now().astimezone().isoformat(timespec="seconds")
        sections = [
            "# Kairo CLI Session Export",
            "",
            f"**Exported at**: {exported_at}",
            "",
            "---",
            "",
            "## System",
            "",
            self.system_prompt,
        ]
        for message in self.history:
            content = (
                message.content
                if isinstance(message.content, str)
                else json.dumps(message.content, ensure_ascii=False, indent=2)
            )
            role = "Tool Result" if message.role == "tool" else message.role.title()
            sections.extend(["", f"## {role}", ""])
            if message.reasoning_content and message.reasoning_content.strip():
                sections.extend(["> **Reasoning**:", ">"])
                sections.extend(
                    f"> {line}"
                    for line in _normalize_markdown_text(message.reasoning_content).split("\n")
                )
                sections.append("")
            if message.tool_calls:
                rendered_calls = json.dumps(
                    [asdict(call) for call in message.tool_calls],
                    ensure_ascii=False,
                    indent=2,
                )
                fence = markdown_fence_for(rendered_calls)
                sections.extend(["**Tool calls**:", "", fence + "json", rendered_calls, fence, ""])
            if content:
                normalized = _normalize_markdown_text(content)
                if message.role == "tool":
                    original_length = len(normalized)
                    if original_length > 8_000:
                        normalized = (
                            normalized[:8_000]
                            + f"\n... (truncated; original length {original_length} characters)"
                        )
                    fence = markdown_fence_for(normalized)
                    sections.extend([fence, normalized, fence])
                elif isinstance(message.content, str):
                    sections.append(normalized)
                else:
                    fence = markdown_fence_for(normalized)
                    sections.extend([fence + "json", normalized, fence])
        return "\n".join(sections) + "\n"


def markdown_fence_for(content: str) -> str:
    longest = max((len(match.group(0)) for match in re.finditer(r"`+", content)), default=0)
    return "`" * max(3, longest + 1)


def _normalize_markdown_text(content: str) -> str:
    return content.replace("\r\n", "\n").replace("\r", "\n")


def _tool_call_snapshot(calls: list[ToolCall]) -> list[ToolCall]:
    return [ToolCall(call.id, call.name, copy.deepcopy(call.arguments)) for call in calls]


def _canceled_tool_result(call: ToolCall) -> ToolOutput:
    side_effects_may_have_completed = (
        call.name in SERIALIZED_WORKSPACE_MUTATION_TOOLS or call.name.startswith("mcp__")
    )
    return ToolOutput(
        json.dumps(
            {
                "error": "Tool batch canceled before a reliable result was available",
                "canceled": True,
                "side_effects_may_have_completed": side_effects_may_have_completed,
            },
            separators=(",", ":"),
        )
    )


async def _await_tool_observer(outcome: Awaitable[Any]) -> bool:
    future = asyncio.ensure_future(outcome)
    try:
        done, _ = await asyncio.wait({future}, timeout=TOOL_OBSERVER_TIMEOUT_SECONDS)
    except BaseException:
        future.cancel()
        _detach_tool_observer(future)
        raise
    if done:
        result = await asyncio.gather(future, return_exceptions=True)
        return not isinstance(result[0], BaseException)
    future.cancel()
    done, _ = await asyncio.wait({future}, timeout=TOOL_OBSERVER_CANCEL_GRACE_SECONDS)
    if done:
        await asyncio.gather(future, return_exceptions=True)
    else:
        _detach_tool_observer(future)
    return False


def _detach_tool_observer(future: asyncio.Future[Any]) -> None:
    if future.done():
        try:
            future.exception()
        except (asyncio.CancelledError, Exception):
            pass
        return
    _DETACHED_TOOL_OBSERVERS.add(future)
    future.add_done_callback(_finish_detached_tool_observer)


def _finish_detached_tool_observer(future: asyncio.Future[Any]) -> None:
    _DETACHED_TOOL_OBSERVERS.discard(future)
    try:
        future.exception()
    except (asyncio.CancelledError, Exception):
        pass


def _positive_int(value: str | None, default: int | None = None) -> int | None:
    try:
        parsed = int(value) if value is not None else 0
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _nonnegative_int(value: str | None, default: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


def _bounded_float(value: str | None, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value) if value is not None else default
    except ValueError:
        return default
    if not minimum <= parsed <= maximum:
        return default
    return parsed


def _format_tokens(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}m"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)
