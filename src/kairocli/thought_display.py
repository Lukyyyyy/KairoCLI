from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from .terminal import sanitize_terminal_text
from .text_safety import safe_text
from .trace import redact_sensitive_text

MAX_THOUGHT_DETAIL_CHARS = 100_000
MAX_THOUGHT_TURNS = 50


@dataclass(frozen=True)
class ThoughtTurn:
    prompt: str
    answer: str
    details: str
    elapsed_seconds: int
    answer_streamed: bool = False


class ThoughtDisplay:
    """Bounded, terminal-safe details for the most recent interactive turn."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._started_at = clock()
        self._detail_parts: list[str] = []
        self._detail_chars = 0
        self._accepting_detail_delta = False
        self._prompt = ""
        self._thinking_elapsed_seconds: int | None = None
        self.turns: deque[ThoughtTurn] = deque(maxlen=MAX_THOUGHT_TURNS)
        self.elapsed_seconds = 1
        self.expanded = False
        self.finished = False

    def start(self, prompt: object = "") -> None:
        self._started_at = self._clock()
        self._detail_parts.clear()
        self._detail_chars = 0
        self._accepting_detail_delta = False
        self._prompt = self._safe(prompt)
        self._thinking_elapsed_seconds = None
        self.elapsed_seconds = 1
        self.expanded = False
        self.finished = False

    def add(self, value: object) -> None:
        if value is None:
            return
        safe = self._safe(value)
        if not safe or self._detail_chars >= MAX_THOUGHT_DETAIL_CHARS:
            return
        remaining = MAX_THOUGHT_DETAIL_CHARS - self._detail_chars
        bounded = safe[:remaining]
        self._detail_parts.append(bounded)
        self._detail_chars += len(bounded)
        self._accepting_detail_delta = False

    def add_delta(self, value: object) -> None:
        if value is None:
            return
        safe = self._safe_fragment(value)
        if not safe or self._detail_chars >= MAX_THOUGHT_DETAIL_CHARS:
            return
        remaining = MAX_THOUGHT_DETAIL_CHARS - self._detail_chars
        bounded = safe[:remaining]
        if self._accepting_detail_delta and self._detail_parts:
            self._detail_parts[-1] += bounded
        else:
            self._detail_parts.append(bounded)
        self._detail_chars += len(bounded)
        self._accepting_detail_delta = True

    def finish(self, answer: object = "", *, answer_streamed: bool = False) -> None:
        self.finish_thinking()
        self.turns.append(
            ThoughtTurn(
                prompt=self._prompt,
                answer=self._safe(answer),
                details=self.details,
                elapsed_seconds=self.elapsed_seconds,
                answer_streamed=answer_streamed,
            )
        )
        self.finished = True

    def finish_thinking(self) -> None:
        if self._thinking_elapsed_seconds is not None:
            return
        elapsed = max(0.0, self._clock() - self._started_at)
        self._thinking_elapsed_seconds = max(1, int(elapsed))
        self.elapsed_seconds = self._thinking_elapsed_seconds

    def toggle(self) -> bool:
        if not self.finished:
            return False
        self.expanded = not self.expanded
        return True

    def dismiss(self) -> None:
        """Hide the completed turn after prompt_toolkit commits it to the terminal."""
        self.expanded = False
        self.finished = False

    @property
    def details(self) -> str:
        return redact_sensitive_text("\n".join(self._detail_parts)).strip()

    def summary(self, *, expanded: bool | None = None) -> str:
        is_expanded = self.expanded if expanded is None else expanded
        action = "collapse" if is_expanded else "expand"
        return f"Thought for {self.elapsed_seconds}s (ctrl+o to {action})"

    def live_summary(self) -> str:
        if self._thinking_elapsed_seconds is None:
            elapsed = max(0.0, self._clock() - self._started_at)
            seconds = max(1, int(elapsed))
        else:
            seconds = self._thinking_elapsed_seconds
        return f"Thought for {seconds}s (ctrl+o to expand)"

    def render(self) -> str:
        if not self.finished:
            return ""
        if not self.expanded:
            return self.summary()
        details = self.details or "No model reasoning or tool activity was returned."
        return f"{self.summary()}\n{details}"

    def render_transcript(self) -> str:
        return "\n\n".join(self._indent(turn.details) for turn in self.turns if turn.details)

    @staticmethod
    def _indent(value: str) -> str:
        return "\n".join("  " + line for line in value.splitlines())

    @staticmethod
    def _safe(value: object) -> str:
        return sanitize_terminal_text(redact_sensitive_text(safe_text(value))).strip()

    @staticmethod
    def _safe_fragment(value: object) -> str:
        return sanitize_terminal_text(safe_text(value))
