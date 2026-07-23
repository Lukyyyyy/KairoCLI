from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]
Content = str | list[dict[str, Any]]


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class FileDiff:
    path: str
    before: str | None
    after: str | None
    omitted_reason: str = ""


@dataclass(frozen=True, slots=True)
class ToolOutput:
    text: str = ""
    image_urls: tuple[str, ...] = ()
    truncated: bool = False
    original_chars: int = 0
    timed_out: bool = False
    elapsed_ms: int = 0
    diffs: tuple[FileDiff, ...] = ()

    @property
    def has_images(self) -> bool:
        return bool(self.image_urls)


@dataclass(slots=True)
class Message:
    role: Role
    content: Content = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    reasoning_content: str | None = None


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_tokens: int = 0


@dataclass(slots=True)
class LlmResponse:
    content: str = ""
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    streamed: bool = False
    reasoning_streamed: bool = False
