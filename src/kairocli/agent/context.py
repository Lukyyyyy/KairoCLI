from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Protocol

from ..models import Message

MAX_TOKEN_ESTIMATION_JSON_CHARS = 1_000_000
INVALID_JSON_ESTIMATE_CHARS = 4_096


class ModelCapabilities(Protocol):
    def max_context_window(self) -> int: ...

    def supports_prompt_caching(self) -> bool: ...

    def prompt_cache_mode(self) -> str: ...


@dataclass(frozen=True, slots=True)
class ContextProfile:
    max_context_window: int
    agent_token_budget: int
    compression_trigger_ratio: float
    short_term_memory_budget: int
    memory_context_tokens: int
    mcp_resource_index_enabled: bool
    prompt_caching_supported: bool
    prompt_cache_mode: str

    @classmethod
    def from_client(cls, client: ModelCapabilities) -> ContextProfile:
        window = max(8_000, client.max_context_window())
        trigger = cls.auto_compact_trigger_tokens(window)
        return cls(
            max_context_window=window,
            agent_token_budget=max(4_000, math.floor(window * 0.8)),
            compression_trigger_ratio=max(0.5, min(0.99, trigger / window)),
            short_term_memory_budget=max(4_000, math.floor(window * 0.45)),
            memory_context_tokens=max(500, min(5_000, window // 200)),
            mcp_resource_index_enabled=window >= 32_000,
            prompt_caching_supported=client.supports_prompt_caching(),
            prompt_cache_mode=client.prompt_cache_mode(),
        )

    @property
    def compression_trigger_tokens(self) -> int:
        return self.auto_compact_trigger_tokens(self.max_context_window)

    @staticmethod
    def auto_compact_trigger_tokens(window: int) -> int:
        safe_window = max(8_000, window)
        summary_reserve = min(20_000, max(1_000, safe_window // 4))
        buffer = min(13_000, max(1_000, safe_window // 8))
        trigger = safe_window - summary_reserve - buffer
        return max(1_000, min(safe_window - 1, trigger))


def estimate_text_tokens(value: str) -> int:
    chinese = sum(1 for char in value if "\u4e00" < char < "\u9fff")
    other = len(value) - chinese
    return math.ceil(chinese / 1.5 + other / 4)


def estimate_message_tokens(messages: list[Message]) -> int:
    total = 0
    for message in messages:
        if isinstance(message.content, str):
            total += estimate_text_tokens(message.content)
        else:
            for part in message.content:
                if not isinstance(part, dict):
                    total += INVALID_JSON_ESTIMATE_CHARS // 4
                    continue
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    total += estimate_text_tokens(part["text"])
                elif part.get("type") == "image_url":
                    image = part.get("image_url")
                    url = image.get("url", "") if isinstance(image, dict) else ""
                    if not isinstance(url, str):
                        total += 1_024
                        continue
                    if ";base64," in url:
                        encoded = url.partition(",")[2]
                        total += max(256, min(4_096, (len(encoded) * 3 // 4) // 768))
                    else:
                        total += 1_024
        for call in message.tool_calls:
            total += estimate_text_tokens(call.name if isinstance(call.name, str) else "invalid")
            total += estimate_text_tokens(_safe_json_for_estimation(call.arguments))
        total += 4
    return total


def estimate_schema_tokens(schemas: list[dict[str, Any]]) -> int:
    return estimate_text_tokens(_safe_json_for_estimation(schemas))


def _safe_json_for_estimation(value: Any) -> str:
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (RecursionError, TypeError, UnicodeError, ValueError):
        return "[invalid JSON]" + "x" * INVALID_JSON_ESTIMATE_CHARS
    if len(serialized) > MAX_TOKEN_ESTIMATION_JSON_CHARS:
        return serialized[:MAX_TOKEN_ESTIMATION_JSON_CHARS]
    return serialized


def estimated_cost_cny(
    provider: str, input_tokens: int, output_tokens: int, cached_tokens: int
) -> float:
    if provider.lower() == "deepseek":
        input_rate, cached_rate, output_rate = 2.0, 0.5, 8.0
    elif provider.lower() == "glm":
        input_rate, cached_rate, output_rate = 5.0, 1.0, 15.0
    else:
        input_rate, cached_rate, output_rate = 5.0, 1.0, 15.0
    cached = max(0, min(input_tokens, cached_tokens))
    uncached = max(0, input_tokens - cached)
    return (
        uncached / 1_000_000 * input_rate
        + cached / 1_000_000 * cached_rate
        + max(0, output_tokens) / 1_000_000 * output_rate
    )
