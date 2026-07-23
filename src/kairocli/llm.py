from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterable
from typing import Any

from .config import (
    AppConfig,
    ProviderConfig,
    normalize_provider_base_url,
    normalize_provider_name,
    validate_provider_protocol_fields,
)
from .models import LlmResponse, Message, ToolCall, Usage
from .trace import redact_sensitive_text, safe_redacted_text

MAX_LLM_TOOL_CALLS = 100
MAX_LLM_TOOL_ARGUMENT_BYTES = 1024 * 1024
MAX_LLM_RESPONSE_BYTES = 20 * 1024 * 1024
MAX_LLM_SSE_EVENT_BYTES = 2 * 1024 * 1024
MAX_LLM_ERROR_BYTES = 10_000
MAX_LLM_USAGE_TOKENS = 1_000_000_000_000
MAX_LLM_JSON_DEPTH = 32
MAX_LLM_JSON_NODES = 200_000
LLM_CONNECT_TIMEOUT_SECONDS = 60.0
LLM_READ_TIMEOUT_SECONDS = 300.0
LLM_WRITE_TIMEOUT_SECONDS = 60.0
LLM_POOL_TIMEOUT_SECONDS = 60.0
LLM_CALL_TIMEOUT_SECONDS = 600.0
_TOOL_PROTOCOL_VALUE = re.compile(r"[A-Za-z0-9_.:-]+\Z")


class LlmError(RuntimeError):
    """Raised when an upstream model request cannot be completed."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after


class LlmClient(ABC):
    provider: str
    model: str
    on_reasoning_delta: Callable[[str], Any] | None = None

    @abstractmethod
    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        raise NotImplementedError

    async def stream(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[str]:
        response = await self.complete(messages, tools)
        if response.content:
            yield response.content

    async def complete_streaming(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        on_delta: Callable[[str], Any] | None = None,
    ) -> LlmResponse:
        return await self.complete(messages, tools)

    def supports_image_input(self) -> bool:
        return self.provider != "deepseek"

    def max_context_window(self) -> int:
        return {
            "glm": 200_000,
            "deepseek": 1_000_000,
            "step": 256_000,
            "kimi": 256_000,
            "agnes": 1_000_000,
        }.get(self.provider, 128_000)

    def supports_prompt_caching(self) -> bool:
        return self.provider in {"glm", "deepseek", "step", "kimi"}

    def prompt_cache_mode(self) -> str:
        return {
            "glm": "glm-prompt-cache",
            "deepseek": "automatic-prefix-cache",
            "step": "step-prefix-cache",
            "kimi": "moonshot-context-cache",
        }.get(self.provider, "none")

    def supports_tools(self) -> bool:
        return self.provider != "xfyun"


class OpenAiCompatibleClient(LlmClient):
    def __init__(self, provider: str, config: ProviderConfig, transport: Any = None) -> None:
        validate_provider_protocol_fields(config, provider)
        self.provider = provider
        self.model = config.model
        self.config = config
        self.transport = transport

    def max_context_window(self) -> int:
        return self.config.context_window or super().max_context_window()

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - installation guard
            raise LlmError("Install Kairo CLI dependencies to call a model") from exc
        if not self.config.api_key:
            raise LlmError(f"Missing API key for provider: {self.provider}")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                _serialize_message(
                    message,
                    self.supports_image_input(),
                    self.provider == "glm" and self.model.casefold().startswith("glm-5v"),
                    self.provider in {"deepseek", "kimi"},
                )
                for message in messages
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        if tools and self.supports_tools():
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        _customize_payload(payload, self.provider, self.model)
        headers = _request_headers(self.provider, self.config)
        url = _chat_completions_url(self.config.base_url, self.provider, self.model)
        try:
            async with asyncio.timeout(LLM_CALL_TIMEOUT_SECONDS):
                async with httpx.AsyncClient(
                    timeout=_httpx_timeout(httpx), transport=self.transport
                ) as client:
                    async with client.stream(
                        "POST", url, headers=headers, json=payload
                    ) as response:
                        await _ensure_success(response)
                        body = await _read_bounded_response(
                            response,
                            MAX_LLM_RESPONSE_BYTES,
                            "Model response exceeds the 20 MiB limit",
                        )
            try:
                parsed = _decode_llm_json(body)
            except (OverflowError, RecursionError, UnicodeError, ValueError) as exc:
                raise LlmError("Model returned invalid JSON") from exc
            if not isinstance(parsed, dict):
                raise LlmError("Model response JSON must be an object")
            return _parse_response(parsed)
        except LlmError:
            raise
        except Exception as exc:
            raise LlmError(
                f"{self.provider} request failed: "
                + safe_redacted_text(exc, MAX_LLM_ERROR_BYTES, "...[model error truncated]"),
                retryable=isinstance(exc, (httpx.TransportError, TimeoutError)),
            ) from exc

    async def complete_streaming(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        on_delta: Callable[[str], Any] | None = None,
    ) -> LlmResponse:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise LlmError("Install Kairo CLI dependencies to call a model") from exc
        if not self.config.api_key:
            raise LlmError(f"Missing API key for provider: {self.provider}")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                _serialize_message(
                    message,
                    self.supports_image_input(),
                    self.provider == "glm" and self.model.casefold().startswith("glm-5v"),
                    self.provider in {"deepseek", "kimi"},
                )
                for message in messages
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools and self.supports_tools():
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        _customize_payload(payload, self.provider, self.model)
        headers = _request_headers(self.provider, self.config)
        url = _chat_completions_url(self.config.base_url, self.provider, self.model)
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        usage = Usage()
        reasoning_streamed = False
        try:
            async with asyncio.timeout(LLM_CALL_TIMEOUT_SECONDS):
                async with httpx.AsyncClient(
                    timeout=_httpx_timeout(httpx), transport=self.transport
                ) as client:
                    request = client.build_request("POST", url, headers=headers, json=payload)
                    response = await client.send(request, stream=True)
                    await _ensure_success(response)
                    async for data in _iter_llm_sse_data(response):
                        if not data or data == "[DONE]":
                            continue
                        try:
                            event = _decode_llm_json(data)
                        except (
                            OverflowError,
                            RecursionError,
                            UnicodeError,
                            ValueError,
                        ) as exc:
                            raise LlmError("Model SSE event contains invalid JSON") from exc
                        if not isinstance(event, dict):
                            raise LlmError("Model SSE event must be a JSON object")
                        if event.get("error") is not None:
                            raise LlmError(
                                "API request failed: "
                                + redact_sensitive_text(_format_streaming_error(event["error"])),
                                retryable=_retryable_error_payload(event["error"]),
                            )
                        raw_usage = event.get("usage")
                        if raw_usage is not None:
                            usage = _parse_usage(raw_usage)
                        choices = event.get("choices") or []
                        if not choices:
                            continue
                        if not isinstance(choices, list) or not isinstance(choices[0], dict):
                            raise LlmError("Model choices must be an array of objects")
                        delta = choices[0].get("delta") or choices[0].get("message") or {}
                        if not isinstance(delta, dict):
                            raise LlmError("Model delta must be an object")
                        content = delta.get("content") or ""
                        if not isinstance(content, str):
                            raise LlmError("Model content delta must be text")
                        if content:
                            content_parts.append(content)
                            if on_delta:
                                callback_result = on_delta(content)
                                if inspect.isawaitable(callback_result):
                                    await callback_result
                        reasoning = _extract_reasoning(delta)
                        if reasoning:
                            reasoning_parts.append(reasoning)
                            if self.on_reasoning_delta is not None:
                                callback_result = self.on_reasoning_delta(reasoning)
                                reasoning_streamed = True
                                if inspect.isawaitable(callback_result):
                                    await callback_result
                        raw_calls = delta.get("tool_calls") or []
                        if not isinstance(raw_calls, list):
                            raise LlmError("Model tool_calls delta must be an array")
                        for raw_call in raw_calls:
                            if not isinstance(raw_call, dict):
                                raise LlmError("Each model tool-call delta must be an object")
                            index = raw_call.get("index", 0)
                            if type(index) is not int or not 0 <= index < MAX_LLM_TOOL_CALLS:
                                raise LlmError("Model tool-call index must be a bounded integer")
                            if index not in calls and len(calls) >= MAX_LLM_TOOL_CALLS:
                                raise LlmError(
                                    f"Model returned more than {MAX_LLM_TOOL_CALLS} tool calls"
                                )
                            target = calls.setdefault(
                                index, {"id": "", "name": "", "arguments": ""}
                            )
                            raw_id = raw_call.get("id")
                            if raw_id is not None:
                                if not isinstance(raw_id, str):
                                    raise LlmError("Model tool-call ID delta must be text")
                                target["id"] = raw_id
                            function = raw_call.get("function") or {}
                            if not isinstance(function, dict):
                                raise LlmError("Model tool-call function delta must be an object")
                            name_chunk = function.get("name")
                            argument_chunk = function.get("arguments")
                            if name_chunk is not None and not isinstance(name_chunk, str):
                                raise LlmError("Model tool-call name delta must be text")
                            if argument_chunk is not None and not isinstance(argument_chunk, str):
                                raise LlmError("Model tool-call arguments delta must be text")
                            target["name"] += name_chunk or ""
                            target["arguments"] += argument_chunk or ""
                            if (
                                len(target["arguments"].encode("utf-8"))
                                > MAX_LLM_TOOL_ARGUMENT_BYTES
                            ):
                                raise LlmError("Model tool arguments exceed the 1 MiB limit")
        except LlmError:
            raise
        except Exception as exc:
            raise LlmError(
                f"{self.provider} streaming request failed: "
                + safe_redacted_text(exc, MAX_LLM_ERROR_BYTES, "...[model error truncated]"),
                retryable=isinstance(exc, (httpx.TransportError, TimeoutError)),
            ) from exc
        tool_calls: list[ToolCall] = []
        seen_ids: set[str] = set()
        for position, (_, raw_call) in enumerate(sorted(calls.items())):
            tool_calls.append(
                _normalize_tool_call(
                    raw_call["id"],
                    raw_call["name"],
                    raw_call["arguments"],
                    position,
                    seen_ids,
                )
            )
        if not content_parts and not reasoning_parts and not tool_calls:
            raise LlmError(
                "Model returned no content; verify the provider/model and request capabilities"
            )
        return LlmResponse(
            content="".join(content_parts),
            reasoning_content="".join(reasoning_parts) or None,
            tool_calls=tool_calls,
            usage=usage,
            streamed=bool(content_parts and on_delta),
            reasoning_streamed=reasoning_streamed,
        )


def create_llm_client(config: AppConfig, provider: str | None = None) -> LlmClient:
    selected = normalize_provider_name(provider or config.default_provider)
    if selected not in config.providers:
        raise ValueError(f"Unsupported provider: {selected}")
    return OpenAiCompatibleClient(selected, config.providers[selected])


async def _iter_llm_sse_data(response: Any) -> AsyncIterator[str]:
    total_bytes = 0
    buffer = bytearray()
    data_parts: list[str] = []
    event_bytes = 0

    def consume(raw_line: bytes) -> str | None:
        nonlocal data_parts, event_bytes
        if raw_line.endswith(b"\r"):
            raw_line = raw_line[:-1]
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise LlmError("Model SSE stream is not valid UTF-8") from exc
        if not line:
            if not data_parts:
                return None
            payload = "\n".join(data_parts)
            data_parts = []
            event_bytes = 0
            return payload
        if line == "data":
            part = ""
        elif line.startswith("data:"):
            part = line[5:]
            if part.startswith(" "):
                part = part[1:]
        else:
            return None
        event_bytes += len(part.encode("utf-8")) + (1 if data_parts else 0)
        if event_bytes > MAX_LLM_SSE_EVENT_BYTES:
            raise LlmError("Model SSE event exceeds the 2 MiB limit")
        data_parts.append(part)
        return None

    async for chunk in response.aiter_bytes():
        total_bytes += len(chunk)
        if total_bytes > MAX_LLM_RESPONSE_BYTES:
            raise LlmError("Model response exceeds the 20 MiB limit")
        buffer.extend(chunk)
        while True:
            newline = buffer.find(b"\n")
            if newline < 0:
                break
            raw_line = bytes(buffer[:newline])
            del buffer[: newline + 1]
            payload = consume(raw_line)
            if payload is not None:
                yield payload
        if len(buffer) > MAX_LLM_SSE_EVENT_BYTES + 6:
            raise LlmError("Model SSE event exceeds the 2 MiB limit")
    if buffer:
        payload = consume(bytes(buffer))
        if payload is not None:
            yield payload
    if data_parts:
        yield "\n".join(data_parts)


def _decode_llm_json(value: str | bytes) -> Any:
    payload = json.loads(
        value,
        object_pairs_hook=_llm_object_without_duplicates,
        parse_constant=_reject_llm_json_constant,
    )
    _validate_llm_json_shape(payload)
    return payload


def _llm_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate model JSON key: {key}")
        result[key] = value
    return result


def _reject_llm_json_constant(value: str) -> Any:
    raise ValueError(f"Non-standard JSON constant is not allowed: {value}")


def _validate_llm_json_shape(value: Any) -> None:
    stack: list[tuple[Any, int]] = [(value, 0)]
    visited = 0
    while stack:
        current, depth = stack.pop()
        visited += 1
        if visited > MAX_LLM_JSON_NODES:
            raise ValueError("Model JSON exceeds the node limit")
        if depth > MAX_LLM_JSON_DEPTH:
            raise ValueError("Model JSON exceeds the nesting limit")
        children: Iterable[Any]
        if isinstance(current, dict):
            children = current.values()
        elif isinstance(current, list):
            children = current
        else:
            continue
        child_count = len(current)
        if visited + len(stack) + child_count > MAX_LLM_JSON_NODES:
            raise ValueError("Model JSON exceeds the node limit")
        stack.extend((child, depth + 1) for child in children)


def _serialize_message(
    message: Message,
    supports_images: bool = True,
    raw_image_base64: bool = False,
    include_reasoning: bool = False,
) -> dict[str, Any]:
    content = message.content
    if isinstance(content, list) and not supports_images:
        omitted = sum(1 for part in content if part.get("type") == "image_url")
        content = [part for part in content if part.get("type") != "image_url"]
        if omitted:
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"[The current provider does not support image input; omitted {omitted} "
                        "image attachment(s). Continue from the textual tool result.]"
                    ),
                }
            )
    elif isinstance(content, list) and raw_image_base64:
        normalized_parts: list[dict[str, Any]] = []
        for part in content:
            copied = dict(part)
            if copied.get("type") == "image_url":
                image_url = copied.get("image_url")
                if isinstance(image_url, dict):
                    url = str(image_url.get("url", ""))
                    if url.startswith("data:") and ";base64," in url:
                        copied["image_url"] = {**image_url, "url": url.split(",", 1)[1]}
            normalized_parts.append(copied)
        content = normalized_parts
    payload: dict[str, Any] = {"role": message.role, "content": content}
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    if include_reasoning and message.role == "assistant" and message.reasoning_content:
        payload["reasoning_content"] = message.reasoning_content
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in message.tool_calls
        ]
    return payload


def _customize_payload(payload: dict[str, Any], provider: str, model: str) -> None:
    if provider == "step":
        payload["reasoning_format"] = "deepseek-style"
        if "2603" in model:
            payload["reasoning_effort"] = "high"


def _httpx_timeout(httpx: Any) -> Any:
    return httpx.Timeout(
        connect=LLM_CONNECT_TIMEOUT_SECONDS,
        read=LLM_READ_TIMEOUT_SECONDS,
        write=LLM_WRITE_TIMEOUT_SECONDS,
        pool=LLM_POOL_TIMEOUT_SECONDS,
    )


def _request_headers(provider: str, config: ProviderConfig) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {config.api_key}"}
    # lora_id is an iFlytek MaaS resource-card selector, not a portable
    # OpenAI-compatible header. Never forward it to another provider.
    if provider == "xfyun" and config.lora_id.strip():
        headers["lora_id"] = config.lora_id.strip()
    return headers


async def _ensure_success(response: Any) -> None:
    if int(response.status_code) < 400:
        return
    body = await _read_bounded_response(
        response,
        MAX_LLM_ERROR_BYTES + 1,
        "",
        reject_excess=False,
    )
    truncated = len(body) > MAX_LLM_ERROR_BYTES
    detail = body[:MAX_LLM_ERROR_BYTES].decode("utf-8", errors="replace").strip()
    detail = detail or "empty response body"
    if truncated:
        detail += "..."
    status = int(response.status_code)
    raise LlmError(
        f"API request failed: HTTP {response.status_code} - {redact_sensitive_text(detail)}",
        retryable=status in {408, 425, 429, 500, 502, 503, 504},
        retry_after=_retry_after_seconds(response.headers.get("retry-after")),
    )


async def _read_bounded_response(
    response: Any,
    limit: int,
    error: str,
    *,
    reject_excess: bool = True,
) -> bytes:
    body = bytearray()
    async for chunk in response.aiter_bytes():
        remaining = limit - len(body)
        if len(chunk) > remaining:
            if reject_excess:
                raise LlmError(error)
            body.extend(chunk[:remaining])
            break
        body.extend(chunk)
    return bytes(body)


def _chat_completions_url(base_url: str, provider: str, model: str) -> str:
    try:
        normalized = normalize_provider_base_url(base_url, provider)
    except ValueError as exc:
        raise LlmError(
            safe_redacted_text(exc, MAX_LLM_ERROR_BYTES, "...[model error truncated]")
        ) from exc
    if provider == "glm" and model.casefold().startswith("glm-5v"):
        normalized = normalized.replace("/api/coding/paas/v4", "/api/paas/v4")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def _extract_reasoning(delta: dict[str, Any]) -> str:
    for key in ("reasoning_content", "reasoning"):
        value = delta.get(key)
        if isinstance(value, str) and value:
            return value
    details = delta.get("reasoning_details")
    if not isinstance(details, list):
        return ""
    parts: list[str] = []
    for detail in details:
        if not isinstance(detail, dict):
            continue
        value = detail.get("text") or detail.get("content")
        if isinstance(value, str):
            parts.append(value)
    return "".join(parts)


def _format_streaming_error(error: Any) -> str:
    if isinstance(error, dict):
        code = str(error.get("code", "")).strip()
        message = str(error.get("message", "")).strip()
        if code and message:
            return f"{code} - {message}"
        if message:
            return message
    return json.dumps(error, ensure_ascii=False, default=str)


def _retry_after_seconds(value: Any) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(str(value).strip())
    except ValueError:
        return None
    if not 0 <= seconds <= 60:
        return None
    return seconds


def _retryable_error_payload(error: Any) -> bool:
    if not isinstance(error, dict):
        return False
    values = {str(error.get(key, "")).strip().casefold() for key in ("code", "type", "status")}
    if values & {"408", "425", "429", "500", "502", "503", "504"}:
        return True
    return any(
        marker in value
        for value in values
        for marker in ("rate_limit", "server_error", "temporarily_unavailable", "timeout")
    )


def _parse_response(payload: dict[str, Any]) -> LlmResponse:
    if payload.get("error") is not None:
        raise LlmError(
            "API request failed: "
            + redact_sensitive_text(_format_streaming_error(payload["error"])),
            retryable=_retryable_error_payload(payload["error"]),
        )
    try:
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LlmError("Malformed model response") from exc
    if not isinstance(message, dict):
        raise LlmError("Model message must be an object")
    calls: list[ToolCall] = []
    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        raise LlmError("Model tool_calls must be an array")
    if len(raw_calls) > MAX_LLM_TOOL_CALLS:
        raise LlmError(f"Model returned more than {MAX_LLM_TOOL_CALLS} tool calls")
    seen_ids: set[str] = set()
    for position, raw_call in enumerate(raw_calls):
        if not isinstance(raw_call, dict):
            raise LlmError("Each model tool call must be an object")
        function = raw_call.get("function", {})
        if not isinstance(function, dict):
            raise LlmError("Model tool-call function must be an object")
        raw_id = raw_call.get("id", "")
        raw_name = function.get("name", "")
        if not isinstance(raw_id, str):
            raise LlmError("Model tool-call ID must be text")
        if not isinstance(raw_name, str):
            raise LlmError("Model tool-call name must be text")
        arguments = function.get("arguments", {})
        calls.append(
            _normalize_tool_call(
                raw_id,
                raw_name,
                arguments,
                position,
                seen_ids,
            )
        )
    content = message.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise LlmError("Model message content must be text")
    reasoning = _extract_reasoning(message)
    if not content and not reasoning and not calls:
        raise LlmError(
            "Model returned no content; verify the provider/model and request capabilities"
        )
    raw_usage = payload.get("usage")
    usage = Usage() if raw_usage is None else _parse_usage(raw_usage)
    return LlmResponse(
        content=content,
        reasoning_content=reasoning or None,
        tool_calls=calls,
        usage=usage,
    )


def _normalize_tool_call(
    identifier: str,
    name: str,
    arguments: Any,
    position: int,
    seen_ids: set[str],
) -> ToolCall:
    normalized_name = name.strip()
    if (
        not normalized_name
        or len(normalized_name) > 256
        or not _TOOL_PROTOCOL_VALUE.fullmatch(normalized_name)
    ):
        raise LlmError("Model returned an invalid tool name")
    if isinstance(arguments, str):
        try:
            arguments = _decode_llm_json(arguments or "{}")
        except (OverflowError, RecursionError, UnicodeError, ValueError) as exc:
            raise LlmError(
                f"Model returned malformed JSON arguments for {normalized_name}"
            ) from exc
    if not isinstance(arguments, dict):
        raise LlmError(f"Model tool arguments for {normalized_name} must be an object")
    try:
        serialized = json.dumps(
            arguments,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise LlmError(
            f"Model tool arguments for {normalized_name} are not JSON serializable"
        ) from exc
    if len(serialized.encode("utf-8")) > MAX_LLM_TOOL_ARGUMENT_BYTES:
        raise LlmError(f"Model tool arguments for {normalized_name} exceed the 1 MiB limit")
    normalized_id = identifier.strip()
    if (
        not normalized_id
        or len(normalized_id) > 200
        or not _TOOL_PROTOCOL_VALUE.fullmatch(normalized_id)
        or normalized_id in seen_ids
    ):
        digest = hashlib.sha256(f"{position}:{normalized_name}:{serialized}".encode()).hexdigest()[
            :12
        ]
        normalized_id = f"call_kairo_{position}_{digest}"
        suffix = 1
        while normalized_id in seen_ids:  # pragma: no cover - digest collision defense
            normalized_id = f"call_kairo_{position}_{digest}_{suffix}"
            suffix += 1
    seen_ids.add(normalized_id)
    return ToolCall(normalized_id, normalized_name, arguments)


def _parse_usage(value: Any) -> Usage:
    if not isinstance(value, dict):
        raise LlmError("Model usage must be an object")
    return Usage(
        _usage_token(value.get("prompt_tokens", 0), "prompt_tokens"),
        _usage_token(value.get("completion_tokens", 0), "completion_tokens"),
        _parse_cached_tokens(value),
    )


def _usage_token(value: Any, field: str) -> int:
    if type(value) is not int or not 0 <= value <= MAX_LLM_USAGE_TOKENS:
        raise LlmError(f"Model usage {field} must be a bounded non-negative integer")
    return value


def _parse_cached_tokens(usage: dict[str, Any]) -> int:
    cached = 0
    for key in ("cached_tokens", "prompt_cache_hit_tokens", "input_cache_hit_tokens"):
        if key in usage:
            cached = _usage_token(usage[key], key)
            break
    for key in ("prompt_tokens_details", "input_tokens_details"):
        details = usage.get(key)
        if details is None:
            continue
        if not isinstance(details, dict):
            raise LlmError(f"Model usage {key} must be an object")
        cached = _usage_token(details.get("cached_tokens", cached), "cached_tokens")
    return cached
