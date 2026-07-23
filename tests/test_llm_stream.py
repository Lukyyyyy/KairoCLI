import asyncio
import json

import httpx
import pytest

import kairocli.llm as llm_module
from kairocli.config import AppConfig, ProviderConfig
from kairocli.llm import (
    MAX_LLM_TOOL_ARGUMENT_BYTES,
    LlmError,
    OpenAiCompatibleClient,
    create_llm_client,
)
from kairocli.models import Message, ToolCall


def test_direct_client_rejects_header_unsafe_provider_config() -> None:
    config = ProviderConfig(
        api_key="secret\nInjected: value",
        base_url="https://example.test/v1",
        model="model",
    )

    with pytest.raises(ValueError, match="visible ASCII"):
        OpenAiCompatibleClient("glm", config, transport=object())


async def test_streaming_content_reasoning_tools_and_usage() -> None:
    events = [
        {"choices": [{"delta": {"reasoning_content": "think "}}]},
        {"choices": [{"delta": {"content": "hello "}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "function": {"name": "read_file", "arguments": '{"path":'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"a.py"}'}}]}}
            ]
        },
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "prompt_tokens_details": {"cached_tokens": 2},
            },
        },
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"

    async def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    client = OpenAiCompatibleClient(
        "glm",
        ProviderConfig(api_key="test", base_url="https://model.test/v1", model="model"),
        httpx.MockTransport(handler),
    )
    deltas: list[str] = []
    reasoning_deltas: list[str] = []
    client.on_reasoning_delta = reasoning_deltas.append
    response = await client.complete_streaming([Message("user", "hi")], on_delta=deltas.append)
    assert response.content == "hello "
    assert response.reasoning_content == "think "
    assert response.tool_calls[0].arguments == {"path": "a.py"}
    assert response.usage.input_tokens == 10
    assert response.streamed is True
    assert response.reasoning_streamed is True
    assert deltas == ["hello "]
    assert reasoning_deltas == ["think "]


@pytest.mark.parametrize("kind", ["duplicate", "nonfinite", "overdeep"])
async def test_non_streaming_rejects_ambiguous_or_pathological_json(
    kind: str,
) -> None:
    if kind == "duplicate":
        body = b'{"choices":[{"message":{"content":"first","content":"second"}}]}'
    elif kind == "nonfinite":
        body = b'{"choices":[{"message":{"content":"ok"}}],"unknown":NaN}'
    else:
        body = (
            '{"choices":[{"message":{"content":"ok"}}],"unknown":' + "[" * 40 + "0" + "]" * 40 + "}"
        ).encode()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    client = OpenAiCompatibleClient(
        "glm",
        ProviderConfig(api_key="test", base_url="https://model.test/v1", model="model"),
        httpx.MockTransport(handler),
    )

    with pytest.raises(LlmError, match="Model returned invalid JSON"):
        await client.complete([Message("user", "inspect")])


@pytest.mark.parametrize("kind", ["duplicate", "nonfinite", "overdeep", "utf8"])
async def test_streaming_rejects_ambiguous_or_pathological_json(kind: str) -> None:
    if kind == "duplicate":
        event = b'{"choices":[{"delta":{"content":"first","content":"second"}}]}'
    elif kind == "nonfinite":
        event = b'{"choices":[{"delta":{"content":"ok"}}],"unknown":NaN}'
    elif kind == "overdeep":
        event = (
            '{"choices":[{"delta":{"content":"ok"}}],"unknown":' + "[" * 40 + "0" + "]" * 40 + "}"
        ).encode()
    else:
        event = b'{"choices":[{"delta":{"content":"\xff"}}]}'
    body = b"data: " + event + b"\n\n"

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    client = OpenAiCompatibleClient(
        "glm",
        ProviderConfig(api_key="test", base_url="https://model.test/v1", model="model"),
        httpx.MockTransport(handler),
    )

    expected = "not valid UTF-8" if kind == "utf8" else "contains invalid JSON"
    with pytest.raises(LlmError, match=expected):
        await client.complete_streaming([Message("user", "inspect")])


async def test_streaming_accepts_standard_multiline_sse_data_event() -> None:
    body = (
        "event: message\r\n"
        'data: {"choices":[\r\n'
        'data: {"delta":{"content":"joined"}}]}\r\n'
        "\r\n"
        "data: [DONE]\r\n\r\n"
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    client = OpenAiCompatibleClient(
        "glm",
        ProviderConfig(api_key="test", base_url="https://model.test/v1", model="model"),
        httpx.MockTransport(handler),
    )

    assert (await client.complete_streaming([Message("user", "inspect")])).content == "joined"


async def test_missing_and_duplicate_tool_call_ids_are_repaired() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "same",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path":"a.py"}',
                                    },
                                },
                                {
                                    "id": "same",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path":"b.py"}',
                                    },
                                },
                                {
                                    "function": {
                                        "name": "list_dir",
                                        "arguments": "{}",
                                    }
                                },
                            ],
                        }
                    }
                ]
            },
        )

    client = OpenAiCompatibleClient(
        "glm",
        ProviderConfig(api_key="test", base_url="https://model.test/v1", model="model"),
        httpx.MockTransport(handler),
    )
    calls = (await client.complete([Message("user", "inspect")])).tool_calls

    assert calls[0].id == "same"
    assert calls[1].id.startswith("call_kairo_1_")
    assert calls[2].id.startswith("call_kairo_2_")
    assert len({call.id for call in calls}) == 3


@pytest.mark.parametrize(
    ("name", "arguments", "error"),
    [
        ("../escape", "{}", "invalid tool name"),
        ("read_file", "{broken", "malformed JSON arguments"),
        ("read_file", "[]", "must be an object"),
        ("read_file", '{"offset":NaN}', "malformed JSON arguments"),
        (
            "read_file",
            '{"path":"first","path":"second"}',
            "malformed JSON arguments",
        ),
        (
            "read_file",
            json.dumps({"path": "x" * (MAX_LLM_TOOL_ARGUMENT_BYTES + 1)}),
            "exceed the 1 MiB limit",
        ),
    ],
)
async def test_invalid_model_tool_calls_fail_before_dispatch(
    name: str, arguments: str, error: str
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {"name": name, "arguments": arguments},
                                }
                            ]
                        }
                    }
                ]
            },
        )

    client = OpenAiCompatibleClient(
        "glm",
        ProviderConfig(api_key="test", base_url="https://model.test/v1", model="model"),
        httpx.MockTransport(handler),
    )

    with pytest.raises(LlmError, match=error):
        await client.complete([Message("user", "inspect")])


@pytest.mark.parametrize(
    ("message", "usage", "error"),
    [
        ({"content": ["not text"]}, {}, "message content must be text"),
        (
            {"tool_calls": [{"id": 1, "function": {"name": "read_file", "arguments": {}}}]},
            {},
            "ID must be text",
        ),
        (
            {"tool_calls": [{"id": "call-1", "function": {"name": 123, "arguments": {}}}]},
            {},
            "name must be text",
        ),
        ({"content": "ok"}, {"prompt_tokens": True}, "bounded non-negative integer"),
        ({"content": "ok"}, {"completion_tokens": -1}, "bounded non-negative integer"),
        ({"content": "ok"}, {"cached_tokens": False}, "bounded non-negative integer"),
        (
            {"content": "ok"},
            {"prompt_tokens_details": []},
            "prompt_tokens_details must be an object",
        ),
    ],
)
async def test_non_streaming_response_contract_rejects_type_confusion(
    message: dict[str, object], usage: dict[str, object], error: str
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": message}], "usage": usage})

    client = OpenAiCompatibleClient(
        "glm",
        ProviderConfig(api_key="test", base_url="https://model.test/v1", model="model"),
        httpx.MockTransport(handler),
    )

    with pytest.raises(LlmError, match=error):
        await client.complete([Message("user", "inspect")])


@pytest.mark.parametrize(
    ("event", "error"),
    [
        (
            {"choices": [{"delta": {"tool_calls": [{"index": True}]}}]},
            "index must be a bounded integer",
        ),
        (
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": 1}]}}]},
            "ID delta must be text",
        ),
        (
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": 1}}]}}]},
            "name delta must be text",
        ),
        (
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": {"path": "x"}}}]
                        }
                    }
                ]
            },
            "arguments delta must be text",
        ),
        ({"usage": {"prompt_tokens": "10"}}, "bounded non-negative integer"),
    ],
)
async def test_streaming_response_contract_rejects_type_confusion(
    event: dict[str, object], error: str
) -> None:
    body = f"data: {json.dumps(event)}\n\n"

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    client = OpenAiCompatibleClient(
        "glm",
        ProviderConfig(api_key="test", base_url="https://model.test/v1", model="model"),
        httpx.MockTransport(handler),
    )

    with pytest.raises(LlmError, match=error):
        await client.complete_streaming([Message("user", "inspect")])


async def test_streaming_malformed_tool_arguments_are_rejected() -> None:
    body = (
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1",'
        '"function":{"name":"read_file","arguments":"{broken"}}]}}]}\n\n'
        "data: [DONE]\n\n"
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    client = OpenAiCompatibleClient(
        "glm",
        ProviderConfig(api_key="test", base_url="https://model.test/v1", model="model"),
        httpx.MockTransport(handler),
    )

    with pytest.raises(LlmError, match="malformed JSON arguments"):
        await client.complete_streaming([Message("user", "inspect")])


async def test_text_provider_omits_image_payload_but_keeps_notice() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        content = payload["messages"][0]["content"]
        assert all(part["type"] != "image_url" for part in content)
        assert "does not support image input" in content[-1]["text"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}], "usage": {}},
        )

    client = OpenAiCompatibleClient(
        "deepseek",
        ProviderConfig(api_key="test", base_url="https://model.test/v1", model="model"),
        httpx.MockTransport(handler),
    )
    response = await client.complete(
        [
            Message(
                "user",
                [
                    {"type": "text", "text": "analyze"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,aGVsbG8="},
                    },
                ],
            )
        ]
    )
    assert response.content == "ok"


async def test_glm_5v_sends_raw_base64_image_payload() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        image = payload["messages"][0]["content"][1]["image_url"]["url"]
        assert image == "aGVsbG8="
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}], "usage": {}},
        )

    client = OpenAiCompatibleClient(
        "glm",
        ProviderConfig(api_key="test", base_url="https://model.test/v1", model="glm-5v-turbo"),
        httpx.MockTransport(handler),
    )
    response = await client.complete(
        [
            Message(
                "user",
                [
                    {"type": "text", "text": "analyze"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,aGVsbG8="},
                    },
                ],
            )
        ]
    )
    assert response.content == "ok"


async def test_glm_5v_switches_from_coding_to_multimodal_endpoint() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/paas/v4/chat/completions"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}], "usage": {}},
        )

    client = OpenAiCompatibleClient(
        "glm",
        ProviderConfig(
            api_key="test",
            base_url="https://open.bigmodel.cn/api/coding/paas/v4",
            model="glm-5v-turbo",
        ),
        httpx.MockTransport(handler),
    )
    assert (await client.complete([Message("user", "hi")])).content == "ok"


async def test_step_reasoning_payload_and_full_chat_url_are_provider_specific() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert payload["reasoning_format"] == "deepseek-style"
        assert payload["reasoning_effort"] == "high"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}], "usage": {}},
        )

    client = OpenAiCompatibleClient(
        "step",
        ProviderConfig(
            api_key="test",
            base_url="https://model.test/v1/chat/completions",
            model="step-3.5-flash-2603",
        ),
        httpx.MockTransport(handler),
    )
    assert (await client.complete([Message("user", "hi")])).content == "ok"


async def test_provider_request_rejects_ambiguous_base_url_before_transport() -> None:
    called = False

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={})

    client = OpenAiCompatibleClient(
        "glm",
        ProviderConfig(
            api_key="test",
            base_url="https://model.test/v1?Authorization=secret",
            model="model",
        ),
        httpx.MockTransport(handler),
    )

    with pytest.raises(LlmError, match="query or fragment"):
        await client.complete([Message("user", "hi")])
    assert called is False


async def test_xfyun_omits_tools_and_sends_lora_header() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert "tools" not in payload
        assert payload["stream_options"]["include_usage"] is True
        assert request.headers["lora_id"] == "resource-card-id"
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
            headers={"content-type": "text/event-stream"},
        )

    client = OpenAiCompatibleClient(
        "xfyun",
        ProviderConfig(
            api_key="test",
            base_url="https://maas.test/v2",
            model="model-card-id",
            lora_id="resource-card-id",
        ),
        httpx.MockTransport(handler),
    )
    tools = [{"type": "function", "function": {"name": "read_file"}}]
    assert (await client.complete_streaming([Message("user", "hi")], tools)).content == "ok"


async def test_lora_header_is_never_forwarded_to_other_providers() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert "lora_id" not in request.headers
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}], "usage": {}},
        )

    client = OpenAiCompatibleClient(
        "step",
        ProviderConfig(
            api_key="test",
            base_url="https://model.test/v1",
            model="step-3.5-flash",
            lora_id="must-not-leak",
        ),
        httpx.MockTransport(handler),
    )
    assert (await client.complete([Message("user", "hi")])).content == "ok"


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("stepfun", "step"),
        ("moonshot", "kimi"),
        ("free-llm-api", "freellmapi"),
        ("maas", "xfyun"),
        ("agnes-ai", "agnes"),
    ],
)
def test_client_factory_resolves_reference_provider_aliases(alias: str, canonical: str) -> None:
    config = AppConfig(
        providers={
            canonical: ProviderConfig(
                api_key="test",
                base_url="https://model.test/v1",
                model="test-model",
            )
        }
    )

    client = create_llm_client(config, alias)

    assert client.provider == canonical


async def test_reasoning_aliases_and_details_are_merged() -> None:
    events = [
        {"choices": [{"delta": {"reasoning": "first "}}]},
        {
            "choices": [
                {
                    "delta": {
                        "reasoning_details": [
                            {"text": "second "},
                            {"content": "third"},
                        ]
                    }
                }
            ]
        },
        {"choices": [{"delta": {"content": "ok"}}]},
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    client = OpenAiCompatibleClient(
        "step",
        ProviderConfig(api_key="test", base_url="https://model.test/v1", model="step"),
        httpx.MockTransport(handler),
    )
    response = await client.complete_streaming([Message("user", "hi")])
    assert response.reasoning_content == "first second third"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            'data: {"error":{"code":10040,"message":"image rejected"}}\n\n',
            "10040 - image rejected",
        ),
        ("data: [DONE]\n\n", "Model returned no content"),
    ],
)
async def test_streaming_errors_are_visible(body: str, expected: str) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    client = OpenAiCompatibleClient(
        "glm",
        ProviderConfig(api_key="test", base_url="https://model.test/v1", model="model"),
        httpx.MockTransport(handler),
    )
    with pytest.raises(LlmError, match=expected):
        await client.complete_streaming([Message("user", "hi")])


async def test_http_error_includes_bounded_upstream_body() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text='{"error":"rate limited"}', headers={"retry-after": "2"})

    client = OpenAiCompatibleClient(
        "glm",
        ProviderConfig(api_key="test", base_url="https://model.test/v1", model="model"),
        httpx.MockTransport(handler),
    )
    with pytest.raises(LlmError, match="HTTP 429.*rate limited") as captured:
        await client.complete([Message("user", "hi")])
    assert captured.value.retryable is True
    assert captured.value.retry_after == 2


async def test_provider_uses_long_reasoning_read_timeout_and_bounded_phases() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        timeout = request.extensions["timeout"]
        assert timeout == {
            "connect": 60.0,
            "read": 300.0,
            "write": 60.0,
            "pool": 60.0,
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}], "usage": {}},
        )

    client = OpenAiCompatibleClient(
        "glm",
        ProviderConfig(api_key="test", base_url="https://model.test/v1", model="model"),
        httpx.MockTransport(handler),
    )

    assert (await client.complete([Message("user", "hi")])).content == "ok"


async def test_provider_overall_call_timeout_is_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_module, "LLM_CALL_TIMEOUT_SECONDS", 0.01)

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1)
        return httpx.Response(200, json={"choices": [{"message": {"content": "late"}}]})

    client = OpenAiCompatibleClient(
        "glm",
        ProviderConfig(api_key="test", base_url="https://model.test/v1", model="model"),
        httpx.MockTransport(handler),
    )

    with pytest.raises(LlmError) as captured:
        await client.complete([Message("user", "hi")])
    assert captured.value.retryable is True


async def test_non_streaming_model_response_is_read_with_a_hard_byte_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_module, "MAX_LLM_RESPONSE_BYTES", 128)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "x" * 256}}]},
        )

    client = OpenAiCompatibleClient(
        "glm",
        ProviderConfig(api_key="test", base_url="https://model.test/v1", model="model"),
        httpx.MockTransport(handler),
    )

    with pytest.raises(LlmError, match="response exceeds"):
        await client.complete([Message("user", "hi")])


@pytest.mark.parametrize(
    ("response_limit", "event_limit", "expected"),
    [(120, 1_000, "response exceeds"), (1_000, 80, "SSE event exceeds")],
)
async def test_streaming_model_response_and_event_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
    response_limit: int,
    event_limit: int,
    expected: str,
) -> None:
    monkeypatch.setattr(llm_module, "MAX_LLM_RESPONSE_BYTES", response_limit)
    monkeypatch.setattr(llm_module, "MAX_LLM_SSE_EVENT_BYTES", event_limit)
    event = {"choices": [{"delta": {"content": "x" * 200}}]}
    body = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n"

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    client = OpenAiCompatibleClient(
        "glm",
        ProviderConfig(api_key="test", base_url="https://model.test/v1", model="model"),
        httpx.MockTransport(handler),
    )

    with pytest.raises(LlmError, match=expected):
        await client.complete_streaming([Message("user", "hi")])


async def test_model_http_error_body_is_bounded_before_reporting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_module, "MAX_LLM_ERROR_BYTES", 16)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="visible-prefix--secret-tail")

    client = OpenAiCompatibleClient(
        "glm",
        ProviderConfig(api_key="test", base_url="https://model.test/v1", model="model"),
        httpx.MockTransport(handler),
    )

    with pytest.raises(LlmError) as captured:
        await client.complete([Message("user", "hi")])
    assert "visible-prefix" in str(captured.value)
    assert "secret-tail" not in str(captured.value)
    assert str(captured.value).endswith("...")


async def test_upstream_model_errors_are_redacted_before_reaching_callers() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            text="authorization: Bearer raw-secret token=another-secret",
        )

    client = OpenAiCompatibleClient(
        "glm",
        ProviderConfig(api_key="test", base_url="https://model.test/v1", model="model"),
        httpx.MockTransport(handler),
    )

    with pytest.raises(LlmError) as captured:
        await client.complete([Message("user", "hi")])
    error = str(captured.value)
    assert "raw-secret" not in error
    assert "another-secret" not in error
    assert "token=***" in error
    assert error.count("***") >= 2
    assert captured.value.retryable is False


@pytest.mark.parametrize("streaming", [False, True])
async def test_transport_error_wrapping_survives_unprintable_exception(
    streaming: bool,
) -> None:
    class UnprintableTransportError(RuntimeError):
        def __str__(self) -> str:
            raise KeyboardInterrupt

    async def handler(_request: httpx.Request) -> httpx.Response:
        raise UnprintableTransportError

    client = OpenAiCompatibleClient(
        "glm",
        ProviderConfig(api_key="test", base_url="https://model.test/v1", model="model"),
        httpx.MockTransport(handler),
    )

    with pytest.raises(LlmError) as captured:
        if streaming:
            await client.complete_streaming([Message("user", "hi")])
        else:
            await client.complete([Message("user", "hi")])

    assert "UnprintableTransportError message unavailable" in str(captured.value)


@pytest.mark.parametrize(
    ("provider", "included"),
    [("deepseek", True), ("kimi", True), ("glm", False)],
)
async def test_reasoning_history_only_for_compatible_providers(
    provider: str, included: bool
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        message = json.loads(request.content)["messages"][0]
        assert ("reasoning_content" in message) is included
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}], "usage": {}},
        )

    client = OpenAiCompatibleClient(
        provider,
        ProviderConfig(api_key="test", base_url="https://model.test/v1", model="model"),
        httpx.MockTransport(handler),
    )
    message = Message(
        "assistant",
        "",
        tool_calls=[ToolCall("call-1", "read_file", {"path": "README.md"})],
        reasoning_content="private reasoning",
    )
    assert (await client.complete([message])).content == "ok"
