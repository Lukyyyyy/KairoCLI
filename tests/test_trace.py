import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from kairocli.agent import Agent
from kairocli.cli import _handle_trace
from kairocli.commands import CommandType, parse_command
from kairocli.llm import LlmClient, LlmError
from kairocli.models import LlmResponse, Message, Usage
from kairocli.paths import KairoPaths
from kairocli.tools import ToolRegistry
from kairocli.trace import LlmTraceLogger, redact_sensitive_text


class RecordingConsole:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def print(self, message: str) -> None:
        self.messages.append(message)


class TraceClient(LlmClient):
    provider = "test"
    model = "trace-model"

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        return LlmResponse(
            content="private answer body",
            reasoning_content=(
                "analysis GLM_API_KEY=super-secret Authorization: Bearer abc.def "
                "data:image/png;base64," + "A" * 300
            ),
            usage=Usage(12, 4, 3),
        )


class TraceFailureClient(TraceClient):
    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        raise LlmError("failed token=upstream-secret")


@pytest.mark.parametrize(
    ("value", "secrets", "preserved"),
    [
        (
            "fetch https://user:url-password@example.test/private",
            ("user", "url-password"),
            "example.test/private",
        ),
        (
            "https://example.test/cb?access_token=query-secret&state=visible",
            ("query-secret",),
            "state=visible",
        ),
        (
            '{"OPENAI_API_KEY": "json-secret\\\"suffix"}',
            ("json-secret", "suffix"),
            "OPENAI_API_KEY",
        ),
        (
            'run --api-key "flag secret value" --mode safe',
            ("flag secret value",),
            "--mode safe",
        ),
        (
            "Authorization: Basic dXNlcjpwYXNzd29yZA==",
            ("dXNlcjpwYXNzd29yZA",),
            "Authorization",
        ),
        (
            "sk-proj-abcdefghijklmnopqrstuvwx github_pat_abcdefghijklmnopqrstuvwxyz "
            "sk_live_abcdefghijklmnop hf_abcdefghijklmnopqrstuvwxyz",
            (
                "sk-proj-abcdefghijklmnopqrstuvwx",
                "github_pat_abcdefghijklmnopqrstuvwxyz",
                "sk_live_abcdefghijklmnop",
                "hf_abcdefghijklmnopqrstuvwxyz",
            ),
            "<credential-redacted>",
        ),
        (
            "eyJabcdefghijk.abcdefghijklmnop.signature123",
            ("eyJabcdefghijk", "signature123"),
            "<jwt-redacted>",
        ),
        (
            "before -----BEGIN PRIVATE KEY-----\nshort-secret-body\n"
            "-----END PRIVATE KEY----- after",
            ("short-secret-body",),
            "before <private-key-redacted> after",
        ),
    ],
)
def test_shared_redactor_covers_common_text_credential_forms(
    value: str, secrets: tuple[str, ...], preserved: str
) -> None:
    redacted = redact_sensitive_text(value)
    assert all(secret not in redacted for secret in secrets)
    assert preserved in redacted


def test_shared_redactor_preserves_similar_noncredential_text() -> None:
    value = "tokenizer=enabled https://example.test/path sk-short ordinary-key=value"
    assert redact_sensitive_text(value) == value


async def test_trace_is_opt_in_and_never_records_prompt_or_answer(tmp_path: Path) -> None:
    disabled = LlmTraceLogger(tmp_path / "disabled", enabled=False)
    agent = Agent(TraceClient(), ToolRegistry(tmp_path), "system", trace_logger=disabled)
    assert await agent.run("private user prompt") == "private answer body"
    assert not (tmp_path / "disabled").exists()
    await agent.tools.close()

    enabled = LlmTraceLogger(tmp_path / "enabled", enabled=True)
    agent = Agent(TraceClient(), ToolRegistry(tmp_path), "system", trace_logger=enabled)
    await agent.run("private user prompt")
    trace_file = next((tmp_path / "enabled").glob("*.jsonl"))
    raw = trace_file.read_text(encoding="utf-8")
    event = json.loads(raw)

    assert "private user prompt" not in raw
    assert "private answer body" not in raw
    assert "super-secret" not in raw
    assert "reasoning" not in event
    assert event["status"] == "success"
    assert event["response"]["content_chars"] == len("private answer body")
    assert event["usage"] == {
        "input_tokens": 12,
        "output_tokens": 4,
        "cached_tokens": 3,
    }
    if os.name == "posix":
        assert (trace_file.stat().st_mode & 0o777) == 0o600
        assert ((tmp_path / "enabled").stat().st_mode & 0o777) == 0o700
    await agent.tools.close()


async def test_reasoning_trace_requires_second_opt_in_and_redacts_payloads(
    tmp_path: Path,
) -> None:
    logger = LlmTraceLogger(
        tmp_path / "traces", enabled=True, include_reasoning=True
    )
    agent = Agent(TraceClient(), ToolRegistry(tmp_path), "system", trace_logger=logger)
    await agent.run("request")
    raw = next((tmp_path / "traces").glob("*.jsonl")).read_text(encoding="utf-8")
    event = json.loads(raw)

    assert event["reasoning"].startswith("analysis GLM_API_KEY=***")
    assert "super-secret" not in raw
    assert "abc.def" not in raw
    assert "data:image" not in raw
    assert "A" * 240 not in raw
    assert "<image-data-redacted>" in raw
    await agent.tools.close()


async def test_error_trace_is_typed_redacted_and_diagnostics_never_break_agent(
    tmp_path: Path,
) -> None:
    logger = LlmTraceLogger(tmp_path / "traces", enabled=True)
    agent = Agent(
        TraceFailureClient(), ToolRegistry(tmp_path), "system", trace_logger=logger
    )
    with pytest.raises(LlmError, match="upstream-secret"):
        await agent.run("request")
    event = json.loads(
        next((tmp_path / "traces").glob("*.jsonl")).read_text(encoding="utf-8")
    )
    assert event["status"] == "error"
    assert event["error"] == {"type": "LlmError", "message": "failed token=***"}
    await agent.tools.close()

    blocked = tmp_path / "not-a-directory"
    blocked.write_text("file", encoding="utf-8")
    broken_logger = LlmTraceLogger(blocked, enabled=True)
    healthy = Agent(
        TraceClient(), ToolRegistry(tmp_path), "system", trace_logger=broken_logger
    )
    assert await healthy.run("still succeeds") == "private answer body"
    await healthy.tools.close()


async def test_error_trace_survives_unprintable_exception(tmp_path: Path) -> None:
    class UnprintableError(RuntimeError):
        def __str__(self) -> str:
            raise KeyboardInterrupt

    logger = LlmTraceLogger(tmp_path / "traces", enabled=True)
    await logger.record_error(
        scope="react",
        provider="test",
        model="test",
        duration_ms=1,
        message_count=1,
        tool_schema_count=0,
        error=UnprintableError(),
    )
    event = json.loads(
        next((tmp_path / "traces").glob("*.jsonl")).read_text(encoding="utf-8")
    )
    assert event["error"] == {
        "type": "UnprintableError",
        "message": "UnprintableError message unavailable",
    }


async def test_trace_rejects_symlinked_user_container_without_external_writes(
    tmp_path: Path,
) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    paths.home.mkdir(parents=True)
    outside = tmp_path / "outside-state"
    traces = outside / "traces"
    traces.mkdir(parents=True)
    sentinel = traces / "sentinel.jsonl"
    sentinel.write_text("external-secret\n", encoding="utf-8")
    paths.user_dir.symlink_to(outside, target_is_directory=True)
    logger = LlmTraceLogger.from_environment(paths)
    logger.enabled = True

    await logger.record_error(
        scope="react",
        provider="test",
        model="test",
        duration_ms=1,
        message_count=1,
        tool_schema_count=0,
        error=RuntimeError("local failure"),
    )

    assert sentinel.read_text(encoding="utf-8") == "external-secret\n"
    assert list(traces.iterdir()) == [sentinel]


async def test_trace_rejects_symlinked_daily_file_without_external_writes(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "traces"
    directory.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text("external-secret\n", encoding="utf-8")
    target = directory / f"llm-trace-{date.today().isoformat()}.jsonl"
    target.symlink_to(outside)
    logger = LlmTraceLogger(directory, enabled=True)

    await logger.record_error(
        scope="react",
        provider="test",
        model="test",
        duration_ms=1,
        message_count=1,
        tool_schema_count=0,
        error=RuntimeError("local failure"),
    )

    assert outside.read_text(encoding="utf-8") == "external-secret\n"


async def test_trace_rejects_nonfinite_event_without_creating_invalid_json(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "traces"
    logger = LlmTraceLogger(directory, enabled=True)

    await logger._append({"timestamp": "now", "score": float("nan")})

    assert not directory.exists()


async def test_trace_lock_symlink_prevents_external_access(tmp_path: Path) -> None:
    directory = tmp_path / "traces"
    directory.mkdir()
    outside = tmp_path / "outside-trace-lock"
    outside.write_text("sentinel", encoding="utf-8")
    (directory / ".trace.lock").symlink_to(outside)
    logger = LlmTraceLogger(directory, enabled=True)

    await logger.record_error(
        scope="react",
        provider="test",
        model="test",
        duration_ms=1,
        message_count=1,
        tool_schema_count=0,
        error=RuntimeError("failure"),
    )

    assert outside.read_text(encoding="utf-8") == "sentinel"
    assert list(directory.glob("*.jsonl")) == []


async def test_trace_slash_command_requires_explicit_reasoning_toggle(
    tmp_path: Path,
) -> None:
    logger = LlmTraceLogger(tmp_path / "traces")
    agent = Agent(TraceClient(), ToolRegistry(tmp_path), "system", trace_logger=logger)
    console = RecordingConsole()
    assert parse_command("/trace status").type == CommandType.TRACE

    _handle_trace("on", agent, console)
    assert logger.enabled is True
    assert logger.include_reasoning is False
    _handle_trace("reasoning on", agent, console)
    assert logger.include_reasoning is True
    _handle_trace("off", agent, console)
    assert logger.enabled is False
    assert logger.include_reasoning is False
    assert str(tmp_path / "traces") in console.messages[-1]
    await agent.tools.close()
