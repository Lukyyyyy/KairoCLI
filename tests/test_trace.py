import os
from pathlib import Path
from typing import Any

import pytest

from kairocli.agent import Agent
from kairocli.cli import _handle_trace
from kairocli.commands import CommandType, parse_command
from kairocli.llm import LlmClient, LlmError
from kairocli.memory import MemoryStore
from kairocli.models import LlmResponse, Message, ToolCall, ToolOutput, Usage
from kairocli.paths import KairoPaths
from kairocli.tools import ToolDefinition, ToolRegistry
from kairocli.trace import LlmTraceLogger, redact_sensitive_text


class RecordingConsole:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def print(self, message: str) -> None:
        self.messages.append(message)


class SimpleClient(LlmClient):
    provider = "test"
    model = "test-model"

    def __init__(self, content: str = "hello", reasoning: str | None = None) -> None:
        self._content = content
        self._reasoning = reasoning

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        return LlmResponse(
            content=self._content,
            reasoning_content=self._reasoning,
            usage=Usage(10, 5, 2),
        )


class FailureClient(SimpleClient):
    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        raise LlmError("upstream failure token=leaked-secret")


class CredentialClient(SimpleClient):
    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        return LlmResponse(
            content="api_key=super-secret answer",
            reasoning_content=(
                "thinking GLM_API_KEY=top-secret Authorization: Bearer abc.def "
                "data:image/png;base64," + "A" * 300
            ),
            usage=Usage(10, 5, 2),
        )


# ── redaction tests (these do not depend on the session format) ──────────────

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
            '{"OPENAI_API_KEY": "json-secret\\"suffix"}',
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


# ── session lifecycle tests ───────────────────────────────────────────────────

async def test_trace_disabled_creates_no_files(tmp_path: Path) -> None:
    logger = LlmTraceLogger(tmp_path / "traces", enabled=False)
    agent = Agent(SimpleClient(), ToolRegistry(tmp_path), "system", trace_logger=logger)
    assert await agent.run("hello") == "hello"
    assert not (tmp_path / "traces").exists()
    await agent.tools.close()


async def test_trace_records_full_session_content(tmp_path: Path) -> None:
    logger = LlmTraceLogger(tmp_path / "traces", enabled=True)
    agent = Agent(
        SimpleClient("the answer"),
        ToolRegistry(tmp_path),
        "sys-prompt",
        trace_logger=logger,
    )
    await agent.run("user-question")

    files = list((tmp_path / "traces").glob("sess-*.log"))
    assert len(files) == 1
    raw = files[0].read_text(encoding="utf-8")

    # Session structure markers
    assert "KAIRO TRACE" in raw
    assert "DONE" in raw
    assert "[REQUEST #1]" in raw
    assert "[MESSAGE #1 SYSTEM]" in raw
    assert "[MESSAGE #2 USER]" in raw
    assert "[TOOL SCHEMAS]" in raw
    assert "[CALL #1]" in raw
    assert "[ASSISTANT]" in raw

    # Content is recorded
    assert "sys-prompt" in raw
    assert "user-question" in raw
    assert "the answer" in raw

    # Reasoning not present when not enabled
    assert "[REASONING]" not in raw

    # File permissions
    if os.name == "posix":
        assert (files[0].stat().st_mode & 0o777) == 0o600
        assert ((tmp_path / "traces").stat().st_mode & 0o777) == 0o700

    await agent.tools.close()


async def test_trace_records_complete_model_request_context(tmp_path: Path) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    memory = MemoryStore(paths)
    memory.save("用户偏好使用 Java 开发", "global")
    tools = ToolRegistry(paths.workspace)

    async def inspect_memory(arguments: dict[str, Any]) -> str:
        return str(arguments)

    tools.register(
        ToolDefinition(
            "inspect_memory",
            "Inspect injected memory context",
            {"type": "object", "properties": {"query": {"type": "string"}}},
            inspect_memory,
        )
    )
    logger = LlmTraceLogger(tmp_path / "traces", enabled=True)
    agent = Agent(
        SimpleClient("answer"),
        tools,
        "base instructions",
        memory_store=memory,
        trace_logger=logger,
    )
    agent.history.extend(
        [
            Message("user", "较早的短期记忆"),
            Message("assistant", "较早的回答"),
            Message("tool", "历史工具结果", tool_call_id="call_old"),
        ]
    )

    await agent.run("Java 偏好")
    raw = next((tmp_path / "traces").glob("sess-*.log")).read_text(encoding="utf-8")

    assert "[REQUEST #1]" in raw
    assert "<relevant_long_term_memory>" in raw
    assert "用户偏好使用 Java 开发" in raw
    assert "较早的短期记忆" in raw
    assert "较早的回答" in raw
    assert "历史工具结果" in raw
    assert "tool_call_id=call_old" in raw
    assert "Java 偏好" in raw
    assert "[TOOL SCHEMAS]" in raw
    assert "inspect_memory" in raw
    assert "Inspect injected memory context" in raw

    await agent.tools.close()


async def test_trace_redacts_credentials_in_content(tmp_path: Path) -> None:
    logger = LlmTraceLogger(tmp_path / "traces", enabled=True)
    agent = Agent(CredentialClient(), ToolRegistry(tmp_path), "system", trace_logger=logger)
    await agent.run("request with api_key=leaked-key")

    raw = list((tmp_path / "traces").glob("sess-*.log"))[0].read_text(encoding="utf-8")

    # Secrets must not appear anywhere
    assert "super-secret" not in raw
    assert "top-secret" not in raw
    assert "leaked-key" not in raw
    assert "abc.def" not in raw
    assert "data:image" not in raw
    assert "A" * 240 not in raw

    # Safe content survives
    assert "KAIRO TRACE" in raw
    assert "[ASSISTANT]" in raw

    await agent.tools.close()


async def test_trace_reasoning_requires_opt_in(tmp_path: Path) -> None:
    logger = LlmTraceLogger(tmp_path / "traces", enabled=True, include_reasoning=True)
    agent = Agent(
        CredentialClient(), ToolRegistry(tmp_path), "system", trace_logger=logger
    )
    await agent.run("request")

    raw = list((tmp_path / "traces").glob("sess-*.log"))[0].read_text(encoding="utf-8")

    assert "[REASONING]" in raw
    assert "thinking GLM_API_KEY=***" in raw   # key redacted but prefix preserved
    assert "top-secret" not in raw
    assert "abc.def" not in raw
    assert "<image-data-redacted>" in raw

    await agent.tools.close()


async def test_trace_error_session_is_recorded(tmp_path: Path) -> None:
    logger = LlmTraceLogger(tmp_path / "traces", enabled=True)
    agent = Agent(FailureClient(), ToolRegistry(tmp_path), "system", trace_logger=logger)

    with pytest.raises(LlmError):
        await agent.run("request")

    files = list((tmp_path / "traces").glob("sess-*.log"))
    assert len(files) == 1
    raw = files[0].read_text(encoding="utf-8")

    assert "[ERROR]" in raw
    assert "LlmError" in raw
    assert "leaked-secret" not in raw   # redacted
    assert "ABORTED" in raw

    await agent.tools.close()


async def test_trace_survives_broken_directory_without_breaking_agent(tmp_path: Path) -> None:
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("file", encoding="utf-8")
    logger = LlmTraceLogger(blocked, enabled=True)
    agent = Agent(SimpleClient("ok"), ToolRegistry(tmp_path), "system", trace_logger=logger)
    assert await agent.run("still works") == "ok"
    await agent.tools.close()


async def test_trace_rejects_symlinked_user_container_without_external_writes(
    tmp_path: Path,
) -> None:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    paths.home.mkdir(parents=True)
    outside = tmp_path / "outside-state"
    traces = outside / "traces"
    traces.mkdir(parents=True)
    sentinel = traces / "sentinel.log"
    sentinel.write_text("external-secret\n", encoding="utf-8")
    paths.user_dir.symlink_to(outside, target_is_directory=True)
    logger = LlmTraceLogger.from_environment(paths)
    logger.enabled = True

    session_id = logger.open_session(scope="test", provider="test", model="test")
    await logger.record_llm_request(
        session_id=session_id,
        turn=1,
        messages=[Message("system", "sys"), Message("user", "input")],
        tools=[],
    )
    await logger.close_session(
        session_id=session_id,
        duration_ms=1,
        total_calls=0,
        total_input=0,
        total_output=0,
        total_cached=0,
    )

    assert sentinel.read_text(encoding="utf-8") == "external-secret\n"
    assert list(traces.iterdir()) == [sentinel]


async def test_trace_rejects_symlinked_session_file_without_external_writes(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "traces"
    directory.mkdir()
    outside = tmp_path / "outside.log"
    outside.write_text("external-secret\n", encoding="utf-8")
    fake_session = directory / "sess-2099-01-01T00-00-00-aabbccdd1122.log"
    fake_session.symlink_to(outside)

    logger = LlmTraceLogger(directory, enabled=True)
    logger._session_files["fakeid"] = fake_session
    await logger.record_llm_request(
        session_id="fakeid",
        turn=1,
        messages=[Message("system", "sys"), Message("user", "input")],
        tools=[],
    )

    assert outside.read_text(encoding="utf-8") == "external-secret\n"


async def test_trace_prunes_excess_session_files(tmp_path: Path) -> None:
    from kairocli.trace import MAX_TRACE_FILES

    directory = tmp_path / "traces"
    directory.mkdir()
    # Create MAX_TRACE_FILES + 5 dummy session files
    for i in range(MAX_TRACE_FILES + 5):
        (directory / f"sess-2020-01-{i:02d}T00-00-00-{i:012d}.log").write_text("x")

    logger = LlmTraceLogger(directory, enabled=True)
    logger._prune()

    remaining = list(directory.glob("sess-*.log"))
    assert len(remaining) == MAX_TRACE_FILES


async def test_trace_slash_command_requires_explicit_reasoning_toggle(
    tmp_path: Path,
) -> None:
    logger = LlmTraceLogger(tmp_path / "traces")
    agent = Agent(SimpleClient(), ToolRegistry(tmp_path), "system", trace_logger=logger)
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


async def test_trace_records_tool_calls_and_results(tmp_path: Path) -> None:
    logger = LlmTraceLogger(tmp_path / "traces", enabled=True)
    session_id = logger.open_session(scope="test", provider="p", model="m")
    assert session_id is not None

    calls = [ToolCall(id="call_001", name="read_file", arguments={"path": "/tmp/x.py"})]
    results = [ToolOutput(text="file content here", elapsed_ms=42)]

    await logger.record_tool_results(session_id=session_id, calls=calls, results=results)
    await logger.close_session(
        session_id=session_id,
        duration_ms=500,
        total_calls=1,
        total_input=10,
        total_output=5,
        total_cached=0,
    )

    raw = list((tmp_path / "traces").glob("sess-*.log"))[0].read_text(encoding="utf-8")
    assert "[TOOL RESULT]  read_file" in raw
    assert "id=call_001" in raw
    assert "elapsed=42ms" in raw
    assert "file content here" in raw
    assert "DONE" in raw
