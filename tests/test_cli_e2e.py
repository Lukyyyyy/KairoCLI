import asyncio
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

import pytest

SOURCE_ROOT = str(Path(__file__).resolve().parents[1] / "src")


class LocalStreamingProvider:
    def __init__(self) -> None:
        self.server: asyncio.Server | None = None
        self.requests: list[dict[str, Any]] = []

    async def __aenter__(self) -> "LocalStreamingProvider":
        try:
            self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        except PermissionError:
            pytest.skip("loopback bind is unavailable in the current sandbox")
        return self

    async def __aexit__(self, *_args: object) -> None:
        assert self.server is not None
        self.server.close()
        await self.server.wait_closed()

    @property
    def base_url(self) -> str:
        assert self.server is not None and self.server.sockets
        port = self.server.sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}/v1"

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            headers = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
            content_length = 0
            for line in headers.decode("ascii").split("\r\n")[1:]:
                name, separator, value = line.partition(":")
                if separator and name.casefold() == "content-length":
                    content_length = int(value.strip())
            body = await asyncio.wait_for(reader.readexactly(content_length), 5)
            payload = json.loads(body)
            assert isinstance(payload, dict)
            self.requests.append(payload)
            if "fail request" in body.decode("utf-8"):
                await self._respond(
                    writer,
                    401,
                    b"authorization: Bearer upstream-secret token=second-secret",
                    "text/plain",
                )
                return
            messages = payload.get("messages", [])
            transcript = json.dumps(messages, ensure_ascii=False)
            tool_messages = [
                message
                for message in messages
                if isinstance(message, dict) and message.get("role") == "tool"
            ]
            if not tool_messages and "read tool request" in transcript:
                await self._respond_tool_call(
                    writer,
                    "call-read-1",
                    "read_file",
                    {"path": "sample.txt"},
                )
                return
            if not tool_messages and "write tool request" in transcript:
                await self._respond_tool_call(
                    writer,
                    "call-write-1",
                    "write_file",
                    {"path": "created.txt", "content": "created by real CLI E2E"},
                )
                return
            if not tool_messages and "patch tool request" in transcript:
                patch = """diff --git a/patched.txt b/patched.txt
--- a/patched.txt
+++ b/patched.txt
@@ -1 +1 @@
-before
+after
"""
                await self._respond_tool_call(
                    writer,
                    "call-patch-1",
                    "apply_patch",
                    {"patch": patch},
                )
                return
            if not tool_messages and "project tool request" in transcript:
                await self._respond_tool_call(
                    writer,
                    "call-project-1",
                    "create_project",
                    {"path": "generated", "kind": "python"},
                )
                return
            if not tool_messages and "command tool request" in transcript:
                command = (
                    f"{shlex.quote(sys.executable)} -c "
                    + shlex.quote(
                        "from pathlib import Path; "
                        "Path('command-created.txt').write_text('command side effect'); "
                        "print('command-ok')"
                    )
                )
                await self._respond_tool_call(
                    writer,
                    "call-command-1",
                    "execute_command",
                    {"command": command, "timeout": 5},
                )
                return
            if not tool_messages and "command timeout request" in transcript:
                marker = "command-timeout-marker"
                child = (
                    "import time; from pathlib import Path; time.sleep(0.8); "
                    f"Path({marker!r}).write_text('orphan')"
                )
                command = f"{shlex.quote(sys.executable)} -c {shlex.quote(child)} &"
                await self._respond_tool_call(
                    writer,
                    "call-command-timeout-1",
                    "execute_command",
                    {"command": command, "timeout": 0.1},
                )
                return
            if tool_messages and "read tool request" in transcript:
                answer = "tool-read:" + str(tool_messages[-1].get("content", ""))
            elif tool_messages and "write tool request" in transcript:
                tool_result = str(tool_messages[-1].get("content", ""))
                answer = (
                    "tool-write:denied"
                    if "denied" in tool_result.casefold()
                    else "tool-write:executed"
                )
            elif tool_messages and "patch tool request" in transcript:
                tool_result = str(tool_messages[-1].get("content", ""))
                answer = (
                    "tool-patch:denied"
                    if "denied" in tool_result.casefold()
                    else "tool-patch:executed"
                )
            elif tool_messages and "project tool request" in transcript:
                tool_result = str(tool_messages[-1].get("content", ""))
                answer = (
                    "tool-project:denied"
                    if "denied" in tool_result.casefold()
                    else "tool-project:executed"
                )
            elif tool_messages and "command tool request" in transcript:
                tool_result = str(tool_messages[-1].get("content", ""))
                answer = (
                    "tool-command:denied"
                    if "denied" in tool_result.casefold()
                    else "tool-command:executed"
                )
            elif tool_messages and "command timeout request" in transcript:
                answer = "tool-command:timeout:" + str(
                    tool_messages[-1].get("content", "")
                )
            else:
                answer = f"answer-{len(self.requests)} 世界"
            await self._respond_answer(writer, answer)
        finally:
            writer.close()
            await writer.wait_closed()

    async def _respond_answer(self, writer: asyncio.StreamWriter, answer: str) -> None:
        events = [
            {"choices": [{"delta": {"content": answer[:8]}}]},
            {"choices": [{"delta": {"content": answer[8:]}}]},
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 4,
                    "prompt_tokens_details": {"cached_tokens": 2},
                },
            },
        ]
        encoded = (
            "".join(
                f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                for event in events
            )
            + "data: [DONE]\n\n"
        ).encode("utf-8")
        await self._respond(writer, 200, encoded, "text/event-stream")

    async def _respond_tool_call(
        self,
        writer: asyncio.StreamWriter,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> None:
        events = [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": call_id,
                                    "function": {
                                        "name": name,
                                        "arguments": json.dumps(arguments),
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            },
        ]
        encoded = (
            "".join(f"data: {json.dumps(event)}\n\n" for event in events)
            + "data: [DONE]\n\n"
        ).encode()
        await self._respond(writer, 200, encoded, "text/event-stream")

    @staticmethod
    async def _respond(
        writer: asyncio.StreamWriter,
        status: int,
        body: bytes,
        content_type: str,
    ) -> None:
        reason = "OK" if status == 200 else "Unauthorized"
        writer.write(
            f"HTTP/1.1 {status} {reason}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n".encode("ascii")
            + body
        )
        await writer.drain()


async def run_cli(
    workspace: Path,
    home: Path,
    provider: LocalStreamingProvider,
    *arguments: str,
) -> tuple[int, str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "HOME": str(home),
            "KAIROCLI_PROVIDER": "glm",
            "GLM_API_KEY": "local-test-key",
            "GLM_BASE_URL": provider.base_url,
            "GLM_MODEL": "local-stream-model",
            "KAIROCLI_SNAPSHOT_ENABLED": "0",
            "KAIROCLI_TRACE_ENABLED": "0",
            "PYTHONPATH": os.getenv(
                "KAIROCLI_E2E_PYTHONPATH",
                SOURCE_ROOT,
            ),
        }
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "kairocli",
        *arguments,
        cwd=workspace,
        env=environment,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
    return process.returncode or 0, stdout.decode(), stderr.decode()


@pytest.mark.parametrize("output_format", ["text", "json", "jsonl"])
async def test_real_cli_process_streaming_output_contracts(
    tmp_path: Path, output_format: str
) -> None:
    workspace = tmp_path / "work"
    home = tmp_path / "home"
    workspace.mkdir()
    home.mkdir()
    async with LocalStreamingProvider() as provider:
        code, stdout, stderr = await run_cli(
            workspace,
            home,
            provider,
            "-p",
            "hello subprocess",
            "--output-format",
            output_format,
        )

    assert code == 0
    assert stderr == ""
    assert len(provider.requests) == 1
    assert provider.requests[0]["stream"] is True
    assert provider.requests[0]["model"] == "local-stream-model"
    if output_format == "text":
        assert stdout == "answer-1 世界\n"
    elif output_format == "json":
        payload = json.loads(stdout)
        assert payload["status"] == "success"
        assert payload["result"] == "answer-1 世界"
        assert payload["usage"]["cached_tokens"] == 2
    else:
        events = [json.loads(line) for line in stdout.splitlines()]
        assert [event["type"] for event in events] == ["start", "result"]
        assert events[-1]["status"] == "success"
        assert events[-1]["result"] == "answer-1 世界"


async def test_real_cli_process_persists_and_resumes_session(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    home = tmp_path / "home"
    workspace.mkdir()
    home.mkdir()
    async with LocalStreamingProvider() as provider:
        first_code, first_stdout, first_stderr = await run_cli(
            workspace,
            home,
            provider,
            "-p",
            "first turn",
            "--output-format",
            "json",
            "--save-session",
        )
        first = json.loads(first_stdout)
        second_code, second_stdout, second_stderr = await run_cli(
            workspace,
            home,
            provider,
            "-p",
            "second turn",
            "--output-format",
            "json",
            "--resume",
            first["session_id"],
        )

    assert first_code == second_code == 0
    assert first_stderr == second_stderr == ""
    assert json.loads(second_stdout)["result"] == "answer-2 世界"
    second_messages = provider.requests[1]["messages"]
    assert [message["role"] for message in second_messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert second_messages[1]["content"] == "first turn"
    assert second_messages[2]["content"] == "answer-1 世界"
    assert second_messages[3]["content"] == "second turn"


async def test_real_cli_process_failure_is_structured_redacted_and_nonzero(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "work"
    home = tmp_path / "home"
    workspace.mkdir()
    home.mkdir()
    async with LocalStreamingProvider() as provider:
        code, stdout, stderr = await run_cli(
            workspace,
            home,
            provider,
            "-p",
            "fail request",
            "--output-format",
            "jsonl",
        )

    events = [json.loads(line) for line in stdout.splitlines()]
    assert code == 1
    assert stderr == ""
    assert [event["type"] for event in events] == ["start", "result"]
    assert events[-1]["status"] == "error"
    error = events[-1]["error"]["message"]
    assert "upstream-secret" not in error
    assert "second-secret" not in error
    assert "***" in error


async def test_real_cli_process_executes_streamed_read_tool_roundtrip(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "work"
    home = tmp_path / "home"
    workspace.mkdir()
    home.mkdir()
    (workspace / "sample.txt").write_text("real subprocess file content", encoding="utf-8")
    async with LocalStreamingProvider() as provider:
        code, stdout, stderr = await run_cli(
            workspace,
            home,
            provider,
            "-p",
            "read tool request",
        )

    assert code == 0
    assert stderr == ""
    assert stdout.startswith("tool-read:")
    assert "real subprocess file content" in stdout
    assert len(provider.requests) == 2
    second_messages = provider.requests[1]["messages"]
    assert [message["role"] for message in second_messages][-2:] == [
        "assistant",
        "tool",
    ]
    assert second_messages[-1]["tool_call_id"] == "call-read-1"


async def test_real_cli_process_denies_then_exactly_allows_write_tool(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "work"
    home = tmp_path / "home"
    workspace.mkdir()
    home.mkdir()
    target = workspace / "created.txt"
    async with LocalStreamingProvider() as provider:
        denied_code, denied_stdout, denied_stderr = await run_cli(
            workspace,
            home,
            provider,
            "-p",
            "write tool request",
        )
        assert not target.exists()
        allowed_code, allowed_stdout, allowed_stderr = await run_cli(
            workspace,
            home,
            provider,
            "-p",
            "write tool request",
            "--allow-tool",
            "write_file",
        )

    assert denied_code == allowed_code == 0
    assert denied_stderr == allowed_stderr == ""
    assert denied_stdout == "tool-write:denied\n"
    assert allowed_stdout == "tool-write:executed\n"
    assert target.read_text(encoding="utf-8") == "created by real CLI E2E"
    assert len(provider.requests) == 4
    denied_tool = provider.requests[1]["messages"][-1]
    allowed_tool = provider.requests[3]["messages"][-1]
    assert "denied" in denied_tool["content"].casefold()
    allowed_result = json.loads(allowed_tool["content"])
    assert allowed_result == {
        "path": "created.txt",
        "bytes": len("created by real CLI E2E"),
        "diagnostics": [],
    }


async def test_real_cli_process_denies_then_exactly_allows_command_tool(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "work"
    home = tmp_path / "home"
    workspace.mkdir()
    home.mkdir()
    target = workspace / "command-created.txt"
    async with LocalStreamingProvider() as provider:
        denied = await run_cli(workspace, home, provider, "-p", "command tool request")
        assert not target.exists()
        allowed = await run_cli(
            workspace,
            home,
            provider,
            "-p",
            "command tool request",
            "--allow-tool",
            "execute_command",
        )

    assert denied == (0, "tool-command:denied\n", "")
    assert allowed == (0, "tool-command:executed\n", "")
    assert target.read_text(encoding="utf-8") == "command side effect"
    result = json.loads(provider.requests[3]["messages"][-1]["content"])
    assert result["exit_code"] == 0
    assert result["stdout"] == "command-ok\n"
    assert result["timed_out"] is False


async def test_real_cli_process_denies_then_exactly_allows_patch_tool(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "work"
    home = tmp_path / "home"
    workspace.mkdir()
    home.mkdir()
    target = workspace / "patched.txt"
    target.write_text("before\n", encoding="utf-8")
    async with LocalStreamingProvider() as provider:
        denied = await run_cli(workspace, home, provider, "-p", "patch tool request")
        assert target.read_text(encoding="utf-8") == "before\n"
        allowed = await run_cli(
            workspace,
            home,
            provider,
            "-p",
            "patch tool request",
            "--allow-tool",
            "apply_patch",
        )

    assert denied == (0, "tool-patch:denied\n", "")
    assert allowed == (0, "tool-patch:executed\n", "")
    assert target.read_text(encoding="utf-8") == "after\n"
    result = json.loads(provider.requests[3]["messages"][-1]["content"])
    assert result["changed"] == ["patched.txt"]
    assert result["deleted"] == []
    assert result["files"] == 1


async def test_real_cli_process_denies_then_atomically_creates_project(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "work"
    home = tmp_path / "home"
    workspace.mkdir()
    home.mkdir()
    target = workspace / "generated"
    async with LocalStreamingProvider() as provider:
        denied = await run_cli(workspace, home, provider, "-p", "project tool request")
        assert not target.exists()
        allowed = await run_cli(
            workspace,
            home,
            provider,
            "-p",
            "project tool request",
            "--allow-tool",
            "create_project",
        )

    assert denied == (0, "tool-project:denied\n", "")
    assert allowed == (0, "tool-project:executed\n", "")
    assert (target / "main.py").is_file()
    assert (target / "requirements.txt").is_file()
    assert (target / "pyproject.toml").is_file()
    assert (target / "generated" / "__init__.py").is_file()
    result = json.loads(provider.requests[3]["messages"][-1]["content"])
    assert result["path"] == "generated"
    assert result["kind"] == "python"
    assert "pyproject.toml" in result["files"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group assertion")
async def test_real_cli_command_timeout_kills_background_pipe_owner(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "work"
    home = tmp_path / "home"
    workspace.mkdir()
    home.mkdir()
    marker = workspace / "command-timeout-marker"
    async with LocalStreamingProvider() as provider:
        code, stdout, stderr = await run_cli(
            workspace,
            home,
            provider,
            "-p",
            "command timeout request",
            "--allow-tool",
            "execute_command",
        )

    assert code == 0
    assert stderr == ""
    assert stdout.startswith("tool-command:timeout:")
    result = json.loads(provider.requests[1]["messages"][-1]["content"])
    assert result["timed_out"] is True
    assert result["exit_code"] is None
    await asyncio.sleep(0.8)
    assert not marker.exists()
