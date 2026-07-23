import importlib
import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kairocli.agent import Agent
from kairocli.cli import build_parser, noninteractive
from kairocli.config import AppConfig
from kairocli.llm import LlmClient, LlmError
from kairocli.models import LlmResponse, Message, ToolCall, Usage
from kairocli.paths import KairoPaths
from kairocli.policy import ApprovalPolicy
from kairocli.sessions import SessionStore
from kairocli.todos import SessionTodoController
from kairocli.tools import ToolRegistry

cli_interactive_module = importlib.import_module("kairocli.cli.interactive")
cli_main_module = importlib.import_module("kairocli.cli.main")
cli_module = importlib.import_module("kairocli.cli.noninteractive")


class DisabledSnapshots:
    config = SimpleNamespace(enabled=False)

    async def close(self) -> None:
        return None


async def test_component_shutdown_attempts_all_services_and_reports_failures() -> None:
    closed: list[str] = []

    class Service:
        def __init__(self, name: str, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        async def close(self) -> None:
            closed.append(self.name)
            if self.fail:
                raise RuntimeError("close failed")

    warnings = await cli_interactive_module._close_components(
        ("Broken", Service("broken", True)),
        ("Healthy", Service("healthy")),
    )

    assert set(closed) == {"broken", "healthy"}
    assert warnings == ["Broken shutdown warning: close failed"]


class BatchClient(LlmClient):
    provider = "test"
    model = "batch-model"

    def __init__(self, answer: str = "batch answer", tool: bool = False) -> None:
        self.answer = answer
        self.tool = tool

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        if self.tool and not any(message.role == "tool" for message in messages):
            return LlmResponse(
                tool_calls=[
                    ToolCall(
                        "call-1",
                        "write_file",
                        {"path": "generated.txt", "content": "written"},
                    )
                ],
                usage=Usage(3, 1, 0),
            )
        return LlmResponse(content=self.answer, usage=Usage(5, 2, 1))


class FailingClient(BatchClient):
    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LlmResponse:
        raise LlmError("upstream failed")


def _paths(tmp_path: Path) -> KairoPaths:
    workspace = tmp_path / "work"
    workspace.mkdir()
    return KairoPaths.discover(workspace, tmp_path / "home")


def _factory(client: LlmClient):
    def create(
        paths: KairoPaths,
        _config: AppConfig,
        _provider: str | None = None,
        approval_policy: ApprovalPolicy | None = None,
        approver: Any = None,
        **kwargs: Any,
    ) -> Agent:
        tools = ToolRegistry(
            paths.workspace,
            approval_policy=approval_policy,
            approver=approver,
        )
        tools.snapshot_service = DisabledSnapshots()
        controller = kwargs.get("todo_controller")
        if isinstance(controller, SessionTodoController):
            controller.register(tools)
        return Agent(client, tools, "system")

    return create


async def test_noninteractive_text_json_and_jsonl_are_stdout_stable(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(cli_module, "make_agent", _factory(BatchClient()))
    config = AppConfig.load(paths)

    assert await noninteractive(paths, config, "hello", output_format="text") == 0
    captured = capsys.readouterr()
    assert captured.out == "batch answer\n"
    assert captured.err == ""

    assert await noninteractive(paths, config, "hello", output_format="json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert payload["type"] == "result"
    assert payload["status"] == "success"
    assert payload["result"] == "batch answer"
    assert payload["usage"] == {
        "input_tokens": 5,
        "output_tokens": 2,
        "cached_tokens": 1,
        "llm_calls": 1,
        "compactions": 0,
    }
    assert payload["session_id"] is None

    assert await noninteractive(paths, config, "hello", output_format="jsonl") == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["type"] for event in events] == ["start", "result"]
    assert [event["status"] for event in events] == ["running", "success"]


async def test_noninteractive_text_sanitizes_terminal_protocols_only(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    paths = _paths(tmp_path)
    answer = "[red]literal[/red]\x1b[31m colored\x1b[0m\x1b]52;c;secret\x07"
    monkeypatch.setattr(cli_module, "make_agent", _factory(BatchClient(answer)))

    assert await noninteractive(paths, AppConfig.load(paths), "hello") == 0
    captured = capsys.readouterr()
    assert captured.out == "[red]literal[/red] colored\n"
    assert "\x1b" not in captured.out


async def test_noninteractive_json_repairs_unicode_and_bounds_result(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    paths = _paths(tmp_path)
    answer = "prefix-\ud800-" + "界" * 100
    monkeypatch.setattr(cli_module, "make_agent", _factory(BatchClient(answer)))
    monkeypatch.setattr(cli_module, "MAX_NONINTERACTIVE_RESULT_BYTES", 96)

    assert await noninteractive(paths, AppConfig.load(paths), "hello", output_format="json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "success"
    assert "\ud800" not in payload["result"]
    assert "?" in payload["result"]
    assert "truncated: original_bytes=" in payload["result"]
    assert len(payload["result"].encode("utf-8")) <= 96


async def test_noninteractive_errors_are_redacted_bounded_and_total(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    paths = _paths(tmp_path)
    secret_error = LlmError("GLM_API_KEY=super-secret " + "x" * 1_000 + "\ud800")

    class SecretFailure(BatchClient):
        async def complete(
            self,
            messages: list[Message],
            tools: list[dict[str, Any]] | None = None,
        ) -> LlmResponse:
            raise secret_error

    monkeypatch.setattr(cli_module, "make_agent", _factory(SecretFailure()))
    monkeypatch.setattr(cli_module, "MAX_NONINTERACTIVE_ERROR_BYTES", 128)
    assert await noninteractive(paths, AppConfig.load(paths), "fail", output_format="json") == 1
    failed = json.loads(capsys.readouterr().out)
    message = failed["error"]["message"]
    assert "super-secret" not in message
    assert "\ud800" not in message
    assert len(message.encode("utf-8")) <= 128

    class UnprintableError(RuntimeError):
        def __str__(self) -> str:
            raise RuntimeError("string conversion failed")

    class UnprintableFailure(BatchClient):
        async def complete(
            self,
            messages: list[Message],
            tools: list[dict[str, Any]] | None = None,
        ) -> LlmResponse:
            raise UnprintableError()

    monkeypatch.setattr(cli_module, "make_agent", _factory(UnprintableFailure()))
    assert await noninteractive(paths, AppConfig.load(paths), "fail", output_format="json") == 1
    failed = json.loads(capsys.readouterr().out)
    assert failed["error"]["message"] == "Exception message was unavailable"


async def test_noninteractive_cleanup_warning_cannot_replace_success(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(cli_module, "make_agent", _factory(BatchClient("completed")))

    async def fail_close(_self: ToolRegistry) -> None:
        raise RuntimeError("Authorization: Bearer cleanup-secret\ud800")

    monkeypatch.setattr(ToolRegistry, "close", fail_close)
    assert await noninteractive(paths, AppConfig.load(paths), "hello", output_format="json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "success"
    assert payload["result"] == "completed"
    assert len(payload["warnings"]) == 1
    assert "cleanup-secret" not in payload["warnings"][0]
    assert "\ud800" not in payload["warnings"][0]


def test_noninteractive_json_normalizes_nonfinite_or_invalid_counters(capsys: Any) -> None:
    cli_module._write_json(
        {
            "type": "result",
            "status": "success",
            "result": "ok",
            "usage": {
                "input_tokens": float("nan"),
                "output_tokens": True,
                "cached_tokens": -1,
                "llm_calls": 10**20,
                "compactions": 2,
            },
            "duration_ms": float("inf"),
            "warnings": [],
        }
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["usage"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "llm_calls": 1_000_000_000_000,
        "compactions": 2,
    }
    assert payload["duration_ms"] == 0


async def test_noninteractive_denies_tools_by_default_and_allows_exact_opt_in(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(cli_module, "make_agent", _factory(BatchClient("finished", tool=True)))
    config = AppConfig.load(paths)

    denied = await noninteractive(paths, config, "write", output_format="json")
    denied_payload = json.loads(capsys.readouterr().out)
    assert denied == 0
    assert denied_payload["status"] == "success"
    assert not (paths.workspace / "generated.txt").exists()

    allowed = await noninteractive(
        paths,
        config,
        "write",
        output_format="json",
        allowed_tools=["write_file"],
    )
    assert allowed == 0
    assert json.loads(capsys.readouterr().out)["status"] == "success"
    assert (paths.workspace / "generated.txt").read_text(encoding="utf-8") == "written"


async def test_noninteractive_structures_errors_and_persists_only_on_request(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    paths = _paths(tmp_path)
    config = AppConfig.load(paths)
    monkeypatch.setattr(cli_module, "make_agent", _factory(FailingClient()))

    code = await noninteractive(paths, config, "fail", output_format="json")
    failed = json.loads(capsys.readouterr().out)
    assert code == 1
    assert failed["status"] == "error"
    assert failed["error"] == {"type": "LlmError", "message": "upstream failed"}

    monkeypatch.setattr(cli_module, "make_agent", _factory(BatchClient("saved")))
    code = await noninteractive(
        paths,
        config,
        "remember this turn",
        output_format="json",
        save_session=True,
    )
    saved = json.loads(capsys.readouterr().out)
    assert code == 0
    assert saved["session_id"].startswith("session_")
    state = SessionStore(paths.session_database).load(saved["session_id"], paths.workspace)
    assert state is not None
    assert [message.role for message in state.messages] == ["user", "assistant"]

    code = await noninteractive(
        paths,
        config,
        "continue turn",
        output_format="json",
        resume_id=saved["session_id"],
    )
    resumed = json.loads(capsys.readouterr().out)
    assert code == 0
    assert resumed["session_id"] == saved["session_id"]
    reloaded = SessionStore(paths.session_database).load(saved["session_id"], paths.workspace)
    assert reloaded is not None
    assert len(reloaded.messages) == 4


async def test_noninteractive_reads_stdin_and_rejects_invalid_approval_names(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    paths = _paths(tmp_path)
    config = AppConfig.load(paths)
    monkeypatch.setattr(cli_module, "make_agent", _factory(BatchClient("from stdin")))
    monkeypatch.setattr("sys.stdin", StringIO("stdin request"))

    assert await noninteractive(paths, config, "", output_format="json") == 0
    assert json.loads(capsys.readouterr().out)["result"] == "from stdin"
    assert (
        await noninteractive(
            paths,
            config,
            "request",
            output_format="json",
            allowed_tools=["bad/name"],
        )
        == 1
    )
    invalid = json.loads(capsys.readouterr().out)
    assert invalid["error"]["type"] == "ApprovalConfigError"


def test_noninteractive_parser_contract() -> None:
    parser = build_parser()
    stdin_mode = parser.parse_args(["--print", "--output-format", "jsonl"])
    assert stdin_mode.print_prompt == ""
    assert stdin_mode.output_format == "jsonl"
    explicit = parser.parse_args(
        [
            "-p",
            "run tests",
            "--mode",
            "plan",
            "--allow-tool",
            "execute_command",
            "--save-session",
        ]
    )
    assert explicit.print_prompt == "run tests"
    assert explicit.mode == "plan"
    assert explicit.allow_tool == ["execute_command"]
    assert explicit.save_session is True
    daemon = parser.parse_args(["wechat", "daemon", "restart"])
    assert daemon.action == "daemon"
    assert daemon.daemon_action == "restart"
    daemon_status = parser.parse_args(["wechat", "daemon"])
    assert daemon_status.daemon_action is None


def test_wechat_parser_rejects_daemon_action_for_other_commands(capsys: Any) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as raised:
        parser.parse_args(["wechat", "status", "start"])

    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unrecognized arguments: start" in captured.err


@pytest.mark.parametrize("port", ["0", "65536", "-1", "not-a-port"])
def test_serve_parser_rejects_invalid_ports(port: str, capsys: Any) -> None:
    with pytest.raises(SystemExit) as raised:
        build_parser().parse_args(["serve", "--http", "--port", port])

    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "port must be an integer from 1 to 65535" in captured.err


def test_run_server_rejects_invalid_programmatic_port(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    with pytest.raises(ValueError, match="port must be an integer from 1 to 65535"):
        cli_main_module.run_server(paths, AppConfig.load(paths), None, 65_536)


async def test_noninteractive_structures_session_startup_failures(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    paths = _paths(tmp_path)

    class UnavailableSessionStore:
        def __init__(self, _database: Path) -> None:
            raise PermissionError("session storage unavailable")

    monkeypatch.setattr(cli_module, "SessionStore", UnavailableSessionStore)
    code = await noninteractive(
        paths,
        AppConfig.load(paths),
        "resume this",
        output_format="json",
        resume_id="session_123456789abc",
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert code == 1
    assert payload["status"] == "error"
    assert payload["error"] == {
        "type": "PermissionError",
        "message": "session storage unavailable",
    }


def test_main_structures_os_errors_before_noninteractive_runtime(
    monkeypatch: Any, capsys: Any
) -> None:
    def fail_config(*_args: Any, **_kwargs: Any) -> AppConfig:
        raise PermissionError("configuration unavailable")

    monkeypatch.setattr(cli_main_module.AppConfig, "load", fail_config)
    with pytest.raises(SystemExit) as raised:
        cli_main_module.main(["-p", "hello", "--output-format", "json"])

    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["status"] == "error"
    assert payload["error"] == {
        "type": "PermissionError",
        "message": "configuration unavailable",
    }


def test_main_structures_unexpected_startup_errors(monkeypatch: Any, capsys: Any) -> None:
    def fail_config(*_args: Any, **_kwargs: Any) -> AppConfig:
        raise LookupError("provider registry unavailable")

    monkeypatch.setattr(cli_main_module.AppConfig, "load", fail_config)
    with pytest.raises(SystemExit) as raised:
        cli_main_module.main(["-p", "hello", "--output-format", "json"])

    assert raised.value.code == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["error"] == {
        "type": "LookupError",
        "message": "provider registry unavailable",
    }


def test_main_structures_unprintable_startup_error(monkeypatch: Any, capsys: Any) -> None:
    class UnprintableStartupError(RuntimeError):
        def __str__(self) -> str:
            raise KeyboardInterrupt

    def fail_config(*_args: Any, **_kwargs: Any) -> AppConfig:
        raise UnprintableStartupError

    monkeypatch.setattr(cli_main_module.AppConfig, "load", fail_config)
    with pytest.raises(SystemExit) as raised:
        cli_main_module.main(["-p", "hello", "--output-format", "json"])

    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["error"] == {
        "type": "UnprintableStartupError",
        "message": "Exception message was unavailable",
    }


def test_main_structures_workspace_discovery_errors(monkeypatch: Any, capsys: Any) -> None:
    def fail_discovery(_cls: type[KairoPaths]) -> KairoPaths:
        raise PermissionError("workspace unavailable")

    monkeypatch.setattr(cli_main_module.KairoPaths, "discover", classmethod(fail_discovery))
    with pytest.raises(SystemExit) as raised:
        cli_main_module.main(["-p", "hello", "--output-format", "json"])

    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["error"] == {
        "type": "PermissionError",
        "message": "workspace unavailable",
    }


def test_main_treats_closed_stdout_pipe_as_clean_exit(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    paths = _paths(tmp_path)

    async def fail_write(*_args: Any, **_kwargs: Any) -> int:
        raise BrokenPipeError

    silenced: list[Any] = []
    monkeypatch.setattr(
        cli_main_module.KairoPaths,
        "discover",
        classmethod(lambda _cls: paths),
    )
    monkeypatch.setattr(cli_main_module, "noninteractive", fail_write)
    monkeypatch.setattr(cli_main_module, "_silence_broken_pipe", silenced.append)

    with pytest.raises(SystemExit) as raised:
        cli_main_module.main(["-p", "hello"])

    assert raised.value.code == 0
    assert silenced == [cli_main_module.sys.stdout]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_main_treats_closed_help_pipe_as_clean_exit(monkeypatch: Any) -> None:
    class BrokenParser:
        def parse_args(self, _argv: list[str] | None) -> Any:
            raise BrokenPipeError

    silenced: list[Any] = []
    monkeypatch.setattr(cli_main_module, "build_parser", BrokenParser)
    monkeypatch.setattr(cli_main_module, "_silence_broken_pipe", silenced.append)

    with pytest.raises(SystemExit) as raised:
        cli_main_module.main(["--help"])

    assert raised.value.code == 0
    assert silenced == [cli_main_module.sys.stdout]
