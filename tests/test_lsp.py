import asyncio
import io
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import kairocli.lsp as lsp_module
from kairocli.lsp import (
    Diagnostic,
    LspClient,
    LspManager,
    LspServerConfig,
    _content_length,
    _normalize_code_actions,
    format_diagnostics,
)
from kairocli.tools import ToolRegistry


async def test_lsp_close_cleans_background_tasks_after_unexpected_shutdown_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = LspClient(LspServerConfig("python", (), (".py",), "python"), tmp_path)
    process = SimpleNamespace(returncode=None)
    client.process = process  # type: ignore[assignment]
    reader = asyncio.create_task(asyncio.Event().wait())
    stderr = asyncio.create_task(asyncio.Event().wait())
    client._reader_task = reader  # type: ignore[assignment]
    client._stderr_task = stderr  # type: ignore[assignment]
    terminated: list[object] = []

    async def fail_request(*_args: object, **_kwargs: object) -> object:
        raise OSError("shutdown transport failed")

    async def fake_terminate(value: object) -> None:
        terminated.append(value)

    monkeypatch.setattr(client, "_request", fail_request)
    monkeypatch.setattr(lsp_module, "_terminate_lsp_process", fake_terminate)

    await client.close()
    await client.close()

    assert terminated == [process]
    assert reader.cancelled() and stderr.cancelled()
    assert client.process is None
    assert client._reader_task is None
    assert client._stderr_task is None


async def test_lsp_close_fences_a_process_that_finishes_spawning_late(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = LspClient(LspServerConfig("unused", (), (".py",), "python"), tmp_path)
    spawn_entered = asyncio.Event()
    release_spawn = asyncio.Event()
    aborted: list[object] = []
    process = SimpleNamespace(returncode=None)

    async def delayed_spawn(*_args: object, **_kwargs: object) -> object:
        spawn_entered.set()
        await release_spawn.wait()
        return process

    async def abort() -> None:
        aborted.append(client.process)
        client.process = None

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_spawn)
    monkeypatch.setattr(client, "_abort", abort)

    starting = asyncio.create_task(client.start())
    await spawn_entered.wait()
    closing = asyncio.create_task(client.close())
    await asyncio.sleep(0)
    release_spawn.set()

    with pytest.raises(RuntimeError, match="closed"):
        await starting
    await closing
    assert aborted == [process]
    assert client.process is None
    with pytest.raises(RuntimeError, match="closed"):
        await client.start()


async def test_lsp_close_finishes_when_the_calling_task_is_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = LspClient(LspServerConfig("unused", (), (".py",), "python"), tmp_path)
    close_entered = asyncio.Event()
    release_close = asyncio.Event()
    finished = False

    async def delayed_close() -> None:
        nonlocal finished
        close_entered.set()
        await release_close.wait()
        finished = True

    monkeypatch.setattr(client, "_close_process", delayed_close)
    caller = asyncio.create_task(client.close())
    await close_entered.wait()
    caller.cancel()
    await asyncio.sleep(0)
    assert not caller.done()
    release_close.set()

    with pytest.raises(asyncio.CancelledError):
        await caller
    assert finished
    await client.close()


def test_parser_fallback_covers_python_json_toml_and_braces(tmp_path: Path) -> None:
    manager = LspManager(tmp_path, configs=[])
    samples = {
        "broken.py": "def x(:\n",
        "broken.json": '{"x":}',
        "broken.toml": 'name = "unterminated',
        "Broken.java": "class Broken {\n",
        "broken.cpp": "int main() {\n",
    }
    for name, content in samples.items():
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        diagnostics = manager.diagnose_file(path)
        assert diagnostics and diagnostics[0].severity == "error"
        assert diagnostics[0].path == name


async def test_all_local_parser_entrypoints_share_bounded_growth_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "growing.py"
    path.write_text("x", encoding="utf-8")
    requested: list[int] = []

    class GrowingReader(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            requested.append(size)
            return super().read(size)

    def growing_open(
        _path: Path, mode: str = "r", *args: object, **kwargs: object
    ) -> GrowingReader:
        assert mode == "rb"
        return GrowingReader(b"x" * 9)

    monkeypatch.setattr(lsp_module, "MAX_LSP_PARSER_FILE_BYTES", 8)
    monkeypatch.setattr(Path, "open", growing_open)
    manager = LspManager(tmp_path, configs=[])

    assert manager.diagnose_file(path) == []
    assert await manager.diagnose_file_async(path) == []
    assert await manager.inspect_file_async(path) == {
        "path": "growing.py",
        "diagnostics": [],
        "code_actions": [],
        "server": None,
    }
    workspace = await manager.workspace_diagnostics_async(tmp_path)

    assert workspace["scanned_files"] == 1
    assert workspace["diagnostics"] == []
    assert requested == [9, 9, 9, 9]


def test_local_parser_rejects_binary_source_without_caching(tmp_path: Path) -> None:
    path = tmp_path / "binary.py"
    path.write_bytes(b"def x():\0broken")
    manager = LspManager(tmp_path, configs=[])
    manager.diagnostics[str(path.resolve())] = [Diagnostic("binary.py", 1, 1, "error", "stale")]

    assert manager.diagnose_file(path) == []
    assert str(path.resolve()) not in manager.diagnostics


def test_diagnostic_formatting_reports_omitted_count() -> None:
    diagnostics = [
        Diagnostic("a.py", index, 1, "warning", f"issue {index}", "test") for index in range(1, 4)
    ]
    rendered = format_diagnostics(diagnostics, max_items=2)
    assert rendered.count("issue") == 2
    assert "1 additional diagnostics omitted" in rendered


def test_code_action_summaries_are_bounded_and_drop_edit_payloads() -> None:
    actions = _normalize_code_actions(
        [
            {
                "title": "x" * 2_000,
                "kind": "quickfix",
                "edit": {"changes": {"file:///x": [{"newText": "private"}]}},
                "command": {"command": "apply", "arguments": ["secret"]},
            }
        ]
        * 60
    )

    assert len(actions) == 50
    assert len(actions[0]["title"]) == 1_000
    assert actions[0]["has_edit"] is True
    assert actions[0]["command"] == "apply"
    assert "private" not in json.dumps(actions)
    assert "secret" not in json.dumps(actions)


async def test_lsp_inspect_has_parser_fallback_without_a_server(tmp_path: Path) -> None:
    path = tmp_path / "broken.py"
    path.write_text("def broken(:\n", encoding="utf-8")
    registry = ToolRegistry(tmp_path)
    await registry.lsp.close()
    registry.lsp = LspManager(tmp_path, configs=[])

    result = json.loads(await registry.execute("lsp_inspect", {"path": "broken.py"}))

    assert result["path"] == "broken.py"
    assert result["diagnostics"][0]["severity"] == "error"
    assert result["code_actions"] == []
    assert result["server"] is None
    schema = next(item for item in registry.schemas() if item["function"]["name"] == "lsp_inspect")
    assert schema["function"]["parameters"]["required"] == ["path"]
    outside = tmp_path.parent / "outside.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    escaped = json.loads(await registry.execute("lsp_inspect", {"path": "../outside.py"}))
    assert escaped["policy_denied"] is True
    await registry.close()


async def test_workspace_diagnostic_parser_scan_is_bounded_and_excludes_dependencies(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.py").write_text("def a(:\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def b(:\n", encoding="utf-8")
    dependency = tmp_path / "node_modules"
    dependency.mkdir()
    (dependency / "ignored.py").write_text("def ignored(:\n", encoding="utf-8")
    registry = ToolRegistry(tmp_path)
    await registry.lsp.close()
    registry.lsp = LspManager(tmp_path, configs=[])

    result = json.loads(
        await registry.execute(
            "lsp_workspace_diagnostics",
            {"max_files": 1, "max_diagnostics": 10},
        )
    )

    assert result["root"] == "."
    assert result["scanned_files"] == 1
    assert result["partial"] is True
    assert [item["path"] for item in result["diagnostics"]] == ["a.py"]
    assert "ignored.py" not in json.dumps(result)
    schema = next(
        item
        for item in registry.schemas()
        if item["function"]["name"] == "lsp_workspace_diagnostics"
    )
    assert schema["function"]["parameters"]["properties"]["max_files"]["maximum"] == 500
    await registry.close()


async def test_lsp_timeout_does_not_reuse_stale_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "demo.py"
    path.write_text("value = 1\n", encoding="utf-8")
    client = LspClient(
        LspServerConfig("unused", (), (".py",), "python"),
        tmp_path,
        diagnostic_timeout=0.1,
    )
    uri = path.resolve().as_uri()
    client._diagnostics[uri] = [Diagnostic("demo.py", 1, 1, "error", "stale")]

    async def no_start() -> None:
        return None

    async def no_notification(method: str, params: object) -> None:
        return None

    monkeypatch.setattr(client, "start", no_start)
    monkeypatch.setattr(client, "_notify", no_notification)

    assert await client.diagnose(path, "value = 2\n") == []
    assert client._diagnostics[uri] == []


async def test_unsupported_code_actions_preserve_published_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "demo.py"
    client = LspClient(
        LspServerConfig("unused", (), (".py",), "python"),
        tmp_path,
        diagnostic_timeout=1,
    )

    async def no_start() -> None:
        return None

    async def publish(method: str, params: Any) -> None:
        document = params["textDocument"]
        client._dispatch(
            {
                "jsonrpc": "2.0",
                "method": "textDocument/publishDiagnostics",
                "params": {
                    "uri": document["uri"],
                    "version": document["version"],
                    "diagnostics": [
                        {
                            "range": {"start": {"line": 0, "character": 0}},
                            "severity": 1,
                            "message": "keep me",
                        }
                    ],
                },
            }
        )

    async def unsupported(method: str, params: Any) -> Any:
        raise RuntimeError("method not found")

    monkeypatch.setattr(client, "start", no_start)
    monkeypatch.setattr(client, "_notify", publish)
    monkeypatch.setattr(client, "_request", unsupported)

    diagnostics, actions = await client.diagnose_with_code_actions(path, "x = 1\n")

    assert diagnostics[0].message == "keep me"
    assert actions == []


async def test_same_document_diagnostics_are_serialized_by_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "demo.py"
    client = LspClient(
        LspServerConfig("unused", (), (".py",), "python"),
        tmp_path,
        diagnostic_timeout=1,
    )
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    notifications: list[tuple[str, int]] = []
    active = 0
    max_active = 0

    async def no_start() -> None:
        return None

    async def publish(method: str, params: Any) -> None:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        document = params["textDocument"]
        version = int(document["version"])
        notifications.append((method, version))
        if version == 1:
            first_entered.set()
            await release_first.wait()
        client._dispatch(
            {
                "jsonrpc": "2.0",
                "method": "textDocument/publishDiagnostics",
                "params": {
                    "uri": document["uri"],
                    "version": version,
                    "diagnostics": [
                        {
                            "range": {"start": {"line": version - 1, "character": 0}},
                            "severity": 2,
                            "message": f"version {version}",
                        }
                    ],
                },
            }
        )
        active -= 1

    monkeypatch.setattr(client, "start", no_start)
    monkeypatch.setattr(client, "_notify", publish)
    first = asyncio.create_task(client.diagnose(path, "value = 1\n"))
    await first_entered.wait()
    second = asyncio.create_task(client.diagnose(path, "value = 2\n"))
    await asyncio.sleep(0)
    assert notifications == [("textDocument/didOpen", 1)]
    release_first.set()

    first_result, second_result = await asyncio.gather(first, second)
    assert first_result[0].message == "version 1"
    assert second_result[0].message == "version 2"
    assert notifications == [
        ("textDocument/didOpen", 1),
        ("textDocument/didChange", 2),
    ]
    assert max_active == 1


async def test_dead_lsp_process_is_reset_before_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = LspClient(LspServerConfig("unused", (), (".py",), "python"), tmp_path)
    client.process = SimpleNamespace(returncode=1)  # type: ignore[assignment]
    client._versions["file:///stale.py"] = 9
    reset = False

    async def abort() -> None:
        nonlocal reset
        reset = True
        client.process = None
        client._reset_document_state()

    async def fail_spawn(*args: object, **kwargs: object) -> None:
        assert reset
        assert client._versions == {}
        raise RuntimeError("spawn sentinel")

    monkeypatch.setattr(client, "_abort", abort)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_spawn)

    with pytest.raises(RuntimeError, match="spawn sentinel"):
        await client.start()
    assert reset


async def test_live_process_with_dead_lsp_reader_is_reset_before_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = LspClient(LspServerConfig("unused", (), (".py",), "python"), tmp_path)
    client.process = SimpleNamespace(returncode=None)  # type: ignore[assignment]
    client._reader_task = asyncio.create_task(asyncio.sleep(0))
    await client._reader_task
    reset = False

    async def abort() -> None:
        nonlocal reset
        reset = True
        client.process = None
        client._reader_task = None
        client._reset_document_state()

    async def fail_spawn(*args: object, **kwargs: object) -> None:
        assert reset
        raise RuntimeError("spawn sentinel")

    monkeypatch.setattr(client, "_abort", abort)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_spawn)

    with pytest.raises(RuntimeError, match="spawn sentinel"):
        await client.start()
    assert reset


async def test_lsp_reader_skips_invalid_json_frames_and_keeps_correlation(
    tmp_path: Path,
) -> None:
    client = LspClient(LspServerConfig("unused", (), (".py",), "python"), tmp_path)
    reader = asyncio.StreamReader()
    client.process = SimpleNamespace(stdout=reader)  # type: ignore[assignment]
    future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
    client._pending[1] = future

    invalid_bodies = [
        b'{"jsonrpc":"2.0","id":1,"result":{},"result":{"bad":true}}',
        b'{"jsonrpc":"2.0","id":1,"result":{"value":NaN}}',
        ('{"jsonrpc":"2.0","id":1,"result":{"value":' + "[" * 40 + "0" + "]" * 40 + "}}").encode(),
        b'{"jsonrpc":"1.0","id":1,"result":{"bad":true}}',
    ]
    valid = b'{"jsonrpc":"2.0","id":1,"result":{"ok":true}}'
    for body in [*invalid_bodies, valid]:
        reader.feed_data(f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
    reader.feed_eof()

    await client._reader_loop()

    assert future.result() == {"ok": True}


async def test_lsp_outgoing_nonfinite_payload_fails_before_write(tmp_path: Path) -> None:
    class FakeStdin:
        writes: list[bytes] = []

        def write(self, value: bytes) -> None:
            self.writes.append(value)

        async def drain(self) -> None:
            return None

    stdin = FakeStdin()
    client = LspClient(LspServerConfig("unused", (), (".py",), "python"), tmp_path)
    client.process = SimpleNamespace(returncode=None, stdin=stdin)  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="valid finite JSON"):
        await client._send({"value": float("nan")})

    assert stdin.writes == []


@pytest.mark.parametrize(
    "header",
    [
        b"Content-Length: +1\r\n\r\n",
        b"Content-Length: -1\r\n\r\n",
        b"Content-Length: 1\r\nContent-Length: 2\r\n\r\n",
        b"Content-Length: \xff\r\n\r\n",
    ],
)
def test_lsp_content_length_rejects_ambiguous_or_non_ascii_values(
    header: bytes,
) -> None:
    assert _content_length(header) == -1


def test_lsp_ignores_unopened_and_external_diagnostic_publications(
    tmp_path: Path,
) -> None:
    client = LspClient(LspServerConfig("unused", (), (".py",), "python"), tmp_path)
    unknown = (tmp_path / "unknown.py").as_uri()
    external = (tmp_path.parent / "external.py").as_uri()
    payload = {
        "jsonrpc": "2.0",
        "method": "textDocument/publishDiagnostics",
        "params": {"uri": unknown, "diagnostics": []},
    }

    client._dispatch(payload)
    payload["params"]["uri"] = external  # type: ignore[index]
    client._dispatch(payload)

    assert client._events == {}
    assert client._diagnostics == {}


async def test_duplicate_lsp_response_does_not_kill_reader_state(tmp_path: Path) -> None:
    client = LspClient(LspServerConfig("unused", (), (".py",), "python"), tmp_path)
    future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
    future.set_result({"ok": True})
    client._pending[1] = future

    client._dispatch({"jsonrpc": "2.0", "id": 1, "result": {"duplicate": True}})
    client._dispatch({"jsonrpc": "2.0", "id": True, "result": {"wrong": True}})

    assert future.result() == {"ok": True}

    wrong_version: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
    client._pending[2] = wrong_version
    client._dispatch({"jsonrpc": "1.0", "id": 2, "result": {"wrong": True}})
    assert not wrong_version.done()

    ambiguous: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
    client._pending[3] = ambiguous
    client._dispatch({"jsonrpc": "2.0", "id": 3, "result": {}, "error": {}})
    assert isinstance(ambiguous.exception(), RuntimeError)
    assert "exactly one" in str(ambiguous.exception())

    malformed_error: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
    client._pending[4] = malformed_error
    client._dispatch({"jsonrpc": "2.0", "id": 4, "error": "broken"})
    assert isinstance(malformed_error.exception(), RuntimeError)
    assert "invalid" in str(malformed_error.exception())


def test_explicit_lsp_commands_cover_all_supported_language_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAIROCLI_LSP_AUTO", "false")
    commands = {
        "KAIROCLI_LSP_PYTHON_COMMAND": 'python-lsp "--flag value"',
        "KAIROCLI_LSP_TYPESCRIPT_COMMAND": "ts-lsp --stdio",
        "KAIROCLI_LSP_GO_COMMAND": "go-lsp",
        "KAIROCLI_LSP_RUST_COMMAND": "rust-lsp",
        "KAIROCLI_LSP_JAVA_COMMAND": "java-lsp",
        "KAIROCLI_LSP_CLANG_COMMAND": "clang-lsp --background-index",
    }
    for name, value in commands.items():
        monkeypatch.setenv(name, value)

    configs = lsp_module._discover_configs()

    assert {config.language_id for config in configs} == {
        "python",
        "typescript",
        "go",
        "rust",
        "java",
        "cpp",
    }
    python = next(config for config in configs if config.language_id == "python")
    assert python.command == "python-lsp"
    assert python.args == ("--flag value",)


def test_lsp_auto_discovery_includes_java_and_clang_without_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "KAIROCLI_LSP_PYTHON_COMMAND",
        "KAIROCLI_LSP_TYPESCRIPT_COMMAND",
        "KAIROCLI_LSP_GO_COMMAND",
        "KAIROCLI_LSP_RUST_COMMAND",
        "KAIROCLI_LSP_JAVA_COMMAND",
        "KAIROCLI_LSP_CLANG_COMMAND",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KAIROCLI_LSP_AUTO", "true")
    available = {"gopls", "rust-analyzer", "jdtls", "clangd"}
    monkeypatch.setattr(
        lsp_module.shutil,
        "which",
        lambda command: f"/tools/{command}" if command in available else None,
    )

    configs = lsp_module._discover_configs()

    assert [config.language_id for config in configs] == [
        "go",
        "rust",
        "java",
        "cpp",
    ]


@pytest.mark.parametrize(
    "value",
    ['server "unterminated', "server " + "x " * 65, "x" * 8_193],
)
def test_invalid_lsp_command_configuration_is_rejected(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("KAIROCLI_LSP_AUTO", "false")
    monkeypatch.setenv("KAIROCLI_LSP_GO_COMMAND", value)

    with pytest.raises(ValueError, match="KAIROCLI_LSP_GO_COMMAND"):
        lsp_module._discover_configs()


def test_lsp_discards_old_versions_and_bounds_untrusted_diagnostics(
    tmp_path: Path,
) -> None:
    client = LspClient(LspServerConfig("unused", (), (".py",), "python"), tmp_path)
    uri = (tmp_path / "demo.py").resolve().as_uri()
    client._versions[uri] = 2
    event = client._events.setdefault(uri, asyncio.Event())
    diagnostic = {
        "range": {"start": {"line": "bad", "character": -5}},
        "severity": True,
        "message": "x" * 5_000,
        "source": "server" * 100,
    }

    client._dispatch(
        {
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {"uri": uri, "version": 1, "diagnostics": [diagnostic]},
        }
    )
    assert not event.is_set()
    assert uri not in client._diagnostics

    client._dispatch(
        {
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {
                "uri": uri,
                "version": 2,
                "diagnostics": [diagnostic] * 600,
            },
        }
    )
    assert event.is_set()
    assert len(client._diagnostics[uri]) == 500
    first = client._diagnostics[uri][0]
    assert first.line == 1 and first.column == 1
    assert first.severity == "warning"
    assert len(first.message) == 4_000
    assert len(first.source) == 200


async def test_stdio_lsp_initialize_change_publish_and_shutdown(tmp_path: Path) -> None:
    server = tmp_path / "fake_lsp.py"
    log = tmp_path / "methods.jsonl"
    server.write_text(
        """
import json
import sys

log_path = sys.argv[1]

def read_message():
    length = None
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in {b'\\r\\n', b'\\n'}:
            break
        name, _, value = line.decode().partition(':')
        if name.lower() == 'content-length':
            length = int(value.strip())
    return json.loads(sys.stdin.buffer.read(length)) if length is not None else None

def send(payload):
    body = json.dumps(payload, separators=(',', ':')).encode()
    sys.stdout.buffer.write(f'Content-Length: {len(body)}\\r\\n\\r\\n'.encode() + body)
    sys.stdout.buffer.flush()

document_uri = None
while True:
    message = read_message()
    if message is None:
        break
    method = message.get('method', '')
    with open(log_path, 'a', encoding='utf-8') as stream:
        stream.write(json.dumps({'method': method}) + '\\n')
    if 'id' in message:
        if method == 'initialize':
            result = {'capabilities': {'textDocumentSync': 1, 'codeActionProvider': True}}
        elif method == 'textDocument/codeAction':
            result = [
                {
                    'title': 'Apply safe import',
                    'kind': 'quickfix',
                    'isPreferred': True,
                    'edit': {'changes': {'file:///demo.py': [{'newText': 'private-edit'}]}},
                    'command': {'command': 'server.apply', 'arguments': ['private-argument']}
                },
                {'title': 'Unavailable fix', 'disabled': {'reason': 'not applicable'}},
                {'title': ''},
            ]
        elif method == 'workspace/diagnostic':
            result = {
                'items': [
                    {
                        'uri': document_uri,
                        'kind': 'full',
                        'items': [{
                            'range': {'start': {'line': 0, 'character': 0}},
                            'severity': 2,
                            'message': 'workspace warning',
                            'source': 'fake-workspace'
                        }]
                    },
                    {
                        'uri': 'file:///outside.py',
                        'kind': 'full',
                        'items': [{'message': 'must be filtered'}]
                    }
                ]
            }
        else:
            result = None
        send({'jsonrpc': '2.0', 'id': message['id'], 'result': result})
    if method in {'textDocument/didOpen', 'textDocument/didChange'}:
        document = message['params'].get('textDocument', {})
        document_uri = document['uri']
        send({
            'jsonrpc': '2.0',
            'method': 'textDocument/publishDiagnostics',
            'params': {
                'uri': document['uri'],
                'diagnostics': [{
                    'range': {
                        'start': {'line': 0, 'character': 0},
                        'end': {'line': 0, 'character': 1}
                    },
                    'severity': 2,
                    'message': 'fake warning',
                    'source': 'fake-lsp'
                }]
            }
        })
    if method == 'exit':
        break
""",
        encoding="utf-8",
    )
    config = LspServerConfig(sys.executable, (str(server), str(log)), (".py",), "python")
    registry = ToolRegistry(tmp_path)
    registry.lsp = LspManager(tmp_path, configs=[config], diagnostic_timeout=1)
    first = json.loads(
        await registry.execute("write_file", {"path": "demo.py", "content": "value = 1\n"})
    )
    assert first["diagnostics"] == [
        {
            "line": 1,
            "column": 1,
            "severity": "warning",
            "message": "fake warning",
            "source": "fake-lsp",
        }
    ]
    await registry.execute("write_file", {"path": "demo.py", "content": "value = 2\n"})
    inspection = json.loads(
        await registry.execute("lsp_inspect", {"path": "demo.py", "start_line": 1, "end_line": 1})
    )
    assert inspection["diagnostics"][0]["message"] == "fake warning"
    assert inspection["code_actions"] == [
        {
            "title": "Apply safe import",
            "kind": "quickfix",
            "preferred": True,
            "disabled_reason": "",
            "has_edit": True,
            "command": "server.apply",
        },
        {
            "title": "Unavailable fix",
            "kind": "",
            "preferred": False,
            "disabled_reason": "not applicable",
            "has_edit": False,
            "command": "",
        },
    ]
    assert "private-edit" not in json.dumps(inspection)
    assert "private-argument" not in json.dumps(inspection)
    workspace = json.loads(
        await registry.execute(
            "lsp_workspace_diagnostics",
            {"path": ".", "max_files": 10, "max_diagnostics": 10},
        )
    )
    assert workspace["servers_queried"] == 1
    assert [item["message"] for item in workspace["diagnostics"]] == ["workspace warning"]
    assert "must be filtered" not in json.dumps(workspace)
    await registry.close()
    methods = [json.loads(line)["method"] for line in log.read_text().splitlines()]
    assert methods[:3] == ["initialize", "initialized", "textDocument/didOpen"]
    assert "textDocument/didChange" in methods
    assert "textDocument/codeAction" in methods
    assert "workspace/diagnostic" in methods
    assert methods[-2:] == ["shutdown", "exit"]


@pytest.mark.skipif(shutil.which("clangd") is None, reason="clangd is not installed")
async def test_real_clangd_initialize_change_diagnose_and_shutdown(tmp_path: Path) -> None:
    clangd = shutil.which("clangd")
    assert clangd is not None
    source = "int main( { return 0; }\n"
    path = tmp_path / "broken.c"
    path.write_text(source, encoding="utf-8")
    client = LspClient(
        LspServerConfig(clangd, ("--log=error",), (".c",), "c"),
        tmp_path,
        diagnostic_timeout=5,
    )

    try:
        diagnostics = await client.diagnose(path, source)
        assert diagnostics
        assert any(item.severity == "error" for item in diagnostics)
        assert all(item.path == "broken.c" for item in diagnostics)
        assert all(item.source.casefold() == "clang" for item in diagnostics)
        fixed = "int main(void) { return 0; }\n"
        path.write_text(fixed, encoding="utf-8")
        assert await client.diagnose(path, fixed) == []
    finally:
        await client.close()

    assert client.process is None
