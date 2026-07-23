from kairocli.models import ToolCall, ToolOutput
from kairocli.tool_display import format_tool_calls, format_tool_results


def _call(name: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(f"{name}-id", name, arguments)


def test_tool_display_compacts_single_calls_and_mcp_names() -> None:
    assert format_tool_calls([_call("read_file", {"path": "README.md"})]) == (
        "⏵ 📖 Read file(README.md)"
    )
    assert format_tool_calls([_call("web_fetch", {"url": "https://example.com/a/"})]) == (
        "⏵ 📰 Web fetch(example.com/a)"
    )
    assert format_tool_calls([_call("mcp__chrome__click", {"ref": "x"})]).startswith(
        "⏵ 🔌 MCP chrome.click"
    )


def test_tool_display_groups_plain_calls_and_redacts_details() -> None:
    rendered = format_tool_calls(
        [
            _call("execute_command", {"command": "curl -H 'Authorization: Bearer secret'"}),
            _call("execute_command", {"command": "echo ok"}),
        ],
        compact=False,
    )
    assert "Persistent" not in rendered
    assert "Shell × 2" in rendered
    assert "secret" not in rendered
    assert "echo ok" in rendered


def test_tool_display_bounds_details_and_summarizes_multiple_groups() -> None:
    assert (
        format_tool_calls([_call("read_file", {"path": "a"}), _call("write_file", {"path": "b"})])
        == "⏵ 2 tool groups / 2 calls"
    )
    long_path = "/".join(f"directory-{index}" for index in range(100))
    detail = format_tool_calls([_call("read_file", {"path": long_path})])
    assert len(detail.split("(", 1)[1].removesuffix(")")) == 80


def test_tool_display_is_total_for_recursive_nonfinite_and_invalid_unicode() -> None:
    recursive: dict[str, object] = {}
    recursive["self"] = recursive

    assert "<invalid arguments>" in format_tool_calls([_call("mcp__test__recursive", recursive)])
    assert "<invalid arguments>" in format_tool_calls(
        [_call("mcp__test__nonfinite", {"score": float("nan")})]
    )
    rendered = format_tool_calls([_call("read_file", {"path": "bad\ud800\x1b[31m.txt"})])
    assert "\ud800" not in rendered
    assert "\x1b" not in rendered
    assert "bad?.txt" in rendered


def test_tool_result_summary_uses_metadata_without_leaking_body() -> None:
    calls = [_call("read_file", {"path": "a"}), _call("web_fetch", {"url": "x"})]
    rendered = format_tool_results(
        calls,
        [
            ToolOutput("private file body", truncated=True, elapsed_ms=120),
            ToolOutput(
                '{"error":"private upstream body"}',
                timed_out=True,
                elapsed_ms=1_250,
                image_urls=("data:image/png;base64,x",),
            ),
        ],
    )
    assert rendered == ("⚠ 2 tool(s) · 1 failed · 1 timed out · 1 truncated · 1 image(s) · 1.2 s")
    assert "private" not in rendered


def test_tool_result_summary_uses_shared_strict_failure_contract() -> None:
    calls = [_call("mcp__test__run", {}) for _ in range(5)]
    rendered = format_tool_results(
        calls,
        [
            ToolOutput('{"error":"failed","error":""}'),
            ToolOutput('{"score":NaN}'),
            ToolOutput('{"isError":true}'),
            ToolOutput('{"error":null,"result":"ok"}'),
            ToolOutput("plain successful text"),
        ],
    )

    assert rendered == "⚠ 5 tool(s) · 3 failed · 0 ms"
