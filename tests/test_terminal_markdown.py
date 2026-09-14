from kairocli.cli import _render_interactive_answer
from kairocli.rendering.terminal_markdown import TerminalMarkdownRenderer


def test_terminal_markdown_renders_common_blocks() -> None:
    rendered = TerminalMarkdownRenderer.render(
        "# 规划思考\n\n1. **分析请求**\n  - 列出目录\n\n"
        "| 名称 | 说明 |\n| --- | --- |\n| src | 源码 |\n\n"
        "```python\nprint('hello')\n```\n> done"
    )
    assert "规划思考\n========" in rendered
    assert "1. 分析请求" in rendered
    assert "  - 列出目录" in rendered
    assert "| 名称" in rendered and "| src" in rendered
    assert "┌─ code: python" in rendered
    assert "    print('hello')" in rendered
    assert "└─ end" in rendered
    assert "│ done" in rendered


def test_terminal_markdown_supports_split_streaming_chunks() -> None:
    renderer = TerminalMarkdownRenderer()
    rendered = renderer.append("## 标题\n- 第一")
    rendered += renderer.append("项\n- 第二项\n")
    rendered += renderer.finish()
    assert "标题" in rendered
    assert "- 第一项" in rendered
    assert "- 第二项" in rendered


def test_terminal_markdown_indents_lines_after_answer_marker() -> None:
    rendered = TerminalMarkdownRenderer.render(
        "第一行\n\n- 第一项\n- 第二项", continuation_indent="  "
    )

    assert rendered == "第一行\n\n  - 第一项\n  - 第二项\n"


def test_terminal_markdown_indents_terminal_width_wrapped_lines() -> None:
    rendered = TerminalMarkdownRenderer.render("甲" * 25, columns=40, continuation_indent="  ")

    assert rendered == ("甲" * 19) + "\n  " + ("甲" * 6) + "\n"


def test_terminal_markdown_preserves_list_hanging_indent_when_wrapping() -> None:
    rendered = TerminalMarkdownRenderer.render(
        "- " + "x" * 45, columns=40, continuation_indent="  "
    )

    assert rendered == "- " + "x" * 36 + "\n    " + "x" * 9 + "\n"


def test_terminal_markdown_wraps_wide_table_to_terminal_columns() -> None:
    rendered = TerminalMarkdownRenderer.render(
        "| 特性 | Step | Kimi | GLM | DeepSeek |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| URL | https://api.stepfun.com/v1 | https://api.moonshot.ai/v1 | "
        "动态选择多模态接口 | https://api.deepseek.com/chat/completions |",
        columns=72,
    )
    assert "| 特性" in rendered
    assert all(len(line) <= 72 for line in rendered.splitlines())


def test_terminal_markdown_sanitizes_controls_before_rendering() -> None:
    rendered = TerminalMarkdownRenderer.render(
        "# safe\x1b]52;c;clipboard\x07\n[shown](https://example.com)"
    )
    assert "clipboard" not in rendered
    assert "\x1b" not in rendered
    assert "shown" in rendered
    assert "https://" not in rendered


def test_plain_renderer_preserves_markdown_while_inline_formats_it() -> None:
    markdown = "# Heading\n**bold**\x1b]0;forged\x07"
    plain = _render_interactive_answer(markdown, "plain", 80)
    inline = _render_interactive_answer(markdown, "inline", 80)
    assert plain == "# Heading\n  **bold**"
    assert inline == "Heading\n  =======\n\n  bold"
