import re

import pytest

from kairocli.cli import _render_interactive_answer
from kairocli.rendering.terminal_markdown import (
    RenderedTerminalText,
    TerminalMarkdownRenderer,
    terminal_code_theme,
)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def test_terminal_markdown_renders_common_blocks() -> None:
    rendered = TerminalMarkdownRenderer.render(
        "# 规划思考\n\n1. **分析请求**\n  - 列出目录\n\n"
        "| 名称 | 说明 |\n| --- | --- |\n| src | 源码 |\n\n"
        "```python\nprint('hello')\n```\n> done"
    )
    assert "规划思考" in rendered
    assert "1 分析请求" in rendered
    assert "• 列出目录" in rendered
    assert "名称" in rendered and "src" in rendered
    assert "┌─ code: python" in rendered
    assert "    print('hello')" in rendered
    assert "└─ end" in rendered
    assert "▌ done" in rendered


def test_terminal_markdown_supports_split_streaming_chunks() -> None:
    renderer = TerminalMarkdownRenderer()
    heading = renderer.append("## 标题\n- 第一")
    held = renderer.append("项\n- 第二项\n")
    tail = renderer.finish()
    assert heading.strip() == "标题"
    assert held == ""
    assert "第一项" in tail and "第二项" in tail


def test_terminal_markdown_highlights_complete_fenced_code_block() -> None:
    renderer = TerminalMarkdownRenderer(color_system="standard")

    rendered = renderer.append("```java\npublic class User {\n")
    assert rendered == ""
    rendered += renderer.append('    String name = "Kairo";\n}\n```\n')
    rendered += renderer.finish()

    assert "\x1b[" in rendered
    plain = _ANSI.sub("", rendered)
    assert "public" in plain
    assert 'String name = "Kairo";' in plain
    assert rendered.endswith("└─ end\n")


def test_terminal_markdown_indents_lines_after_answer_marker() -> None:
    rendered = TerminalMarkdownRenderer.render(
        "第一行\n\n- 第一项\n- 第二项", continuation_indent="  "
    )

    assert rendered.startswith("第一行\n\n")
    assert "第一项" in rendered and "第二项" in rendered


def test_terminal_markdown_indents_terminal_width_wrapped_lines() -> None:
    rendered = TerminalMarkdownRenderer.render("甲" * 25, columns=40, continuation_indent="  ")

    assert rendered == ("甲" * 20) + "\n  " + ("甲" * 5) + "\n"


def test_terminal_markdown_preserves_list_hanging_indent_when_wrapping() -> None:
    rendered = TerminalMarkdownRenderer.render(
        "- " + "x" * 45, columns=40, continuation_indent="  "
    )

    assert "x" * 36 in rendered
    assert rendered.replace("\n", "").replace(" ", "").replace("•", "").count("x") == 45


def test_terminal_markdown_wraps_wide_table_to_terminal_columns() -> None:
    rendered = TerminalMarkdownRenderer.render(
        "| 特性 | Step | Kimi | GLM | DeepSeek |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| URL | https://api.stepfun.com/v1 | https://api.moonshot.ai/v1 | "
        "动态选择多模态接口 | https://api.deepseek.com/chat/completions |",
        columns=72,
    )
    assert "特性" in rendered and "DeepSeek" in rendered
    assert all(len(line) <= 72 for line in rendered.splitlines())


def test_terminal_markdown_sanitizes_controls_before_rendering() -> None:
    rendered = TerminalMarkdownRenderer.render(
        "# safe\x1b]52;c;clipboard\x07\n[shown](https://example.com)"
    )
    assert "clipboard" not in rendered
    assert "\x1b" not in rendered
    assert "shown" in rendered
    assert "shown (https://example.com)" in rendered


def test_plain_renderer_preserves_markdown_while_inline_formats_it() -> None:
    markdown = "# Heading\n**bold**\x1b]0;forged\x07"
    plain = _render_interactive_answer(markdown, "plain", 80)
    inline = _render_interactive_answer(markdown, "inline", 80)
    assert plain == "# Heading\n  **bold**"
    assert inline == "Heading\n\n  bold"


def test_terminal_markdown_preserves_literal_punctuation_and_commonmark_fences() -> None:
    rendered = TerminalMarkdownRenderer.render(
        '字段 `user_name`，文件 foo_bar.py，表达式 a*b\n\n~~~python\nprint("ok")\n~~~'
    )

    assert "user_name" in rendered
    assert "foo_bar.py" in rendered
    assert "a*b" in rendered
    assert 'code: python\n    print("ok")' in rendered


def test_terminal_markdown_never_inserts_newlines_in_code() -> None:
    source_line = 'value = "' + "x" * 120 + '"'
    rendered = TerminalMarkdownRenderer.render(
        f"```python\n{source_line}\n```", columns=40, color_system="standard"
    )

    plain = _ANSI.sub("", rendered)
    assert "    " + source_line in plain.splitlines()


def test_terminal_markdown_highlight_limits_fall_back_to_plain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kairocli.rendering.terminal_markdown as markdown_module

    monkeypatch.setattr(markdown_module, "_MAX_HIGHLIGHT_LINE_BYTES", 4)
    rendered = TerminalMarkdownRenderer.render(
        "```python\nvalue = 1\n```", color_system="standard"
    )

    assert "\x1b[" not in rendered
    assert "    value = 1" in rendered


def test_terminal_markdown_links_are_visible_or_clickable_and_unsafe_links_are_inert() -> None:
    plain = TerminalMarkdownRenderer.render("[OpenAI](https://openai.com)")
    colored = TerminalMarkdownRenderer.render(
        "[OpenAI](https://openai.com)", color_system="standard"
    )
    unsafe = TerminalMarkdownRenderer.render(
        "[bad](javascript:alert(1))", color_system="standard"
    )

    assert plain.strip() == "OpenAI (https://openai.com)"
    assert "\x1b]8;" in colored and "https://openai.com" in colored
    assert "\x1b]8;" not in unsafe


def test_terminal_markdown_holds_reference_links_until_the_definition_arrives() -> None:
    renderer = TerminalMarkdownRenderer()

    assert renderer.append("Read [the docs][guide].\n\n") == ""
    assert renderer.append("[guide]: https://example.com/docs\n") == ""
    rendered = renderer.finish()

    assert "the docs (https://example.com/docs)" in rendered


def test_terminal_markdown_theme_follows_terminal_background_hint() -> None:
    assert terminal_code_theme({"COLORFGBG": "15;0"}) == "ansi_dark"
    assert terminal_code_theme({"COLORFGBG": "0;15"}) == "ansi_light"
    assert terminal_code_theme({}) == "ansi_dark"


def test_rendered_terminal_text_marks_the_trusted_ansi_boundary() -> None:
    rendered = TerminalMarkdownRenderer.render("**safe**", color_system="standard")

    assert isinstance(rendered, RenderedTerminalText)
