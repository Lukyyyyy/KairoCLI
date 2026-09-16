import os
import re
import unicodedata
from collections.abc import Mapping, Sequence
from io import StringIO
from typing import Any, Literal
from urllib.parse import urlsplit

from rich.console import Console
from rich.markdown import Heading, Markdown
from rich.syntax import Syntax

from .terminal import TerminalStreamSanitizer

ColorSystemName = Literal["standard", "256", "truecolor", "windows"]

_MAX_HIGHLIGHT_BYTES = 512 * 1024
_MAX_HIGHLIGHT_LINES = 10_000
_MAX_HIGHLIGHT_LINE_BYTES = 4 * 1024
_REFERENCE_LINK = re.compile(r"\[[^]\n]+]\[[^]\n]*]")
_REFERENCE_DEFINITION = re.compile(r"(?m)^\s{0,3}\[[^]]+]:\s*\S+")
_REFERENCE_DEFINITION_LINE = re.compile(r"(?m)^\s{0,3}\[[^]]+]:[^\n]*(?:\n|$)")
_SAFE_LINK_SCHEMES = {"http", "https", "mailto"}
_TRAILING_SPACES = re.compile(r" +((?:\x1b\[[0-9;]*m)*)$")


class RenderedTerminalText(str):
    """Terminal text sanitized before rendering; ANSI may only be renderer-generated."""


class _TerminalHeading(Heading):
    LEVEL_ALIGN = {tag: "left" for tag in Heading.LEVEL_ALIGN}


class _TerminalMarkdown(Markdown):
    elements = {**Markdown.elements, "heading_open": _TerminalHeading}


class TerminalMarkdownRenderer:
    """Render stable CommonMark blocks while retaining the final mutable stream block."""

    def __init__(
        self,
        columns: int = 120,
        continuation_indent: str = "",
        color_system: ColorSystemName | None = None,
        code_theme: str | None = None,
    ) -> None:
        self.columns = max(40, min(int(columns), 1_000))
        self.continuation_indent = continuation_indent
        self.color_system = color_system
        self.code_theme = code_theme or terminal_code_theme(os.environ)
        self._source = ""
        self._sanitizer = TerminalStreamSanitizer()
        self._content_started = False

    def append(self, chunk: str) -> RenderedTerminalText:
        self._source += self._sanitizer.feed(chunk)
        return self._drain(final=False)

    def finish(self) -> RenderedTerminalText:
        self._source += self._sanitizer.finish()
        return self._drain(final=True)

    @classmethod
    def render(
        cls,
        markdown: str,
        columns: int = 120,
        continuation_indent: str = "",
        color_system: ColorSystemName | None = None,
        code_theme: str | None = None,
    ) -> RenderedTerminalText:
        renderer = cls(columns, continuation_indent, color_system, code_theme)
        return RenderedTerminalText(renderer.append(markdown) + renderer.finish())

    def _drain(self, *, final: bool) -> RenderedTerminalText:
        if not self._source:
            return RenderedTerminalText()
        end = len(self._source) if final else _stable_source_end(self._source)
        if not end:
            return RenderedTerminalText()
        source, self._source = self._source[:end], self._source[end:]
        rendered = _render_source(source, self.columns, self.color_system, self.code_theme)
        return self._indent(rendered)

    def _indent(self, rendered: str) -> RenderedTerminalText:
        output: list[str] = []
        if self._content_started and rendered:
            output.append("\n")
        for line in rendered.splitlines(keepends=True):
            if self._content_started and line not in {"\n", "\r\n"}:
                output.append(self.continuation_indent)
            output.append(line)
            if line not in {"\n", "\r\n"}:
                self._content_started = True
        return RenderedTerminalText("".join(output))


def terminal_code_theme(environment: Mapping[str, str]) -> str:
    """Select an ANSI theme from the terminal's commonly exposed background hint."""
    background = environment.get("COLORFGBG", "").rsplit(";", 1)[-1]
    try:
        return "ansi_light" if int(background) >= 7 else "ansi_dark"
    except ValueError:
        return "ansi_dark"


def _stable_source_end(source: str) -> int:
    if _REFERENCE_LINK.search(source) and not _REFERENCE_DEFINITION.search(source):
        return 0
    markdown = _TerminalMarkdown(source)
    blocks = _top_level_blocks(markdown.parsed)
    if not blocks:
        return 0
    if source.endswith("\n\n"):
        return len(source)
    last = blocks[-1]
    if source.endswith("\n") and _self_contained(last, source):
        return _line_offset(source, last.map[1]) if last.map else 0
    if len(blocks) > 1 and last.map:
        return _line_offset(source, last.map[0])
    return 0


def _top_level_blocks(tokens: Sequence[Any]) -> list[Any]:
    return [token for token in tokens if token.level == 0 and token.map and token.nesting >= 0]


def _self_contained(token: Any, source: str) -> bool:
    if token.type in {"heading_open", "hr"}:
        return True
    if token.type != "fence" or not token.map:
        return False
    lines = source.splitlines()
    start, end = token.map
    if end - start < 2 or end > len(lines):
        return False
    closing = lines[end - 1].lstrip()
    return closing.startswith(token.markup) and len(closing) >= len(token.markup)


def _line_offset(source: str, line_number: int) -> int:
    return sum(len(line) for line in source.splitlines(keepends=True)[:line_number])


def _render_source(
    source: str,
    columns: int,
    color_system: ColorSystemName | None,
    code_theme: str,
) -> str:
    markdown = _TerminalMarkdown(source)
    lines = source.splitlines(keepends=True)
    definitions = "".join(_REFERENCE_DEFINITION_LINE.findall(source))
    rendered: list[str] = []
    for token in _top_level_blocks(markdown.parsed):
        if token.type in {"fence", "code_block"}:
            block = _render_code(token, columns, color_system, code_theme)
        elif token.map:
            block_source = "".join(lines[token.map[0] : token.map[1]])
            if definitions and _REFERENCE_LINK.search(block_source):
                block_source += "\n" + definitions
            block = _render_markdown(block_source, columns, color_system, code_theme)
        else:
            continue
        if block:
            rendered.append(block.strip("\n"))
    return "\n\n".join(rendered) + ("\n" if rendered else "")


def _render_markdown(
    source: str,
    columns: int,
    color_system: ColorSystemName | None,
    code_theme: str,
) -> str:
    markdown = _TerminalMarkdown(
        source,
        code_theme=code_theme,
        hyperlinks=color_system is not None,
    )
    _sanitize_links(markdown.parsed)
    stream = StringIO()
    console = Console(
        file=stream,
        force_terminal=color_system is not None,
        color_system=color_system,
        no_color=color_system is None,
        width=columns,
    )
    console.print(markdown)
    return "".join(_TRAILING_SPACES.sub(r"\1", line) for line in stream.getvalue().splitlines(True))


def _render_code(
    token: Any,
    columns: int,
    color_system: ColorSystemName | None,
    code_theme: str,
) -> str:
    language = re.split(r"[,\s]", token.info.strip(), maxsplit=1)[0]
    code = token.content.rstrip("\n")
    label = f"┌─ code: {language}" if language else "┌─ code"
    if not language or color_system is None or _highlight_too_large(code):
        body = "\n".join("    " + line for line in code.split("\n"))
    else:
        stream = StringIO()
        console = Console(
            file=stream,
            force_terminal=True,
            color_system=color_system,
            no_color=False,
            width=columns,
        )
        indented = "\n".join("    " + line for line in code.split("\n"))
        console.print(
            Syntax(
                indented,
                language,
                theme=code_theme,
                background_color="default",
                word_wrap=False,
                padding=0,
            ),
            soft_wrap=True,
        )
        body = stream.getvalue().rstrip("\n")
    return f"{label}\n{body}\n└─ end"


def _highlight_too_large(code: str) -> bool:
    lines = code.splitlines() or [""]
    return (
        len(code.encode()) > _MAX_HIGHLIGHT_BYTES
        or len(lines) > _MAX_HIGHLIGHT_LINES
        or any(len(line.encode()) > _MAX_HIGHLIGHT_LINE_BYTES for line in lines)
    )


def _sanitize_links(tokens: Sequence[Any]) -> None:
    for token in tokens:
        if token.type == "link_open":
            href = str(token.attrs.get("href", ""))
            if not _safe_link(href):
                token.attrs["href"] = ""
        if token.children:
            _sanitize_links(token.children)


def _safe_link(href: str) -> bool:
    if not href or any(unicodedata.category(character) == "Cc" for character in href):
        return False
    try:
        scheme = urlsplit(href).scheme.casefold()
    except ValueError:
        return False
    return not scheme or scheme in _SAFE_LINK_SCHEMES
