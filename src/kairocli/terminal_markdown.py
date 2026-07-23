import re
import unicodedata

from .terminal import TerminalStreamSanitizer

_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")
_ORDERED = re.compile(r"^(\s*)(\d+)\.\s+(.*)$")
_UNORDERED = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$")
_LINK = re.compile(r"\[([^]]+)]\([^)]+\)")


class TerminalMarkdownRenderer:
    """Incrementally render common Markdown blocks as bounded terminal text."""

    def __init__(self, columns: int = 120, continuation_indent: str = "") -> None:
        self.columns = max(40, min(int(columns), 1_000))
        self.continuation_indent = continuation_indent
        self._pending = ""
        self._sanitizer = TerminalStreamSanitizer()
        self._table: list[str] = []
        self._in_code = False
        self._last_blank = True
        self._content_started = False

    def append(self, chunk: str) -> str:
        self._pending += self._sanitizer.feed(chunk)
        output: list[str] = []
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            self._process_line(line, output)
        return "".join(output)

    def finish(self) -> str:
        output: list[str] = []
        self._pending += self._sanitizer.finish()
        if self._pending:
            self._process_line(self._pending, output)
            self._pending = ""
        self._flush_table(output)
        return "".join(output)

    @classmethod
    def render(
        cls,
        markdown: str,
        columns: int = 120,
        continuation_indent: str = "",
    ) -> str:
        renderer = cls(columns, continuation_indent)
        return renderer.append(markdown) + renderer.finish()

    def _process_line(self, line: str, output: list[str]) -> None:
        stripped = line.strip()
        if stripped.startswith("```"):
            self._flush_table(output)
            if self._in_code:
                self._write("└─ end", output)
                self._blank(output)
            else:
                language = stripped[3:].strip()
                self._blank(output)
                self._write(f"┌─ {'code: ' + language if language else 'code'}", output)
            self._in_code = not self._in_code
            return
        if self._in_code:
            self._write("    " + line, output)
            return
        if _looks_like_table(line):
            self._table.append(line)
            return
        self._flush_table(output)
        if not stripped:
            self._blank(output)
            return
        heading = _HEADING.match(line)
        if heading:
            content = _inline(heading.group(2).strip())
            self._blank(output)
            self._write(content, output)
            underline = "=" if len(heading.group(1)) == 1 else "-"
            self._write(underline * max(_display_width(content), 4), output)
            self._blank(output)
            return
        ordered = _ORDERED.match(line)
        if ordered:
            indent = "  " * _indent_level(ordered.group(1))
            self._write(f"{indent}{ordered.group(2)}. {_inline(ordered.group(3))}", output)
            return
        unordered = _UNORDERED.match(line)
        if unordered:
            indent = "  " * _indent_level(unordered.group(1))
            self._write(f"{indent}- {_inline(unordered.group(2))}", output)
            return
        if stripped.startswith(">"):
            self._write("│ " + _inline(stripped[1:].strip()), output)
            return
        self._write(_inline(line), output)

    def _flush_table(self, output: list[str]) -> None:
        if not self._table:
            return
        rows = [_parse_row(line) for line in self._table if not _TABLE_SEPARATOR.fullmatch(line)]
        self._table.clear()
        rows = [row for row in rows if row]
        if not rows:
            return
        columns = max(len(row) for row in rows)
        normalized = [row + [""] * (columns - len(row)) for row in rows]
        if len(normalized) >= 2 and columns == 2 and _prefer_key_value(normalized):
            self._render_key_value(normalized, output)
            return
        content_columns = max(40, self.columns - _display_width(self.continuation_indent))
        widths = _allocate_widths(normalized, content_columns)
        border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
        self._blank(output)
        self._write(border, output)
        for row_index, row in enumerate(normalized):
            wrapped = [_wrap_cell(_inline(cell), widths[i]) for i, cell in enumerate(row)]
            for line_index in range(max(len(cell) for cell in wrapped)):
                cells = [cell[line_index] if line_index < len(cell) else "" for cell in wrapped]
                body = "|" + "".join(
                    f" {_pad_display(cell, widths[i])} |" for i, cell in enumerate(cells)
                )
                self._write(body, output)
            if row_index == 0 and len(normalized) > 1:
                self._write(border, output)
        self._write(border, output)
        self._blank(output)

    def _render_key_value(self, rows: list[list[str]], output: list[str]) -> None:
        header = f"{_inline(rows[0][0])} / {_inline(rows[0][1])}"
        self._blank(output)
        self._write(header, output)
        self._write("-" * max(_display_width(header), 8), output)
        for index, row in enumerate(rows[1:]):
            self._write("- " + _inline(row[0]), output)
            if row[1].strip():
                self._write("  " + _inline(row[1]), output)
            if index < len(rows) - 2:
                self._blank(output)
        self._blank(output)

    def _write(self, line: str, output: list[str]) -> None:
        content_width = max(1, self.columns - _display_width(self.continuation_indent))
        for segment in _wrap_display_line(line, content_width):
            prefix = self.continuation_indent if self._content_started else ""
            output.append(prefix + segment.rstrip() + "\n")
            self._content_started = True
        self._last_blank = not line.strip()

    def _blank(self, output: list[str]) -> None:
        if not self._last_blank:
            output.append("\n")
            self._last_blank = True


def _inline(value: str) -> str:
    text = value.replace("**", "").replace("__", "").replace("~~", "")
    text = text.replace("`", "").replace("*", "").replace("_", "")
    return _LINK.sub(r"\1", text).rstrip()


def _looks_like_table(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped and (_TABLE_SEPARATOR.fullmatch(stripped) or stripped.count("|") >= 2))


def _parse_row(line: str) -> list[str]:
    value = line.strip().removeprefix("|").removesuffix("|")
    return [part.strip() for part in value.split("|")]


def _indent_level(value: str) -> int:
    return sum(4 if character == "\t" else 1 for character in value) // 2


def _display_width(value: str) -> int:
    width = 0
    for character in value:
        if unicodedata.category(character) in {"Mn", "Me", "Mc", "Cc", "Cf"}:
            continue
        width += 2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
    return width


def _wrap_display_line(value: str, width: int) -> list[str]:
    """Wrap before the terminal does, preserving a list item's hanging indent."""
    if _display_width(value) <= width:
        return [value.rstrip()]
    marker = re.match(r"^(\s*(?:-\s+|\d+\.\s+))", value)
    hanging = " " * _display_width(marker.group(1)) if marker else ""
    lines: list[str] = []
    current = ""
    current_width = 0
    for character in value:
        character_width = _display_width(character)
        if current and current_width + character_width > width:
            lines.append(current.rstrip())
            current = hanging
            current_width = _display_width(hanging)
            if character.isspace():
                continue
        current += character
        current_width += character_width
    if current or not lines:
        lines.append(current.rstrip())
    return lines


def _prefer_key_value(rows: list[list[str]]) -> bool:
    return any(
        max(_display_width(_inline(row[0])), _display_width(_inline(row[1]))) > 24
        or _display_width(_inline(row[0])) + _display_width(_inline(row[1])) > 80
        for row in rows
    )


def _allocate_widths(rows: list[list[str]], terminal_columns: int) -> list[int]:
    column_count = len(rows[0])
    available = max(column_count * 4, terminal_columns - (column_count * 3 + 1))
    natural = [
        max(4, max(_display_width(_inline(row[i])) for row in rows)) for i in range(column_count)
    ]
    widths = [4] * column_count
    while sum(widths) < available:
        candidates = [natural[index] - width for index, width in enumerate(widths)]
        gap = max(candidates)
        if gap <= 0:
            break
        widths[candidates.index(gap)] += 1
    return widths


def _wrap_cell(value: str, width: int) -> list[str]:
    if not value.strip():
        return [""]
    lines: list[str] = []
    current = ""
    current_width = 0
    for character in value.strip():
        char_width = _display_width(character)
        if character.isspace():
            if current and current_width < width:
                current += " "
                current_width += 1
            continue
        if current and current_width + char_width > width:
            lines.append(current.rstrip())
            current = ""
            current_width = 0
        current += character
        current_width += char_width
    if current or not lines:
        lines.append(current.rstrip())
    return lines


def _pad_display(value: str, width: int) -> str:
    return value + " " * max(0, width - _display_width(value))
