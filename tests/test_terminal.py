from __future__ import annotations

from typing import Any

from kairocli.cli import _print_status, _print_untrusted, _write_stream
from kairocli.terminal import TerminalStreamSanitizer, sanitize_terminal_text


def test_terminal_sanitizer_removes_escape_protocols_and_spoofing_controls() -> None:
    unsafe = (
        "\x1b[31mred\x1b[0m "
        "\x1b]0;forged title\x07"
        "\x1b]52;c;Y2xpcGJvYXJk\x1b\\"
        "\x1bPprivate payload\x1b\\"
        "a\x08b\x7fc\x85d\u202ee"
    )
    assert sanitize_terminal_text(unsafe) == "red abcde"


def test_terminal_sanitizer_preserves_text_and_normalizes_carriage_returns() -> None:
    assert sanitize_terminal_text("你好\tworld\r\nnext\rover") == (
        "你好\tworld\nnext\nover"
    )


def test_terminal_sanitizer_makes_split_sequences_non_executable() -> None:
    sanitizer = TerminalStreamSanitizer()
    rendered = sanitizer.feed("prefix\x1b")
    rendered += sanitizer.feed("]52;c;secret\x07suffix\x1b[")
    rendered += sanitizer.feed("31mred\x1b[0")
    rendered += sanitizer.feed("m") + sanitizer.finish()
    assert rendered == "prefixsuffixred"
    assert "\x1b" not in rendered
    assert "\x07" not in rendered


def test_stream_sanitizer_handles_c1_controls_and_split_crlf() -> None:
    sanitizer = TerminalStreamSanitizer()
    rendered = sanitizer.feed("a\r") + sanitizer.feed("\nb\x9b31mred\x9b0m")
    rendered += sanitizer.feed("\x9dtitle") + sanitizer.feed("\x07c")
    assert rendered == "a\nbredc"


def test_cli_untrusted_render_boundaries_disable_controls(
    monkeypatch: Any, capsys: Any
) -> None:
    class PlainConsole:
        def print(self, value: object = "", **kwargs: Any) -> None:
            print(value)

    _print_untrusted(PlainConsole(), "[red]literal[/red]\x1b]52;c;x\x07")
    assert capsys.readouterr().out == "[red]literal[/red]\n"

    monkeypatch.setattr("sys.stdout", __import__("io").StringIO())
    _write_stream("safe\x1b[31m red\x1b[0m")
    assert __import__("sys").stdout.getvalue() == "safe red"


def test_plain_console_status_does_not_require_rich_keyword_arguments(
    capsys: Any,
) -> None:
    class PlainConsole:
        def print(self, value: object = "") -> None:
            print(value)

    _print_status(PlainConsole(), "✓ done\x1b]0;forged\x07", "green")
    assert capsys.readouterr().out == "✓ done\n"
