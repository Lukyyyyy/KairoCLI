from kairocli.terminal_capabilities import (
    detect_renderer_mode,
    resolve_renderer_mode,
    supports_truecolor,
)


def test_inline_requires_an_interactive_non_dumb_terminal() -> None:
    assert (
        resolve_renderer_mode(
            "inline",
            input_is_tty=True,
            output_is_tty=True,
            term="xterm-256color",
            columns=120,
            rows=40,
        ).mode
        == "inline"
    )
    assert (
        resolve_renderer_mode(
            "inline",
            input_is_tty=True,
            output_is_tty=True,
            term="dumb",
        ).mode
        == "plain"
    )
    assert resolve_renderer_mode("inline", input_is_tty=False, output_is_tty=True).mode == "plain"


def test_tui_requires_real_minimum_terminal_and_honors_no_tui() -> None:
    normal = dict(
        input_is_tty=True,
        output_is_tty=True,
        term="xterm-256color",
    )
    assert resolve_renderer_mode("tui", columns=120, rows=40, **normal).mode == "tui"
    small = resolve_renderer_mode("tui", columns=79, rows=23, **normal)
    assert small.mode == "inline"
    assert "80x24" in small.notice
    assert (
        resolve_renderer_mode("tui", columns=120, rows=40, no_tui=True, **normal).mode == "inline"
    )
    assert resolve_renderer_mode("tui", input_is_tty=False, output_is_tty=False).mode == "plain"


def test_plain_and_no_color_semantics_do_not_require_ansi() -> None:
    assert (
        resolve_renderer_mode("plain", input_is_tty=False, output_is_tty=False, term="dumb").mode
        == "plain"
    )
    assert supports_truecolor({"COLORTERM": "24BIT", "NO_COLOR": "1"})
    assert not supports_truecolor({"NO_COLOR": "1"})


def test_detection_treats_broken_isatty_as_noninteractive() -> None:
    class BrokenStream:
        def isatty(self) -> bool:
            raise OSError("closed")

    resolution = detect_renderer_mode(
        "tui", BrokenStream(), BrokenStream(), {}, columns=120, rows=40
    )
    assert resolution.mode == "plain"
