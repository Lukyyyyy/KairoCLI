from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RendererResolution:
    mode: str
    notice: str = ""


def resolve_renderer_mode(
    requested: str,
    *,
    input_is_tty: bool,
    output_is_tty: bool,
    term: str = "",
    columns: int = 0,
    rows: int = 0,
    no_tui: bool = False,
) -> RendererResolution:
    mode = requested.casefold()
    if mode == "plain":
        return RendererResolution("plain")
    ansi = output_is_tty and term.casefold() != "dumb"
    interactive = input_is_tty and output_is_tty
    fallback = "inline" if interactive and ansi else "plain"
    if mode == "inline":
        if fallback == "inline":
            return RendererResolution("inline")
        return RendererResolution(
            "plain", "Terminal is not interactive/ANSI-capable; using plain renderer."
        )
    if mode != "tui":
        return RendererResolution("plain", f"Unknown renderer {requested!r}; using plain.")
    if no_tui:
        return RendererResolution(fallback, "KAIROCLI_NO_TUI is enabled; TUI was disabled.")
    if not interactive or not ansi:
        return RendererResolution(
            fallback, "No interactive ANSI terminal is available; TUI was disabled."
        )
    if columns < 80 or rows < 24:
        return RendererResolution(
            "inline",
            f"Terminal size {columns}x{rows} is below 80x24; using inline renderer.",
        )
    return RendererResolution("tui")


def detect_renderer_mode(
    requested: str,
    stdin: Any,
    stdout: Any,
    environment: Mapping[str, str],
    *,
    columns: int,
    rows: int,
) -> RendererResolution:
    return resolve_renderer_mode(
        requested,
        input_is_tty=_isatty(stdin),
        output_is_tty=_isatty(stdout),
        term=environment.get("TERM", ""),
        columns=columns,
        rows=rows,
        no_tui=environment.get("KAIROCLI_NO_TUI", "").casefold() in {"1", "true", "yes", "on"},
    )


def supports_truecolor(environment: Mapping[str, str]) -> bool:
    return environment.get("COLORTERM", "").casefold() in {"truecolor", "24bit"}


def _isatty(stream: Any) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, OSError, ValueError):
        return False
