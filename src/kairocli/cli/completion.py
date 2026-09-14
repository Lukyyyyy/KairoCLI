"""Input highlighting and completion for Kairo CLI frontends."""

from __future__ import annotations

import os
import re
from pathlib import Path

_SLASH_SUBCOMMANDS: dict[str, tuple[str, ...]] = {
    "/browser": ("connect", "disconnect", "status", "tabs"),
    "/config": ("provider",),
    "/history": ("clear",),
    "/hitl": ("on", "off"),
    "/mcp": ("list", "disable", "enable", "logs", "prompts", "resources", "restart"),
    "/memory": ("clear", "delete", "list", "search"),
    "/session": ("delete", "list", "new", "resume", "save", "status"),
    "/shell": ("exec", "list", "start", "stop"),
    "/skill": ("install", "list", "off", "on", "reload", "show"),
    "/snapshot": ("clean", "list", "status"),
    "/task": ("add", "cancel", "list", "log"),
    "/todo": ("add", "clear", "done", "list", "remove", "reopen", "start"),
    "/todo clear": ("completed", "all"),
    "/trace": ("off", "on", "reasoning", "status"),
    "/trace reasoning": ("off", "on"),
    "/wechat": ("setup", "start", "status", "stop"),
}

_CONFIG_FIELDS = (
    "api-key",
    "base-url",
    "model",
    "lora-id",
    "context-window",
    "temperature",
    "max-tokens",
)

_INPUT_HIGHLIGHTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"@[\w./:~${}<>-]+"),
        "fg:#5f87ff underline",
    ),
    (
        re.compile(r"@image:(?:<[^>]*>|\S*)|@clipboard(?![\w])"),
        "fg:#d787ff bold",
    ),
    (
        re.compile(
            r"\b(?:api[_-]?key|token|password|secret|authorization|bearer)\b",
            re.IGNORECASE,
        ),
        "fg:#ffd75f bold",
    ),
    (
        re.compile(
            r"\b(?:sudo|mkfs|shutdown|reboot|halt|poweroff)\b"
            r"|\brm\s+-[a-z]*r[a-z]*f[a-z]*\s+(?:/|~|\$home)"
            r"|\b(?:curl|wget)\b[^|\n]*\|\s*(?:sh|bash|zsh|fish|ksh)\b"
            r"|\bdd\b[^\n]*\bof=/dev/",
            re.IGNORECASE,
        ),
        "fg:#ff5f5f bold underline",
    ),
)


def _highlight_input_line(text: str) -> list[tuple[str, str]]:
    if not text:
        return []
    styles = [""] * len(text)
    for pattern, style in _INPUT_HIGHLIGHTS:
        for match in pattern.finditer(text):
            for index in range(match.start(), match.end()):
                styles[index] = style
    for offset in _unclosed_input_delimiters(text):
        styles[offset] = "fg:#ffd75f underline"
    parts: list[tuple[str, str]] = []
    start = 0
    for index in range(1, len(text) + 1):
        if index == len(text) or styles[index] != styles[start]:
            parts.append((styles[start], text[start:index]))
            start = index
    return parts


def _unclosed_input_delimiters(text: str) -> list[int]:
    result: list[int] = []
    for quote in ("'", '"'):
        open_at = -1
        escaped = False
        for index, char in enumerate(text):
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                open_at = -1 if open_at >= 0 else index
        if open_at >= 0:
            result.append(open_at)
    for marker in ("@image:<", "@<"):
        start = text.rfind(marker)
        if start >= 0:
            angle = start + len(marker) - 1
            if ">" not in text[angle + 1 :]:
                result.append(angle)
    return result


def _slash_completion_candidates(
    text: str,
    commands: list[str],
    *,
    providers: tuple[str, ...] = (),
    mcp_servers: tuple[str, ...] = (),
    skills: tuple[str, ...] = (),
) -> list[str]:
    if " " not in text:
        return [command for command in commands if command.startswith(text)][:50]
    command, payload = text.split(" ", 1)
    normalized_command = command.casefold()
    tokens = payload.split()
    ends_with_space = payload.endswith(" ")
    if normalized_command == "/model" and len(tokens) <= 1:
        prefix = "" if ends_with_space else (tokens[0] if tokens else "")
        return _named_completion_lines(command, (), prefix, providers)
    if normalized_command == "/config" and tokens[:1] and tokens[0].casefold() == "provider":
        if len(tokens) == 1:
            return _named_completion_lines(command, ("provider",), "", providers)
        if len(tokens) == 2 and not ends_with_space:
            return _named_completion_lines(command, ("provider",), tokens[1], providers)
        if len(tokens) == 2:
            return [f"{command} provider {tokens[1]} {field}" for field in _CONFIG_FIELDS]
        if len(tokens) == 3 and not ends_with_space:
            return [
                f"{command} provider {tokens[1]} {field}"
                for field in _CONFIG_FIELDS
                if field.startswith(tokens[2].casefold())
            ]
        return []
    if (
        normalized_command == "/mcp"
        and tokens[:1]
        and tokens[0].casefold()
        in {
            "disable",
            "enable",
            "logs",
            "prompts",
            "resources",
            "restart",
        }
    ):
        if len(tokens) == 1 or (len(tokens) == 2 and not ends_with_space):
            prefix = tokens[1] if len(tokens) == 2 else ""
            return _named_completion_lines(command, (tokens[0],), prefix, mcp_servers)
        return []
    if (
        normalized_command == "/skill"
        and tokens[:1]
        and tokens[0].casefold()
        in {
            "off",
            "on",
            "show",
        }
    ):
        if len(tokens) == 1 or (len(tokens) == 2 and not ends_with_space):
            prefix = tokens[1] if len(tokens) == 2 else ""
            return _named_completion_lines(command, (tokens[0],), prefix, skills)
        return []
    parent_tokens = tokens if ends_with_space else tokens[:-1]
    prefix = "" if ends_with_space else (tokens[-1] if tokens else "")
    parent = " ".join((normalized_command, *(token.casefold() for token in parent_tokens)))
    base = " ".join((command, *parent_tokens))
    return [
        f"{base} {option}"
        for option in _SLASH_SUBCOMMANDS.get(parent, ())
        if option.casefold().startswith(prefix.casefold())
    ][:50]


def _named_completion_lines(
    command: str,
    fixed: tuple[str, ...],
    prefix: str,
    names: tuple[str, ...],
) -> list[str]:
    base = " ".join((command, *fixed))
    return [
        f"{base} {name}"
        for name in sorted(set(names), key=str.casefold)
        if name.casefold().startswith(prefix.casefold())
    ][:50]


def _completion_word(text: str) -> str:
    angle = text.rfind("@<")
    image_angle = text.rfind("@image:<")
    start = max(angle, image_angle)
    if start >= 0 and ">" not in text[start:]:
        return text[start:]
    return re.split(r"\s", text)[-1]


def _local_path_completion_candidates(
    workspace: Path,
    word: str,
    *,
    max_results: int = 50,
    max_scan: int = 1_000,
) -> list[str]:
    if max_results <= 0 or max_scan <= 0 or len(word) > 4096:
        return []
    if word.startswith("@image:"):
        marker = "@image:"
        prefix = word[len(marker) :]
    elif word.startswith("@"):
        marker = "@"
        prefix = word[1:]
        if prefix.startswith("clipboard") or ":" in prefix:
            return []
    else:
        return []
    angle = prefix.startswith("<")
    if angle:
        prefix = prefix[1:]
    if "\x00" in prefix or ">" in prefix:
        return []
    typed = Path(prefix or ".")
    if typed.is_absolute() or ".." in typed.parts:
        return []
    parent = typed.parent if prefix and typed.parent != Path("") else Path(".")
    name_prefix = "" if not prefix or prefix.endswith(("/", os.sep)) else typed.name
    if prefix.endswith(("/", os.sep)):
        parent = typed
    try:
        root = workspace.resolve(strict=True)
        base = (root / parent).resolve(strict=True)
        base.relative_to(root)
        if not base.is_dir():
            return []
    except (OSError, RuntimeError, ValueError):
        return []
    candidates: list[tuple[str, bool]] = []
    try:
        with os.scandir(base) as entries:
            for scanned, entry in enumerate(entries, start=1):
                if scanned > max_scan:
                    break
                if not entry.name.startswith(name_prefix):
                    continue
                try:
                    lexical = base / entry.name
                    resolved = lexical.resolve(strict=True)
                    resolved.relative_to(root)
                    candidates.append((lexical.relative_to(root).as_posix(), resolved.is_dir()))
                except (OSError, RuntimeError, ValueError):
                    continue
    except OSError:
        return []
    values: list[str] = []
    for display, is_directory in sorted(candidates, key=lambda item: item[0].casefold())[
        :max_results
    ]:
        if is_directory:
            display += "/"
        needs_angle = angle or any(char.isspace() for char in display)
        if marker == "@image:":
            values.append(f"@image:<{display}>" if needs_angle else marker + display)
        else:
            values.append(f"@<{display}>" if needs_angle else marker + display)
    return values
