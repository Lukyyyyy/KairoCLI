"""Pure text formatting and message chunking for the WeChat channel."""

from __future__ import annotations

import re

_ANSI_ESCAPE = re.compile(r"\x1b\[[;?0-9]*[ -/]*[@-~]")
_CJK_ITALIC = re.compile(
    r"(?<!\*)\*([^*\n]*[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af][^*\n]*)\*(?!\*)"
)
_CJK_UNDERSCORE = re.compile(
    r"(?<!_)_([^_\n]*[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af][^_\n]*)_(?!_)"
)
_CODE_LANGUAGES = {
    "bash",
    "c",
    "cpp",
    "cs",
    "csharp",
    "css",
    "diff",
    "go",
    "html",
    "java",
    "javascript",
    "js",
    "json",
    "kotlin",
    "kt",
    "php",
    "properties",
    "py",
    "python",
    "rb",
    "rs",
    "ruby",
    "rust",
    "sh",
    "sql",
    "swift",
    "toml",
    "ts",
    "typescript",
    "xml",
    "yaml",
    "yml",
    "zsh",
}


def format_wechat_text(value: str) -> str:
    if not value or not value.strip():
        return ""
    text = _ANSI_ESCAPE.sub("", value).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("▪ ", "").replace("■ ", "")
    lines = text.split("\n")
    output: list[str] = []
    table: list[str] = []
    fence_language: str | None = None
    fence_lines: list[str] = []

    def flush_table() -> None:
        if table:
            output.extend(_format_wechat_table(table))
            table.clear()

    def flush_fence() -> None:
        nonlocal fence_language, fence_lines
        body = "\n".join(fence_lines).strip()
        language = fence_language or ""
        if body:
            if language.casefold() not in _CODE_LANGUAGES and _looks_like_prose_flow(body):
                output.extend(_normalize_wechat_line(body).splitlines())
            else:
                output.extend([f"```{language}", body, "```"])
        fence_language = None
        fence_lines = []

    for line in lines:
        stripped = line.lstrip()
        if fence_language is not None:
            if stripped.startswith("```"):
                flush_fence()
            else:
                fence_lines.append(line)
            continue
        if stripped.startswith("```"):
            flush_table()
            fence_language = stripped[3:].strip()
            fence_lines = []
            continue
        if _is_wechat_table_line(line):
            table.extend(_expand_collapsed_table_line(line))
            continue
        flush_table()
        output.extend(_normalize_wechat_line(line).splitlines() or [""])
    if fence_language is not None:
        flush_fence()
    flush_table()
    normalized = "\n".join(output)
    normalized = re.sub(r"(?m)^[ \t]+$", "", normalized)
    return re.sub(r"\n{3,}", "\n\n", normalized).strip()


def _normalize_wechat_line(line: str) -> str:
    normalized = re.sub(r"!\[[^]]*]\([^)]*\)", "", line).replace("~~", "")
    stripped = normalized.strip()
    if stripped.startswith("```") and stripped.endswith("```") and len(stripped) > 6:
        normalized = stripped[3:-3].strip()
    elif stripped.endswith("```") and stripped != "```":
        normalized = normalized[: normalized.rfind("```")].rstrip()
    if normalized.count("→") >= 2:
        flow = [part.strip() for part in re.split(r"\s*→\s*", normalized)]
        normalized = flow[0] + "".join(f"\n→ {part}" for part in flow[1:])
    leading = normalized.lstrip()
    if re.match(r"^#{5,6}\s+", leading):
        normalized = re.sub(r"^(\s*)#{5,6}\s+", r"\1", normalized)
    elif re.match(r"^#{1,4}\s*\S", leading):
        heading = re.sub(r"^#{1,4}\s*", "", leading).strip()
        heading = re.sub(r"\*\*\s*([^*\n]*?)\s*\*\*", r"\1", heading)
        normalized = f"**{heading}**" if heading else ""
    normalized = re.sub(r"\*\*\s*([^*\n]*?)\s*\*\*", r"**\1**", normalized)
    normalized = _CJK_ITALIC.sub(r"\1", normalized)
    return _CJK_UNDERSCORE.sub(r"\1", normalized)


def _looks_like_prose_flow(body: str) -> bool:
    return "→" in body or bool(re.search(r"[\u3400-\u9fff]", body)) or len(body) > 120


def _is_wechat_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.find("|", 1) > 0


def _expand_collapsed_table_line(line: str) -> list[str]:
    if "||" not in line:
        return [line]
    return line.replace("||", "|\n|").splitlines()


def _format_wechat_table(lines: list[str]) -> list[str]:
    combined = "\n".join(lines).replace("||", "|\n|")
    raw_lines = [line.strip() for line in combined.splitlines() if line.strip()]
    key_value_only = bool(raw_lines) and bool(
        re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", raw_lines[0])
    )
    rows: list[list[str]] = []
    for line in combined.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        cells = [cell for cell in cells if cell and not re.fullmatch(r":?-{3,}:?", cell)]
        if cells:
            rows.append([_normalize_wechat_line(cell) for cell in cells])
    if not rows:
        return []
    if key_value_only and all(len(row) == 2 for row in rows):
        return [f"- **{row[0]}**：{row[1]}" for row in rows]
    header = rows[0]
    if len(rows) == 1 and len(header) >= 4 and len(header) % 2 == 0:
        return [f"- **{header[index]}**：{header[index + 1]}" for index in range(0, len(header), 2)]
    if len(rows) == 1:
        return [" / ".join(header)]
    result: list[str] = []
    for row in rows[1:]:
        if len(row) == 2:
            result.append(f"- **{row[0]}**：{row[1]}")
        else:
            pairs = [
                f"{header[index] if index < len(header) else f'列{index + 1}'}：{cell}"
                for index, cell in enumerate(row)
            ]
            result.append("- " + "；".join(pairs))
    return result


def split_message(text: str, max_chars: int = 3_800) -> list[str]:
    chunks: list[str] = []
    remaining = text.strip()
    while len(remaining) > max_chars:
        split_at = remaining.rfind("\n", 0, max_chars)
        if split_at < max_chars // 3:
            split_at = remaining.rfind(" ", 0, max_chars)
        if split_at < max_chars // 3:
            split_at = max_chars
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks
