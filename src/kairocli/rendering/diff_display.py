import difflib

from ..models import FileDiff
from .terminal import sanitize_terminal_text

MAX_RENDERED_DIFF_LINES = 400


def render_file_diff(diff: FileDiff, max_lines: int = MAX_RENDERED_DIFF_LINES) -> str:
    path = sanitize_terminal_text(diff.path).replace("\n", " ")[:1_000]
    heading = f"📝 {path or '(unnamed)'}"
    if diff.omitted_reason:
        reason = sanitize_terminal_text(diff.omitted_reason).replace("\n", " ")[:500]
        return f"{heading}\n  (diff omitted: {reason})"
    if diff.before == diff.after:
        return f"{heading}\n  (content unchanged)"
    before = [] if diff.before is None else diff.before.split("\n")
    after = [] if diff.after is None else diff.after.split("\n")
    lines = list(
        difflib.unified_diff(
            before,
            after,
            fromfile="/dev/null" if diff.before is None else f"a/{path}",
            tofile="/dev/null" if diff.after is None else f"b/{path}",
            n=2,
            lineterm="",
        )
    )
    safe_lines = [sanitize_terminal_text(line) for line in lines]
    limit = max(10, min(max_lines, MAX_RENDERED_DIFF_LINES))
    if len(safe_lines) > limit:
        head = limit * 2 // 3
        tail = limit - head
        safe_lines = [
            *safe_lines[:head],
            f"... ({len(lines) - limit} diff lines omitted) ...",
            *safe_lines[-tail:],
        ]
    return "\n".join((heading, *safe_lines))
