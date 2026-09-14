from pathlib import Path

from kairocli.models import FileDiff
from kairocli.rendering.diff_display import render_file_diff
from kairocli.tools import ToolRegistry


def test_diff_display_handles_new_deleted_modified_and_unchanged_files() -> None:
    created = render_file_diff(FileDiff("new.txt", None, "one\ntwo\n"))
    deleted = render_file_diff(FileDiff("old.txt", "one\ntwo\n", None))
    modified = render_file_diff(FileDiff("edit.txt", "before\nsame", "after\nsame"))
    unchanged = render_file_diff(FileDiff("same.txt", "same", "same"))
    assert "@@ -0,0 +1,3 @@" in created
    assert "+one" in created and "+two" in created
    assert "-one" in deleted and "-two" in deleted
    assert "-before" in modified and "+after" in modified
    assert "content unchanged" in unchanged


def test_diff_display_sanitizes_and_bounds_rendered_lines() -> None:
    rendered = render_file_diff(
        FileDiff("bad\x1b]0;title\x07.txt", "\n".join(f"a{i}" for i in range(500)), ""),
        max_lines=30,
    )
    assert "\x1b" not in rendered
    assert "title" not in rendered
    assert "diff lines omitted" in rendered
    assert len(rendered.splitlines()) <= 32
    omitted = render_file_diff(FileDiff("large", None, None, "too large\nfor display"))
    assert omitted.endswith("(diff omitted: too large for display)")


async def test_write_file_returns_display_diff_outside_model_text(tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    target.write_text("before\n", encoding="utf-8")
    output = await ToolRegistry(tmp_path).execute_output(
        "write_file", {"path": "note.txt", "content": "after\n"}
    )
    assert len(output.diffs) == 1
    assert output.diffs[0] == FileDiff("note.txt", "before\n", "after\n")
    assert "before" not in output.text
    assert "after" not in output.text


async def test_apply_patch_returns_before_and_after_display_diff(tmp_path: Path) -> None:
    target = tmp_path / "code.txt"
    target.write_text("before\nsame\n", encoding="utf-8")
    patch = """diff --git a/code.txt b/code.txt
--- a/code.txt
+++ b/code.txt
@@ -1,2 +1,2 @@
-before
+after
 same
"""
    output = await ToolRegistry(tmp_path).execute_output("apply_patch", {"patch": patch})
    assert output.diffs == (FileDiff("code.txt", "before\nsame\n", "after\nsame\n"),)
    assert "before" not in output.text
    assert "after" not in output.text
