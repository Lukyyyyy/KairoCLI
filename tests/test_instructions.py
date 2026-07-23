from pathlib import Path

import pytest

from kairocli.cli import make_agent
from kairocli.config import AppConfig
from kairocli.instructions import InstructionResolver
from kairocli.paths import KairoPaths
from kairocli.prompts import PromptAssembler


def _paths(tmp_path: Path) -> KairoPaths:
    workspace = tmp_path / "work"
    workspace.mkdir()
    return KairoPaths.discover(workspace, tmp_path / "home")


def test_instruction_layers_and_safe_imports_follow_reference_order(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    paths.user_dir.mkdir(parents=True)
    paths.project_dir.mkdir()
    (paths.workspace / "docs").mkdir()
    (paths.user_dir / "KAIRO.md").write_text("user rule", encoding="utf-8")
    (paths.workspace / "docs" / "rules.md").write_text("imported rule", encoding="utf-8")
    (paths.workspace / "KAIRO.md").write_text(
        "@docs/rules.md\n@../outside.md\nroot rule", encoding="utf-8"
    )
    (tmp_path / "outside.md").write_text("outside secret", encoding="utf-8")
    (paths.project_dir / "KAIRO.md").write_text("dot rule", encoding="utf-8")
    (paths.workspace / "KAIRO.local.md").write_text("local rule", encoding="utf-8")
    (paths.project_dir / "KAIRO.local.md").write_text("dot local rule", encoding="utf-8")

    context = InstructionResolver(paths).base_context()

    assert "imported rule" in context
    assert "outside secret" not in context
    assert context.index("user rule") < context.index("root rule")
    assert context.index("root rule") < context.index("dot rule")
    assert context.index("dot rule") < context.index("local rule")
    assert context.index("local rule") < context.index("dot local rule")


def test_nested_instructions_apply_only_to_their_subtree(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    feature = paths.workspace / "src" / "feature"
    sibling = paths.workspace / "src" / "sibling"
    feature.mkdir(parents=True)
    sibling.mkdir(parents=True)
    (paths.workspace / "KAIRO.md").write_text("root", encoding="utf-8")
    (paths.workspace / "src" / "KAIRO.md").write_text("src", encoding="utf-8")
    (feature / "KAIRO.md").write_text("feature", encoding="utf-8")
    (feature / "KAIRO.local.md").write_text("feature local", encoding="utf-8")
    (sibling / "KAIRO.md").write_text("sibling only", encoding="utf-8")

    resolver = InstructionResolver(paths)
    applicable = resolver.for_target("src/feature/new.py")
    sibling_context = resolver.for_target("src/sibling/file.py")

    assert applicable.index("root") < applicable.index("src")
    assert applicable.index("src") < applicable.index("feature")
    assert applicable.index("feature") < applicable.index("feature local")
    assert "sibling only" not in applicable
    assert "feature local" not in sibling_context
    with pytest.raises(ValueError, match="workspace"):
        resolver.for_target("../outside.py")


def test_instruction_import_cycles_depth_binary_and_budget_are_bounded(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    (paths.workspace / "KAIRO.md").write_text("@a.md\n" + "x" * 5_000, encoding="utf-8")
    (paths.workspace / "a.md").write_text("@b.md\na", encoding="utf-8")
    (paths.workspace / "b.md").write_text("@a.md\nb", encoding="utf-8")

    context = InstructionResolver(paths, max_chars=1_000).base_context()

    assert len(context) <= 1_000
    assert "a" in context and "b" in context
    assert "truncated" in context


def test_instruction_import_cannot_follow_symlink_outside_workspace(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    outside = tmp_path / "outside.md"
    outside.write_text("outside secret", encoding="utf-8")
    linked = paths.workspace / "linked.md"
    try:
        linked.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")
    (paths.workspace / "KAIRO.md").write_text("@linked.md\nsafe rule", encoding="utf-8")

    context = InstructionResolver(paths).base_context()

    assert "safe rule" in context
    assert "outside secret" not in context


def test_scoped_index_skips_dependency_and_symlink_trees(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    (paths.workspace / "src").mkdir()
    (paths.workspace / "src" / "KAIRO.md").write_text("src", encoding="utf-8")
    (paths.workspace / "node_modules" / "pkg").mkdir(parents=True)
    (paths.workspace / "node_modules" / "pkg" / "KAIRO.md").write_text(
        "dependency", encoding="utf-8"
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "KAIRO.md").write_text("outside", encoding="utf-8")
    try:
        (paths.workspace / "linked").symlink_to(outside, target_is_directory=True)
    except OSError:
        pass

    index = InstructionResolver(paths).scoped_index()

    assert "src/KAIRO.md" in index
    assert "node_modules" not in index
    assert "linked" not in index


async def test_prompt_index_and_instruction_tool_share_resolver(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    nested = paths.workspace / "pkg"
    nested.mkdir()
    (paths.workspace / "KAIRO.md").write_text("root instruction", encoding="utf-8")
    (nested / "KAIRO.md").write_text("nested instruction", encoding="utf-8")
    prompt = PromptAssembler(paths).assemble()
    agent = make_agent(paths, AppConfig.load(paths))

    assert "instruction_scope_index" in prompt
    assert "pkg/KAIRO.md" in prompt
    schema_names = {item["function"]["name"] for item in agent.tools.schemas()}
    assert "load_project_instructions" in schema_names
    loaded = await agent.tools.execute("load_project_instructions", {"path": "pkg/module.py"})
    assert "root instruction" in loaded
    assert "nested instruction" in loaded
    escaped = await agent.tools.execute("load_project_instructions", {"path": "../outside"})
    assert "Instruction target must stay within the workspace" in escaped
    await agent.tools.close()
