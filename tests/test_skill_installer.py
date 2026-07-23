from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

import kairocli.skill_installer as installer_module
from kairocli.cli import make_agent
from kairocli.config import AppConfig
from kairocli.paths import KairoPaths
from kairocli.policy import ApprovalPolicy, ApprovalResult
from kairocli.skill_installer import (
    SkillInstallError,
    SkillInstallRequest,
    SkillSource,
    install_skill,
    parse_skill_install_request,
    resolve_skill_source,
)
from kairocli.skills import SkillRegistry, handle_skill_command


def _registry(tmp_path: Path) -> SkillRegistry:
    paths = KairoPaths.discover(tmp_path / "work", tmp_path / "home")
    paths.workspace.mkdir()
    registry = SkillRegistry(paths)
    registry.reload()
    return registry


def _skill(root: Path, name: str = "demo", body: str = "Use the demo.") -> Path:
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Demo Skill\n---\n{body}\n",
        encoding="utf-8",
    )
    references = root / "references"
    references.mkdir()
    (references / "guide.txt").write_text("guide", encoding="utf-8")
    return root


def test_parse_install_request_supports_scope_and_source_options() -> None:
    request = parse_skill_install_request(
        'owner/repo --project --ref release-1 --path "skills/demo" --name renamed --force'
    )
    assert request.source == "owner/repo"
    assert request.scope == "project"
    assert request.ref == "release-1"
    assert request.subdirectory == "skills/demo"
    assert request.name == "renamed"
    assert request.force


@pytest.mark.parametrize(
    "arguments",
    (
        "",
        "one two",
        "demo --scope system",
        "demo --ref ../main",
        "demo --path ../escape",
        "owner/repo --path --config=core.hooksPath=evil",
        "demo --unknown",
    ),
)
def test_parse_install_request_rejects_invalid_arguments(arguments: str) -> None:
    with pytest.raises(SkillInstallError):
        parse_skill_install_request(arguments)


def test_resolve_sources_supports_curated_github_and_tree_urls() -> None:
    curated = resolve_skill_source(parse_skill_install_request("linear"))
    assert curated.location == "https://github.com/openai/skills.git"
    assert curated.ref == "main"
    assert curated.subdirectory == "skills/.curated/linear"

    shorthand = resolve_skill_source(parse_skill_install_request("openai/skills --path skills/pdf"))
    assert shorthand.location == "https://github.com/openai/skills.git"
    assert shorthand.subdirectory == "skills/pdf"

    tree = resolve_skill_source(
        parse_skill_install_request(
            "https://github.com/openai/skills/tree/main/skills/.curated/pdf"
        )
    )
    assert tree.location == "https://github.com/openai/skills.git"
    assert tree.ref == "main"
    assert tree.subdirectory == "skills/.curated/pdf"


@pytest.mark.parametrize(
    "source",
    (
        "http://github.com/openai/skills",
        "https://user:secret@github.com/openai/skills",
        "https://github.com:8443/openai/skills",
        "https://github.com/openai/skills?token=secret",
        "https://github.com/openai/skills/tree/../skills/demo",
    ),
)
def test_resolve_source_rejects_unsafe_remote_urls(source: str) -> None:
    with pytest.raises(SkillInstallError):
        resolve_skill_source(parse_skill_install_request(source))


def test_install_local_skill_to_user_scope_and_reload(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    source = _skill(tmp_path / "source")

    result = install_skill(registry, parse_skill_install_request(str(source)))

    target = registry.paths.user_dir / "skills" / "demo"
    assert result.target == target
    assert result.active
    assert not result.replaced
    assert registry.skills["demo"].path == target / "SKILL.md"
    assert (target / "references" / "guide.txt").read_text(encoding="utf-8") == "guide"
    metadata = json.loads((target / ".kairocli-install.json").read_text(encoding="utf-8"))
    assert metadata["source"] == str(source)
    assert len(metadata["sha256"]) == 64
    if os.name != "nt":
        assert stat_mode(target) == 0o700
        assert stat_mode(target / "SKILL.md") == 0o600


def test_install_project_skill_via_shared_command_handler(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    source = _skill(tmp_path / "project source")
    agent = type("Agent", (), {"base_system_prompt": "base", "system_prompt": "base"})()

    output = handle_skill_command(f'install "{source}" --project', registry, agent)

    target = registry.paths.project_dir / "skills" / "demo"
    assert output.startswith("Installed Skill demo [project]")
    assert target.is_dir()
    assert "demo" in agent.system_prompt


def test_conflict_requires_force_and_force_replaces_then_enables(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    first = _skill(tmp_path / "first", body="first")
    second = _skill(tmp_path / "second", body="second")
    install_skill(registry, parse_skill_install_request(str(first)))
    registry.set_enabled("demo", False)

    with pytest.raises(SkillInstallError, match="--force"):
        install_skill(registry, parse_skill_install_request(str(second)))
    assert "first" in (registry.paths.user_dir / "skills" / "demo" / "SKILL.md").read_text()

    result = install_skill(registry, parse_skill_install_request(f"{second} --force"))
    assert result.replaced
    assert result.active
    assert registry.skills["demo"].enabled
    assert "second" in registry.skills["demo"].body


def test_project_override_is_reported_as_the_active_skill(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    project_source = _skill(tmp_path / "project", body="project")
    user_source = _skill(tmp_path / "user", body="user")
    install_skill(registry, parse_skill_install_request(f"{project_source} --project"))

    result = install_skill(registry, parse_skill_install_request(str(user_source)))

    assert not result.active
    assert registry.skills["demo"].source.value == "project"


@pytest.mark.skipif(os.name == "nt", reason="symlink creation differs on Windows")
def test_install_rejects_symlink_without_partial_target(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    source = _skill(tmp_path / "source")
    (source / "escape").symlink_to(tmp_path / "outside")

    with pytest.raises(SkillInstallError, match="symlink"):
        install_skill(registry, parse_skill_install_request(str(source)))
    assert not (registry.paths.user_dir / "skills" / "demo").exists()


@pytest.mark.skipif(os.name == "nt", reason="symlink creation differs on Windows")
def test_install_rejects_source_root_and_path_symlinks(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    source = tmp_path / "repository"
    _skill(source / "skills" / "demo")
    root_link = tmp_path / "linked-skill"
    root_link.symlink_to(source / "skills" / "demo", target_is_directory=True)
    path_link = source / "linked"
    path_link.symlink_to(source / "skills", target_is_directory=True)

    with pytest.raises(SkillInstallError, match="symlink"):
        install_skill(registry, parse_skill_install_request(str(root_link)))
    with pytest.raises(SkillInstallError, match="symlink"):
        install_skill(
            registry,
            parse_skill_install_request(f"{source} --path linked/demo"),
        )


def test_install_rejects_excessive_tree_without_partial_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(tmp_path)
    source = _skill(tmp_path / "source")
    (source / "extra.txt").write_text("extra", encoding="utf-8")
    monkeypatch.setattr(installer_module, "MAX_INSTALL_FILES", 2)

    with pytest.raises(SkillInstallError, match="entries"):
        install_skill(registry, parse_skill_install_request(str(source)))
    assert not (registry.paths.user_dir / "skills" / "demo").exists()


def test_force_install_rolls_back_when_atomic_switch_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(tmp_path)
    first = _skill(tmp_path / "first", body="first")
    second = _skill(tmp_path / "second", body="second")
    install_skill(registry, parse_skill_install_request(str(first)))
    real_replace = installer_module.os.replace

    def fail_staging_switch(source: str | Path, target: str | Path) -> None:
        if Path(source).name.startswith(".skill-install-"):
            raise OSError("switch failed")
        real_replace(source, target)

    monkeypatch.setattr(installer_module.os, "replace", fail_staging_switch)
    with pytest.raises(SkillInstallError, match="switch failed"):
        install_skill(registry, parse_skill_install_request(f"{second} --force"))

    installed = registry.paths.user_dir / "skills" / "demo" / "SKILL.md"
    assert installed.is_file()
    assert "first" in installed.read_text(encoding="utf-8")


def test_remote_install_uses_checked_out_subdirectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(tmp_path)

    def fake_checkout(source: Any, repository: Path) -> None:
        _skill(repository / "skills" / "demo")

    monkeypatch.setattr(installer_module, "_checkout_git_source", fake_checkout)
    result = install_skill(
        registry,
        parse_skill_install_request("owner/repo --path skills/demo --ref main"),
    )
    assert result.active
    assert result.source == "owner/repo"


def test_curated_default_ref_does_not_fetch_main_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(installer_module.shutil, "which", lambda _name: "/usr/bin/git")
    monkeypatch.setattr(installer_module, "_run_git", lambda *args: calls.append(args))

    installer_module._checkout_git_source(  # noqa: SLF001 - exercise checkout sequence
        SkillSource(
            kind="git",
            location=installer_module.CURATED_SKILLS_REPOSITORY,
            display="openai/skills:pdf",
            ref="main",
            subdirectory="skills/.curated/pdf",
        ),
        tmp_path / "checkout",
    )

    assert not any("fetch" in call for call in calls)
    assert calls[-1][-3:] == ("checkout", "--detach", "HEAD")


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_git_checkout_uses_ref_and_sparse_path(tmp_path: Path) -> None:
    source_repository = tmp_path / "source-repository"
    source_repository.mkdir()
    subprocess.run(["git", "init", "-q", str(source_repository)], check=True)
    subprocess.run(
        ["git", "-C", str(source_repository), "config", "user.email", "test@example.test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(source_repository), "config", "user.name", "Test"], check=True)
    _skill(source_repository / "skills" / "demo")
    (source_repository / "outside.txt").write_text("outside", encoding="utf-8")
    subprocess.run(["git", "-C", str(source_repository), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source_repository), "commit", "-qm", "skill"], check=True)

    checkout = tmp_path / "checkout"
    installer_module._checkout_git_source(  # noqa: SLF001 - exercise Git safety contract
        SkillSource(
            kind="git",
            location=str(source_repository),
            display="local-test-repository",
            ref="HEAD",
            subdirectory="skills/demo",
        ),
        checkout,
    )
    assert (checkout / "skills" / "demo" / "SKILL.md").is_file()
    assert not (checkout / "outside.txt").exists()


async def test_agent_install_tool_installs_and_refreshes_prompt(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    source = _skill(tmp_path / "agent-source")
    agent = make_agent(registry.paths, AppConfig.load(registry.paths), skill_registry=registry)

    output = await agent.tools.execute("install_skill", {"source": str(source), "scope": "project"})

    assert output.startswith("Installed Skill demo [project]")
    assert "**demo**" in agent.system_prompt
    assert (registry.paths.project_dir / "skills" / "demo" / "SKILL.md").is_file()
    await agent.tools.close()


async def test_agent_install_tool_requires_hitl_approval(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    source = _skill(tmp_path / "agent-source")
    approvals: list[tuple[str, dict[str, Any]]] = []

    async def reject(tool: str, arguments: dict[str, Any]) -> ApprovalResult:
        approvals.append((tool, arguments))
        return ApprovalResult.reject("not approved")

    agent = make_agent(
        registry.paths,
        AppConfig.load(registry.paths),
        approval_policy=ApprovalPolicy(enabled=True),
        approver=reject,
        skill_registry=registry,
    )
    output = await agent.tools.execute("install_skill", {"source": str(source)})

    assert '"approval_denied": true' in output
    assert approvals == [("install_skill", {"source": str(source)})]
    assert not (registry.paths.user_dir / "skills" / "demo").exists()
    await agent.tools.close()


async def test_agent_install_tool_revalidates_git_arguments(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    agent = make_agent(registry.paths, AppConfig.load(registry.paths), skill_registry=registry)

    output = await agent.tools.execute(
        "install_skill",
        {"source": "openai/skills", "ref": "--upload-pack=malicious", "path": "skills/demo"},
    )

    assert "Git ref is invalid or unsafe" in output
    await agent.tools.close()


def test_direct_request_validation_does_not_rely_on_cli_parser() -> None:
    with pytest.raises(SkillInstallError, match="scope"):
        resolve_skill_source(SkillInstallRequest(source="demo", scope="system"))
    with pytest.raises(SkillInstallError, match="ref"):
        resolve_skill_source(SkillInstallRequest(source="owner/repo", ref="--dangerous"))


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777
