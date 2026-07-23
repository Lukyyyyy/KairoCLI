from __future__ import annotations

import html
import json
import os
import re
import secrets
import threading
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from itertools import islice
from pathlib import Path
from typing import Any

from .paths import KairoPaths, reject_symlink_components
from .text_safety import safe_text

MAX_SKILL_FILE_BYTES = 2 * 1024 * 1024
MAX_SKILL_BODY_CHARS = 5 * 1024
MAX_REFERENCE_CHARS = 100_000
MAX_INDEX_CHARS = 4_096
MAX_INDEX_SKILLS = 20
MAX_DESCRIPTION_CHARS = 500
MAX_SKILL_STATE_BYTES = 1024 * 1024
MAX_DISABLED_SKILLS = 1_000
MAX_SKILL_REFERENCES = 1_000
MAX_SKILL_REFERENCE_SCAN_ENTRIES = 10_000
MAX_SKILL_REFERENCE_SCAN_DEPTH = 16
MAX_SKILL_STATE_JSON_DEPTH = 16
MAX_SKILL_STATE_JSON_NODES = 10_000
_VALID_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SKILL_STATE_THREAD_LOCK = threading.RLock()


class SkillSource(StrEnum):
    BUILTIN = "builtin"
    USER = "user"
    PROJECT = "project"


@dataclass(frozen=True, slots=True)
class FrontmatterResult:
    metadata: dict[str, Any]
    body: str
    warnings: tuple[str, ...] = ()


@dataclass(slots=True)
class Skill:
    name: str
    description: str
    path: Path
    enabled: bool = True
    version: str = ""
    author: str = ""
    tags: tuple[str, ...] = ()
    source: SkillSource = SkillSource.USER
    body: str = ""
    references_dir: Path | None = None

    def load(self, max_chars: int | None = None) -> str:
        if max_chars is None or len(self.body) <= max_chars:
            return self.body
        return (
            self.body[:max_chars] + f"\n\n...(skill body truncated at {max_chars} chars; "
            f"use /skill show {self.name} for the full body)"
        )

    def load_for_agent(self) -> str:
        return self.load(MAX_SKILL_BODY_CHARS)

    def references(self) -> list[str]:
        if self.references_dir is None:
            return []
        if self.references_dir.is_symlink():
            return []
        root = self.references_dir.resolve()
        return _bounded_reference_files(root)

    def load_reference(self, relative_path: str, max_chars: int = MAX_REFERENCE_CHARS) -> str:
        if self.references_dir is None:
            raise FileNotFoundError(f"Skill {self.name} has no references directory")
        if self.references_dir.is_symlink():
            raise PermissionError("Skill references directory cannot be a symlink")
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise PermissionError("Skill reference escapes the references directory")
        root = self.references_dir.resolve()
        lexical = root / relative
        current = root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise PermissionError("Skill reference cannot traverse a symlink")
        target = lexical.resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise PermissionError("Skill reference escapes the references directory") from exc
        if not target.is_file():
            raise FileNotFoundError(f"Skill reference not found: {relative_path}")
        with target.open("rb") as stream:
            sample = stream.read(8_192)
        if b"\x00" in sample:
            raise ValueError("Skill references must be text files")
        bounded = min(max(int(max_chars), 1_000), MAX_REFERENCE_CHARS)
        with target.open("r", encoding="utf-8", errors="replace") as stream:
            content = stream.read(bounded + 1)
        if len(content) <= bounded:
            return content
        return content[:bounded] + (
            f"\n\n...[reference truncated at {bounded} chars; narrow the requested reference]"
        )


class SkillRegistry:
    def __init__(self, paths: KairoPaths) -> None:
        self.paths = paths
        self.state_file = paths.user_dir / "skills.json"
        self.skills: dict[str, Skill] = {}
        self.warnings: list[str] = []

    def reload(self) -> None:
        self.warnings = []
        disabled = self._disabled()
        roots = [
            (
                Path(__file__).parent / "builtin_skills",
                SkillSource.BUILTIN,
                Path(__file__).parent,
            ),
            (self.paths.user_dir / "skills", SkillSource.USER, self.paths.home),
            (
                self.paths.project_dir / "skills",
                SkillSource.PROJECT,
                self.paths.workspace,
            ),
        ]
        found: dict[str, Skill] = {}
        for root, source, trusted_base in roots:
            uses_symlink = _path_uses_symlink(root, trusted_base)
            if uses_symlink or not root.is_dir():
                if uses_symlink:
                    self.warnings.append(f"{root}: skill root symlink ignored")
                continue
            try:
                directories = sorted(
                    item for item in root.iterdir() if not item.is_symlink() and item.is_dir()
                )
            except OSError as exc:
                self.warnings.append(f"{root}: skill root scan failed: {type(exc).__name__}")
                continue
            for directory in directories:
                skill_file = directory / "SKILL.md"
                if skill_file.is_symlink() or not skill_file.is_file():
                    if skill_file.is_symlink():
                        self.warnings.append(f"{skill_file}: symlink ignored")
                    continue
                skill = self._load_skill(directory, skill_file, source, disabled)
                if skill is not None:
                    found[skill.name] = skill
        self.skills = dict(sorted(found.items()))

    def _load_skill(
        self,
        directory: Path,
        skill_file: Path,
        source: SkillSource,
        disabled: set[str],
    ) -> Skill | None:
        try:
            if skill_file.stat().st_size > MAX_SKILL_FILE_BYTES:
                raise ValueError(f"SKILL.md exceeds {MAX_SKILL_FILE_BYTES} bytes")
            with skill_file.open("rb") as stream:
                encoded = stream.read(MAX_SKILL_FILE_BYTES + 1)
            if len(encoded) > MAX_SKILL_FILE_BYTES:
                raise ValueError(f"SKILL.md exceeds {MAX_SKILL_FILE_BYTES} bytes")
            content = encoded.decode("utf-8", errors="replace")
        except (OSError, ValueError) as exc:
            self.warnings.append(f"{skill_file}: {safe_text(exc)}")
            return None
        parsed = parse_skill_document(content)
        self.warnings.extend(f"{skill_file}: {warning}" for warning in parsed.warnings)
        raw_name = str(parsed.metadata.get("name") or directory.name).strip()
        if not _VALID_NAME.fullmatch(raw_name):
            self.warnings.append(
                f"{skill_file}: invalid skill name {raw_name!r}; using directory name"
            )
            raw_name = directory.name
        if not _VALID_NAME.fullmatch(raw_name):
            self.warnings.append(f"{skill_file}: directory name is not a valid skill name")
            return None
        tags_value = parsed.metadata.get("tags", [])
        tags = tuple(str(item) for item in tags_value) if isinstance(tags_value, list) else ()
        references_dir = directory / "references"
        valid_references = references_dir if references_dir.is_dir() else None
        if references_dir.is_symlink():
            self.warnings.append(f"{references_dir}: symlink ignored")
            valid_references = None
        return Skill(
            name=raw_name,
            description=str(parsed.metadata.get("description") or "").strip(),
            path=skill_file,
            enabled=raw_name not in disabled,
            version=str(parsed.metadata.get("version") or "").strip(),
            author=str(parsed.metadata.get("author") or "").strip(),
            tags=tags,
            source=source,
            body=parsed.body,
            references_dir=valid_references,
        )

    def set_enabled(self, name: str, enabled: bool) -> None:
        if name not in self.skills:
            raise KeyError(name)
        with self._state_lock():
            disabled = self._disabled()
            if enabled:
                disabled.discard(name)
            else:
                disabled.add(name)
            self._write_disabled(disabled)
        self.skills[name].enabled = enabled

    def index(self, max_skills: int = MAX_INDEX_SKILLS, max_chars: int = MAX_INDEX_CHARS) -> str:
        enabled = [skill for skill in self.skills.values() if skill.enabled]
        if not enabled:
            return ""
        lines = ["## Available Skills (load full guidance only when relevant)", ""]
        for skill in enabled[: max(1, max_skills)]:
            description = html.escape(skill.description[:MAX_DESCRIPTION_CHARS])
            if len(skill.description) > MAX_DESCRIPTION_CHARS:
                description += "..."
            lines.append(f"- **{skill.name}** [{skill.source}]: {description}")
        lines.extend(
            [
                "",
                "Call load_skill with the exact name when the current task matches. "
                "Do not load unrelated skills.",
            ]
        )
        rendered = "\n".join(lines)
        if len(rendered) <= max_chars:
            return rendered
        return rendered[:max_chars] + "\n...(skill index truncated)"

    def _disabled(self) -> set[str]:
        try:
            reject_symlink_components(self.state_file, "Skill state")
        except ValueError:
            self.warnings.append(f"{self.state_file}: state symlink ignored")
            return set()
        if not self.state_file.is_file():
            return set()
        try:
            if self.state_file.stat().st_size > MAX_SKILL_STATE_BYTES:
                raise ValueError("skill state exceeds the 1 MiB limit")
            with self.state_file.open("rb") as stream:
                encoded = stream.read(MAX_SKILL_STATE_BYTES + 1)
            if len(encoded) > MAX_SKILL_STATE_BYTES:
                raise ValueError("skill state exceeds the 1 MiB limit")
            raw = json.loads(
                encoded.decode("utf-8"),
                object_pairs_hook=_skill_state_object_without_duplicates,
                parse_constant=_reject_skill_state_json_constant,
            )
            _validate_skill_state_json_shape(raw)
            disabled = _parse_disabled_skill_state(raw)
        except (
            OSError,
            OverflowError,
            RecursionError,
            UnicodeError,
            ValueError,
        ) as exc:
            self.warnings.append(f"{self.state_file}: invalid state ignored: {safe_text(exc)}")
            return set()
        if os.name != "nt":
            try:
                self.state_file.parent.chmod(0o700)
                self.state_file.chmod(0o600)
            except OSError as exc:
                self.warnings.append(
                    f"{self.state_file}: mode hardening failed: {type(exc).__name__}"
                )
        return disabled

    def _write_disabled(self, disabled: set[str]) -> None:
        if len(disabled) > MAX_DISABLED_SKILLS or any(
            not _VALID_NAME.fullmatch(name) for name in disabled
        ):
            raise ValueError("Invalid or excessive disabled Skill state")
        reject_symlink_components(self.state_file, "Skill state")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        reject_symlink_components(self.state_file, "Skill state")
        if os.name != "nt":
            self.state_file.parent.chmod(0o700)
        encoded = (
            json.dumps({"disabled": sorted(disabled)}, ensure_ascii=False, indent=2) + "\n"
        ).encode()
        temporary = self.state_file.parent / f".skills.{secrets.token_hex(8)}.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.state_file)
            if os.name != "nt":
                self.state_file.chmod(0o600)
                directory = os.open(self.state_file.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    @contextmanager
    def _state_lock(self) -> Any:
        reject_symlink_components(self.state_file.parent, "Skill state")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        reject_symlink_components(self.state_file.parent, "Skill state")
        if os.name != "nt":
            self.state_file.parent.chmod(0o700)
        lock_file = self.state_file.parent / ".skills.lock"
        reject_symlink_components(lock_file, "Skill state lock")
        descriptor = os.open(
            lock_file,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
        locked = False
        try:
            with _SKILL_STATE_THREAD_LOCK:
                if os.name == "posix":
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                    locked = True
                elif os.name == "nt":  # pragma: no cover - exercised on Windows CI
                    import msvcrt

                    if os.fstat(descriptor).st_size == 0:
                        os.write(descriptor, b"\0")
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined]
                    locked = True
                yield
        finally:
            if locked and os.name == "posix":
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
            elif locked and os.name == "nt":  # pragma: no cover - Windows CI
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
            os.close(descriptor)


def _skill_state_object_without_duplicates(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate Skill state key: {key}")
        result[key] = value
    return result


def _reject_skill_state_json_constant(value: str) -> Any:
    raise ValueError(f"Non-standard JSON constant is not allowed: {value}")


def _validate_skill_state_json_shape(value: Any) -> None:
    stack: list[tuple[Any, int]] = [(value, 0)]
    visited = 0
    while stack:
        current, depth = stack.pop()
        visited += 1
        if visited > MAX_SKILL_STATE_JSON_NODES:
            raise ValueError("Skill state JSON exceeds the node limit")
        if depth > MAX_SKILL_STATE_JSON_DEPTH:
            raise ValueError("Skill state JSON exceeds the nesting limit")
        children: Iterable[Any]
        if isinstance(current, dict):
            children = current.values()
        elif isinstance(current, list):
            children = current
        else:
            continue
        child_count = len(current)
        if visited + len(stack) + child_count > MAX_SKILL_STATE_JSON_NODES:
            raise ValueError("Skill state JSON exceeds the node limit")
        stack.extend((child, depth + 1) for child in children)


def _parse_disabled_skill_state(raw: Any) -> set[str]:
    if not isinstance(raw, dict):
        raise ValueError("Skill state root must be an object")
    if "disabled" in raw:
        values = raw["disabled"]
        if (
            not isinstance(values, list)
            or len(values) > MAX_DISABLED_SKILLS
            or any(not isinstance(item, str) or not _VALID_NAME.fullmatch(item) for item in values)
        ):
            raise ValueError("Invalid or excessive disabled Skill state")
        return set(values)
    # Migrate the original Python {name: enabled} representation.
    if len(raw) > MAX_DISABLED_SKILLS:
        raise ValueError("Invalid or excessive legacy Skill state")
    disabled: set[str] = set()
    for name, enabled in raw.items():
        if not _VALID_NAME.fullmatch(name) or not isinstance(enabled, bool):
            raise ValueError("Invalid legacy Skill state")
        if not enabled:
            disabled.add(name)
    return disabled


def _bounded_reference_files(root: Path) -> list[str]:
    pending: list[tuple[Path, int]] = [(root, 0)]
    result: list[str] = []
    scanned = 0
    while pending and scanned < MAX_SKILL_REFERENCE_SCAN_ENTRIES:
        directory, depth = pending.pop()
        remaining = MAX_SKILL_REFERENCE_SCAN_ENTRIES - scanned
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(islice(iterator, remaining), key=lambda item: item.name)
        except OSError:
            continue
        scanned += len(entries)
        child_directories: list[Path] = []
        for entry in entries:
            try:
                if entry.is_symlink():
                    continue
                candidate = Path(entry.path)
                if entry.is_file(follow_symlinks=False):
                    resolved = candidate.resolve()
                    resolved.relative_to(root)
                    result.append(candidate.relative_to(root).as_posix())
                elif depth < MAX_SKILL_REFERENCE_SCAN_DEPTH and entry.is_dir(follow_symlinks=False):
                    child_directories.append(candidate)
            except (OSError, ValueError):
                continue
        pending.extend((child, depth + 1) for child in reversed(child_directories))
    return sorted(result)[:MAX_SKILL_REFERENCES]


def handle_skill_command(payload: str | None, registry: SkillRegistry, agent: Any = None) -> str:
    normalized = (payload or "list").strip()
    operation, _, name = normalized.partition(" ")
    operation = operation.casefold() or "list"
    name = name.strip()
    if operation == "list":
        if not registry.skills:
            return "No skills discovered."
        return "\n".join(
            f"{skill.name}: {'on' if skill.enabled else 'off'} [{skill.source}]"
            f"{' v' + skill.version if skill.version else ''} — {skill.description}"
            for skill in registry.skills.values()
        )
    if operation == "reload":
        registry.reload()
        refresh_agent_skill_index(agent, registry)
        return f"Reloaded {len(registry.skills)} skills."
    if operation == "install":
        from .skill_installer import (
            SkillInstallError,
            install_skill,
            parse_skill_install_request,
        )

        try:
            request = parse_skill_install_request(name)
            result = install_skill(registry, request)
        except SkillInstallError as exc:
            return f"Skill install failed: {safe_text(exc)}"
        refresh_agent_skill_index(agent, registry)
        action = "Replaced" if result.replaced else "Installed"
        active_note = (
            "" if result.active else " A higher-priority Skill with this name remains active."
        )
        return (
            f"{action} Skill {result.name} [{result.scope}] from {result.source} at "
            f"{result.target}.{active_note}"
        )
    if operation == "show" and name in registry.skills:
        skill = registry.skills[name]
        references = "\n".join(f"- {item}" for item in skill.references())
        return (
            f"Skill: {skill.name} [{skill.source}]"
            f"{' v' + skill.version if skill.version else ''}\n"
            f"Path: {skill.path}\n"
            + (f"References:\n{references}\n" if references else "")
            + "\n"
            + skill.load()
        )
    if operation in {"on", "off"} and name:
        try:
            registry.set_enabled(name, operation == "on")
        except KeyError:
            return f"Unknown skill: {name}"
        refresh_agent_skill_index(agent, registry)
        return f"Skill {name}: {operation}"
    return "Usage: /skill [list|reload|show NAME|on NAME|off NAME|install SOURCE [OPTIONS]]"


def refresh_agent_skill_index(agent: Any, registry: SkillRegistry) -> None:
    if agent is None:
        return
    base = re.sub(
        r"\n*<available_skills>.*?</available_skills>",
        "",
        str(agent.base_system_prompt),
        flags=re.S,
    ).rstrip()
    index = registry.index()
    if index:
        base += f"\n\n<available_skills>\n{index}\n</available_skills>"
    agent.base_system_prompt = base
    agent.system_prompt = base


def parse_skill_document(content: str | None) -> FrontmatterResult:
    if content is None:
        return FrontmatterResult({}, "", ("SKILL.md content is null",))
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.startswith("---\n"):
        return FrontmatterResult({}, normalized, ("Missing opening frontmatter marker ---",))
    lines = normalized.splitlines(keepends=True)
    closing = next(
        (index for index, line in enumerate(lines[1:], 1) if line.rstrip("\n") == "---"),
        None,
    )
    if closing is None:
        return FrontmatterResult({}, normalized, ("Missing closing frontmatter marker ---",))
    header = "".join(lines[1:closing])
    body = "".join(lines[closing + 1 :])
    metadata, warnings = _parse_header(header)
    return FrontmatterResult(metadata, body, tuple(warnings))


def parse_frontmatter(content: str) -> dict[str, Any]:
    """Compatibility wrapper returning only parsed metadata."""
    return parse_skill_document(content).metadata


def _parse_header(header: str) -> tuple[dict[str, Any], list[str]]:
    lines = header.splitlines()
    result: dict[str, Any] = {}
    warnings: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip() or line.lstrip().startswith("#"):
            index += 1
            continue
        colon = _unquoted_colon(line)
        if colon < 0:
            warnings.append(f"Unparseable frontmatter line: {line}")
            index += 1
            continue
        key = line[:colon].strip()
        raw_value = line[colon + 1 :].strip()
        if not key or not raw_value:
            warnings.append(f"Frontmatter field has no supported value: {line}")
            index += 1
            continue
        if raw_value.startswith("{") or raw_value.startswith(("&", "*", "!!")):
            warnings.append(f"Unsupported nested or advanced YAML field: {key}")
            index += 1
            continue
        if raw_value.startswith("|"):
            index += 1
            block: list[str] = []
            base_indent: int | None = None
            while index < len(lines):
                candidate = lines[index]
                if not candidate.strip():
                    block.append("")
                    index += 1
                    continue
                indent = len(candidate) - len(candidate.lstrip(" "))
                if indent == 0:
                    break
                base_indent = indent if base_indent is None else base_indent
                if indent < base_indent:
                    break
                block.append(candidate[base_indent:])
                index += 1
            result[key] = " ".join(" ".join(block).split())
            continue
        if raw_value.startswith("[") and raw_value.endswith("]"):
            result[key] = [
                _unquote(item.strip()) for item in raw_value[1:-1].split(",") if item.strip()
            ]
        else:
            result[key] = _unquote(raw_value)
        index += 1
    return result, warnings


def _unquoted_colon(value: str) -> int:
    single = False
    double = False
    for index, character in enumerate(value):
        if character == "'" and not double:
            single = not single
        elif character == '"' and not single:
            double = not double
        elif character == ":" and not single and not double:
            return index
    return -1


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _path_uses_symlink(path: Path, trusted_base: Path) -> bool:
    try:
        relative = Path(os.path.abspath(path)).relative_to(trusted_base.resolve())
    except ValueError:
        return True
    current = trusted_base.resolve()
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return True
        if not current.exists():
            break
    return False
