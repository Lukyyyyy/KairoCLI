from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from .brand import PROJECT_LOCAL_MEMORY_FILE, PROJECT_MEMORY_FILE
from .paths import KairoPaths

MAX_INSTRUCTION_CHARS = 24_000
MAX_IMPORT_DEPTH = 3
MAX_SCOPED_FILES = 100
MAX_SCOPE_INDEX_CHARS = 8_000
_EXCLUDED_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".kairocli",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
    "target",
}
log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InstructionSource:
    path: Path
    import_root: Path
    scope: Path


class InstructionResolver:
    """Load global, root and subtree-scoped Kairo CLI instruction layers safely."""

    def __init__(
        self,
        paths: KairoPaths,
        max_chars: int = MAX_INSTRUCTION_CHARS,
    ) -> None:
        self.paths = paths
        self.workspace = paths.workspace.resolve()
        self.max_chars = max(1_000, max_chars)

    def base_context(self) -> str:
        return self._render(self._base_sources())

    def for_target(self, target: str | Path) -> str:
        resolved = self._resolve_target(target)
        directory = resolved if resolved.is_dir() else resolved.parent
        sources = self._base_sources()
        for ancestor in _ancestors(self.workspace, directory):
            if ancestor == self.workspace:
                continue
            sources.extend(self._directory_sources(ancestor))
        return self._render(_deduplicate(sources))

    def scoped_index(self) -> str:
        locations = self._scoped_locations()
        if not locations:
            return ""
        rendered = [str(path.relative_to(self.workspace)) for path in locations[:MAX_SCOPED_FILES]]
        suffix = (
            f"\n[Scoped instruction index truncated at {MAX_SCOPED_FILES} files]"
            if len(locations) > MAX_SCOPED_FILES
            else ""
        )
        result = (
            "Nested instruction files apply only to targets under their containing directory. "
            "Before editing in one of these subtrees, call load_project_instructions for the "
            "target path.\n- " + "\n- ".join(rendered) + suffix
        )
        if len(result) <= MAX_SCOPE_INDEX_CHARS:
            return result
        marker = "\n[Scoped instruction index truncated by character budget]"
        return result[: MAX_SCOPE_INDEX_CHARS - len(marker)].rstrip() + marker

    def _base_sources(self) -> list[InstructionSource]:
        sources: list[InstructionSource] = []
        for candidate in self.paths.project_memory_candidates():
            import_root = (
                self.paths.user_dir.resolve()
                if _is_relative_to(candidate.resolve(), self.paths.user_dir.resolve())
                else self.workspace
            )
            sources.append(InstructionSource(candidate.resolve(), import_root, self.workspace))
        return _deduplicate(sources)

    def _directory_sources(self, directory: Path) -> list[InstructionSource]:
        return [
            InstructionSource(directory / PROJECT_MEMORY_FILE, self.workspace, directory),
            InstructionSource(directory / PROJECT_LOCAL_MEMORY_FILE, self.workspace, directory),
        ]

    def _render(self, sources: list[InstructionSource]) -> str:
        sections: list[str] = []
        used = 0
        for source in sources:
            if not source.path.is_file() or source.path.is_symlink():
                continue
            content = self._read_with_imports(source.path, source.import_root, set(), 0).strip()
            if not content:
                continue
            label = self._label(source.path)
            scope = self._scope_label(source.scope)
            section = f"## {label} (scope: {scope})\n{content}"
            separator = "\n\n" if sections else ""
            remaining = self.max_chars - used - len(separator)
            if remaining <= 0:
                break
            if len(section) > remaining:
                marker = f"\n\n[KAIRO.md content truncated at {self.max_chars} characters]"
                if remaining <= len(marker):
                    sections.append(separator + section[:remaining])
                else:
                    keep = remaining - len(marker)
                    sections.append(separator + section[:keep].rstrip() + marker)
                used = self.max_chars
                break
            sections.append(separator + section)
            used += len(separator) + len(section)
        return "".join(sections)

    def _read_with_imports(
        self,
        path: Path,
        import_root: Path,
        stack: set[Path],
        depth: int,
    ) -> str:
        try:
            normalized = path.resolve(strict=True)
        except (OSError, RuntimeError):
            return ""
        if depth > MAX_IMPORT_DEPTH:
            log.warning("Skipping KAIRO.md import beyond depth %s: %s", depth, path)
            return ""
        if not _is_relative_to(normalized, import_root) or not normalized.is_file():
            log.warning("Skipping KAIRO.md import outside its allowed root: %s", path)
            return ""
        if normalized in stack:
            log.warning("Skipping cyclic KAIRO.md import: %s", path)
            return ""
        try:
            with normalized.open("rb") as stream:
                raw = stream.read(self.max_chars + 1)
        except OSError:
            return ""
        source_truncated = len(raw) > self.max_chars
        raw = raw[: self.max_chars]
        if b"\x00" in raw:
            log.warning("Skipping binary KAIRO.md instruction file: %s", path)
            return ""
        stack.add(normalized)
        try:
            output: list[str] = []
            for line in raw.decode("utf-8", errors="replace").splitlines():
                imported = _parse_import(line)
                if imported is None:
                    output.append(line)
                    continue
                imported_content = self._read_with_imports(
                    normalized.parent / imported,
                    import_root,
                    stack,
                    depth + 1,
                ).strip()
                if imported_content:
                    output.append(imported_content)
            result = "\n".join(output)
            if source_truncated:
                result += "\n[KAIRO.md source truncated by per-file character budget]"
            return result
        finally:
            stack.remove(normalized)

    def _scoped_locations(self) -> list[Path]:
        found: list[Path] = []
        for root, directories, files in os.walk(self.workspace, followlinks=False):
            current = Path(root)
            directories[:] = sorted(
                directory
                for directory in directories
                if directory not in _EXCLUDED_DIRECTORIES and not (current / directory).is_symlink()
            )
            if current == self.workspace:
                continue
            for name in (PROJECT_MEMORY_FILE, PROJECT_LOCAL_MEMORY_FILE):
                candidate = current / name
                if name in files and candidate.is_file() and not candidate.is_symlink():
                    found.append(candidate)
                    if len(found) > MAX_SCOPED_FILES:
                        return sorted(found)
        return sorted(found)

    def _resolve_target(self, target: str | Path) -> Path:
        raw = str(target)
        if len(raw) > 4_096:
            raise ValueError("Instruction target exceeds the 4096 character limit")
        if "\x00" in raw:
            raise ValueError("Instruction target contains a NUL byte")
        value = Path(raw)
        candidate = value if value.is_absolute() else self.workspace / value
        resolved = candidate.resolve(strict=False)
        if not _is_relative_to(resolved, self.workspace):
            raise ValueError("Instruction target must stay within the workspace")
        return resolved

    def _label(self, path: Path) -> str:
        if _is_relative_to(path, self.workspace):
            return str(path.relative_to(self.workspace))
        return f"user/{path.name}"

    def _scope_label(self, scope: Path) -> str:
        if scope == self.workspace:
            return "."
        return str(scope.relative_to(self.workspace))


def _parse_import(line: str) -> Path | None:
    value = line.strip()
    if not value.startswith("@") or len(value) < 2 or any(char.isspace() for char in value):
        return None
    raw = value[1:]
    candidate = Path(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    return candidate


def _ancestors(root: Path, target: Path) -> list[Path]:
    relative = target.relative_to(root)
    result = [root]
    current = root
    for part in relative.parts:
        current /= part
        result.append(current)
    return result


def _deduplicate(sources: list[InstructionSource]) -> list[InstructionSource]:
    result: list[InstructionSource] = []
    seen: set[Path] = set()
    for source in sources:
        if source.path in seen:
            continue
        seen.add(source.path)
        result.append(source)
    return result


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
