from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlsplit, urlunsplit

from .paths import reject_symlink_components
from .text_safety import safe_text

if TYPE_CHECKING:
    from .skills import SkillRegistry


MAX_INSTALL_FILES = 5_000
MAX_INSTALL_DEPTH = 16
MAX_INSTALL_FILE_BYTES = 8 * 1024 * 1024
MAX_INSTALL_TOTAL_BYTES = 64 * 1024 * 1024
MAX_GIT_ERROR_BYTES = 64 * 1024
GIT_TIMEOUT_SECONDS = 60
INSTALL_METADATA_FILE = ".kairocli-install.json"
CURATED_SKILLS_REPOSITORY = "https://github.com/openai/skills.git"
_VALID_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_GITHUB_SHORTHAND = re.compile(
    r"^(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/(?P<repo>[A-Za-z0-9._-]{1,100})$"
)
_VALID_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,254}$")


class SkillInstallError(ValueError):
    """A safe, user-facing Skill installation failure."""


@dataclass(frozen=True, slots=True)
class SkillInstallRequest:
    source: str
    scope: str = "user"
    ref: str | None = None
    subdirectory: str | None = None
    name: str | None = None
    force: bool = False


@dataclass(frozen=True, slots=True)
class SkillSource:
    kind: str
    location: str
    display: str
    ref: str | None = None
    subdirectory: str | None = None


@dataclass(frozen=True, slots=True)
class SkillInstallResult:
    name: str
    scope: str
    source: str
    target: Path
    replaced: bool
    active: bool


def parse_skill_install_request(arguments: str) -> SkillInstallRequest:
    try:
        tokens = shlex.split(arguments)
    except ValueError as exc:
        raise SkillInstallError(f"Invalid install arguments: {safe_text(exc)}") from exc
    if not tokens:
        raise SkillInstallError(_install_usage())

    scope = "user"
    ref: str | None = None
    subdirectory: str | None = None
    name: str | None = None
    force = False
    sources: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {"--project", "--user"}:
            scope = token[2:]
        elif token == "--force":
            force = True
        elif token in {"--scope", "--ref", "--path", "--name"}:
            index += 1
            if index >= len(tokens):
                raise SkillInstallError(f"Missing value for {token}.\n{_install_usage()}")
            value = tokens[index]
            if token == "--scope":
                scope = value.casefold()
            elif token == "--ref":
                ref = value
            elif token == "--path":
                subdirectory = value
            else:
                name = value
        elif token.startswith("--"):
            raise SkillInstallError(f"Unknown install option: {token}\n{_install_usage()}")
        else:
            sources.append(token)
        index += 1

    if len(sources) != 1:
        raise SkillInstallError("Skill install requires exactly one source.\n" + _install_usage())
    if scope not in {"user", "project"}:
        raise SkillInstallError("Skill install scope must be user or project.")
    if name is not None and not _VALID_NAME.fullmatch(name):
        raise SkillInstallError("Installed Skill name is invalid.")
    validated_ref = _validate_ref(ref) if ref is not None else None
    validated_path = _validate_subdirectory(subdirectory) if subdirectory is not None else None
    return SkillInstallRequest(
        source=sources[0],
        scope=scope,
        ref=validated_ref,
        subdirectory=validated_path,
        name=name,
        force=force,
    )


def install_skill(registry: SkillRegistry, request: SkillInstallRequest) -> SkillInstallResult:
    try:
        return _install_skill(registry, request)
    except SkillInstallError:
        raise
    except (OSError, ValueError) as exc:
        raise SkillInstallError(f"Skill installation failed: {safe_text(exc)}") from exc


def _install_skill(registry: SkillRegistry, request: SkillInstallRequest) -> SkillInstallResult:
    request = _validate_install_request(request)
    source = resolve_skill_source(request)
    destination_root = (
        registry.paths.user_dir / "skills"
        if request.scope == "user"
        else registry.paths.project_dir / "skills"
    )
    with registry._state_lock():  # noqa: SLF001 - installation shares Skill state serialization
        _prepare_destination_root(destination_root, request.scope)
        if source.kind == "local":
            candidate = _local_candidate(source)
            name, target, replaced = _install_candidate(
                candidate,
                destination_root,
                source,
                request,
            )
        else:
            with tempfile.TemporaryDirectory(prefix="kairocli-skill-") as temporary:
                repository = Path(temporary) / "repository"
                _checkout_git_source(source, repository)
                candidate = _candidate_from_root(repository, source.subdirectory)
                name, target, replaced = _install_candidate(
                    candidate,
                    destination_root,
                    source,
                    request,
                )

    registry.reload()
    installed = registry.skills.get(name)
    if installed is not None and installed.path == target / "SKILL.md" and not installed.enabled:
        registry.set_enabled(name, True)
        registry.reload()
        installed = registry.skills.get(name)
    active = installed is not None and installed.path == target / "SKILL.md" and installed.enabled
    return SkillInstallResult(
        name=name,
        scope=request.scope,
        source=source.display,
        target=target,
        replaced=replaced,
        active=active,
    )


def resolve_skill_source(request: SkillInstallRequest) -> SkillSource:
    request = _validate_install_request(request)
    raw = request.source.strip()
    if not raw or len(raw) > 4_096 or any(ord(character) < 32 for character in raw):
        raise SkillInstallError("Skill source is empty, excessive, or contains control characters.")

    local = Path(raw).expanduser()
    if local.exists() or raw.startswith((".", "/", "~")):
        if request.ref is not None:
            raise SkillInstallError("--ref is only valid for Git sources.")
        return SkillSource(
            kind="local",
            location=str(local),
            display=raw,
            subdirectory=request.subdirectory,
        )

    if _VALID_NAME.fullmatch(raw):
        if request.subdirectory is not None:
            raise SkillInstallError("--path cannot be combined with a curated Skill name.")
        return SkillSource(
            kind="git",
            location=CURATED_SKILLS_REPOSITORY,
            display=f"openai/skills:{raw}",
            ref=request.ref or "main",
            subdirectory=f"skills/.curated/{raw}",
        )

    shorthand = _GITHUB_SHORTHAND.fullmatch(raw.removesuffix(".git"))
    if shorthand is not None:
        repository = shorthand.group("repo").removesuffix(".git")
        return SkillSource(
            kind="git",
            location=f"https://github.com/{shorthand.group('owner')}/{repository}.git",
            display=raw,
            ref=request.ref,
            subdirectory=request.subdirectory,
        )

    parsed = urlsplit(raw)
    if parsed.scheme.casefold() != "https" or not parsed.hostname:
        raise SkillInstallError(
            "Skill source must be a local directory, curated name, GitHub owner/repo, "
            "or HTTPS Git URL."
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise SkillInstallError("Git URLs cannot contain credentials, a query, or a fragment.")
    if parsed.port not in {None, 443}:
        raise SkillInstallError("HTTPS Git URLs must use the default HTTPS port.")

    url_ref: str | None = None
    url_path: str | None = None
    clean_path = parsed.path
    if parsed.hostname.casefold() == "github.com":
        segments = [unquote(item) for item in parsed.path.split("/") if item]
        if len(segments) >= 4 and segments[2] == "tree":
            url_ref = _validate_ref(segments[3])
            url_path = _validate_subdirectory("/".join(segments[4:])) if len(segments) > 4 else None
            clean_path = "/" + "/".join(segments[:2]) + ".git"
        elif len(segments) != 2:
            raise SkillInstallError("GitHub URLs must identify a repository or a tree path.")
        elif not clean_path.endswith(".git"):
            clean_path += ".git"
    location = urlunsplit(("https", parsed.netloc, clean_path, "", ""))
    return SkillSource(
        kind="git",
        location=location,
        display=raw,
        ref=request.ref or url_ref,
        subdirectory=request.subdirectory or url_path,
    )


def _local_candidate(source: SkillSource) -> Path:
    lexical_root = Path(source.location)
    if lexical_root.is_symlink():
        raise SkillInstallError("Local Skill source cannot be a symlink.")
    root = lexical_root.resolve()
    if not root.is_dir():
        raise SkillInstallError(f"Local Skill source is not a directory: {source.display}")
    return _candidate_from_root(root, source.subdirectory)


def _candidate_from_root(root: Path, subdirectory: str | None) -> Path:
    candidate = root
    if subdirectory is not None:
        for part in PurePosixPath(subdirectory).parts:
            candidate /= part
            if candidate.is_symlink():
                raise SkillInstallError("Skill source path cannot traverse a symlink.")
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise SkillInstallError("Skill path escapes its source root.") from exc
    if candidate.is_symlink() or not candidate.is_dir():
        raise SkillInstallError("Skill source path is not a regular directory.")
    if not (candidate / "SKILL.md").is_file() or (candidate / "SKILL.md").is_symlink():
        raise SkillInstallError("Skill source must contain a regular SKILL.md file at its root.")
    return candidate


def _checkout_git_source(source: SkillSource, repository: Path) -> None:
    if shutil.which("git") is None:
        raise SkillInstallError("Git is required to install a remote Skill.")
    _run_git(
        "clone",
        "--no-checkout",
        "--depth",
        "1",
        "--filter=blob:none",
        source.location,
        str(repository),
    )
    if source.subdirectory is not None:
        _run_git("-C", str(repository), "sparse-checkout", "init", "--no-cone")
        _run_git("-C", str(repository), "sparse-checkout", "set", "--", source.subdirectory)
    # The curated catalog defaults to the repository's main branch, which the
    # shallow clone has already selected. Fetching it again can consume a second
    # full network timeout and outlive the registry's bounded tool execution.
    curated_default_ref = source.location == CURATED_SKILLS_REPOSITORY and source.ref == "main"
    if source.ref is not None and not curated_default_ref:
        _run_git("-C", str(repository), "fetch", "--depth", "1", "origin", source.ref)
        _run_git("-C", str(repository), "checkout", "--detach", "FETCH_HEAD")
    else:
        _run_git("-C", str(repository), "checkout", "--detach", "HEAD")


def _run_git(*arguments: str) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_ASKPASS": "",
            "GIT_OPTIONAL_LOCKS": "0",
        }
    )
    with tempfile.TemporaryFile() as error_stream:
        try:
            process = subprocess.Popen(  # noqa: S603 - fixed executable and validated arguments
                [
                    "git",
                    "-c",
                    "credential.helper=",
                    "-c",
                    "core.hooksPath=/dev/null",
                    "-c",
                    "filter.lfs.smudge=",
                    "-c",
                    "filter.lfs.required=false",
                    *arguments,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=error_stream,
                env=environment,
            )
            try:
                return_code = process.wait(timeout=GIT_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired as exc:
                process.kill()
                process.wait()
                raise SkillInstallError("Git operation timed out.") from exc
        except OSError as exc:
            raise SkillInstallError(f"Git could not be started: {type(exc).__name__}") from exc
        if return_code == 0:
            return
        error_stream.seek(0)
        detail = error_stream.read(MAX_GIT_ERROR_BYTES + 1)
    rendered = detail[:MAX_GIT_ERROR_BYTES].decode("utf-8", errors="replace").strip()
    if len(detail) > MAX_GIT_ERROR_BYTES:
        rendered += " ...[truncated]"
    detail_text = f": {safe_text(rendered)}" if rendered else "."
    raise SkillInstallError("Git operation failed" + detail_text)


def _install_candidate(
    candidate: Path,
    destination_root: Path,
    source: SkillSource,
    request: SkillInstallRequest,
) -> tuple[str, Path, bool]:
    from .skills import MAX_SKILL_FILE_BYTES, parse_skill_document

    skill_bytes = _read_bounded_file(candidate / "SKILL.md", MAX_SKILL_FILE_BYTES)
    parsed = parse_skill_document(skill_bytes.decode("utf-8", errors="replace"))
    metadata_name = str(parsed.metadata.get("name") or candidate.name).strip()
    name = request.name or metadata_name
    if not _VALID_NAME.fullmatch(name):
        raise SkillInstallError(
            "SKILL.md name and source directory do not provide a valid Skill name."
        )

    target = destination_root / name
    reject_symlink_components(target, "Installed Skill")
    replaced = target.exists()
    if replaced and not request.force:
        raise SkillInstallError(
            f"Skill {name} is already installed in {request.scope} scope; use --force."
        )
    if replaced and (target.is_symlink() or not target.is_dir()):
        raise SkillInstallError("Existing Skill target is not a regular directory.")

    staging = destination_root.parent / f".skill-install-{secrets.token_hex(12)}.tmp"
    backup = destination_root.parent / f".skill-replace-{secrets.token_hex(12)}.tmp"
    try:
        staging.mkdir(mode=0o700 if request.scope == "user" else 0o755)
        digest = _copy_validated_tree(candidate, staging, private=request.scope == "user")
        metadata = {
            "source": source.display,
            "ref": source.ref,
            "path": source.subdirectory,
            "installed_at": datetime.now(UTC).isoformat(),
            "sha256": digest,
        }
        _write_metadata(staging / INSTALL_METADATA_FILE, metadata, private=request.scope == "user")
        _fsync_directory(staging)
        if replaced:
            os.replace(target, backup)
        try:
            os.replace(staging, target)
            _fsync_directory(destination_root)
        except BaseException:
            if replaced and backup.exists() and not target.exists():
                os.replace(backup, target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return name, target, replaced
    except SkillInstallError:
        raise
    except (OSError, ValueError) as exc:
        raise SkillInstallError(f"Skill installation failed: {safe_text(exc)}") from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if backup.exists() and target.exists():
            shutil.rmtree(backup, ignore_errors=True)


def _copy_validated_tree(source: Path, target: Path, *, private: bool) -> str:
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    pending = [(source, target, 0)]
    while pending:
        source_directory, target_directory, depth = pending.pop()
        if depth > MAX_INSTALL_DEPTH:
            raise SkillInstallError(f"Skill directory exceeds {MAX_INSTALL_DEPTH} levels.")
        try:
            entries = sorted(os.scandir(source_directory), key=lambda item: item.name)
        except OSError as exc:
            raise SkillInstallError(
                f"Skill directory cannot be read: {type(exc).__name__}"
            ) from exc
        for entry in entries:
            if entry.is_symlink():
                raise SkillInstallError(f"Skill source contains a symlink: {entry.name}")
            if entry.name == INSTALL_METADATA_FILE:
                continue
            if entry.name == ".git":
                if source_directory == source:
                    continue
                raise SkillInstallError("Skill source contains a nested .git directory.")
            file_count += 1
            if file_count > MAX_INSTALL_FILES:
                raise SkillInstallError(f"Skill contains more than {MAX_INSTALL_FILES} entries.")
            source_entry = Path(entry.path)
            target_entry = target_directory / entry.name
            if entry.is_dir(follow_symlinks=False):
                target_entry.mkdir(mode=0o700 if private else 0o755)
                pending.append((source_entry, target_entry, depth + 1))
                continue
            if not entry.is_file(follow_symlinks=False):
                raise SkillInstallError(f"Skill source contains a special file: {entry.name}")
            source_stat = entry.stat(follow_symlinks=False)
            if source_stat.st_size > MAX_INSTALL_FILE_BYTES:
                raise SkillInstallError(
                    f"Skill file exceeds {MAX_INSTALL_FILE_BYTES} bytes: {entry.name}"
                )
            total_bytes += source_stat.st_size
            if total_bytes > MAX_INSTALL_TOTAL_BYTES:
                raise SkillInstallError(f"Skill exceeds {MAX_INSTALL_TOTAL_BYTES} total bytes.")
            relative = (
                source_entry.relative_to(source)
                .as_posix()
                .encode("utf-8", errors="surrogatepass")
            )
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(source_entry, flags)
            executable = bool(source_stat.st_mode & stat.S_IXUSR)
            mode = (0o700 if executable else 0o600) if private else (0o755 if executable else 0o644)
            try:
                opened_stat = os.fstat(descriptor)
                if not stat.S_ISREG(opened_stat.st_mode) or (
                    opened_stat.st_dev,
                    opened_stat.st_ino,
                ) != (source_stat.st_dev, source_stat.st_ino):
                    raise SkillInstallError("Skill source changed while it was being installed.")
                output = os.open(target_entry, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
                try:
                    copied = 0
                    while True:
                        chunk = os.read(descriptor, 64 * 1024)
                        if not chunk:
                            break
                        copied += len(chunk)
                        if copied > MAX_INSTALL_FILE_BYTES:
                            raise SkillInstallError("Skill file grew while it was being installed.")
                        digest.update(chunk)
                        _write_all(output, chunk)
                    if copied != source_stat.st_size:
                        raise SkillInstallError("Skill file changed while it was being installed.")
                    os.fsync(output)
                finally:
                    os.close(output)
            finally:
                os.close(descriptor)
    return digest.hexdigest()


def _read_bounded_file(path: Path, limit: int) -> bytes:
    if path.is_symlink():
        raise SkillInstallError("SKILL.md cannot be a symlink.")
    try:
        with path.open("rb") as stream:
            content = stream.read(limit + 1)
    except OSError as exc:
        raise SkillInstallError(f"SKILL.md cannot be read: {type(exc).__name__}") from exc
    if len(content) > limit:
        raise SkillInstallError(f"SKILL.md exceeds {limit} bytes.")
    return content


def _write_metadata(path: Path, value: dict[str, str | None], *, private: bool) -> None:
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 if private else 0o644)
    try:
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write")
        view = view[written:]


def _prepare_destination_root(root: Path, scope: str) -> None:
    reject_symlink_components(root, "Skill install root")
    root.mkdir(parents=True, exist_ok=True)
    reject_symlink_components(root, "Skill install root")
    if os.name != "nt":
        root.chmod(0o700 if scope == "user" else 0o755)


def _validate_ref(value: str) -> str:
    if (
        not _VALID_REF.fullmatch(value)
        or ".." in value
        or "//" in value
        or "@{" in value
        or value.endswith(("/", ".", ".lock"))
    ):
        raise SkillInstallError("Git ref is invalid or unsafe.")
    return value


def _validate_install_request(request: SkillInstallRequest) -> SkillInstallRequest:
    if request.scope not in {"user", "project"}:
        raise SkillInstallError("Skill install scope must be user or project.")
    if request.name is not None and not _VALID_NAME.fullmatch(request.name):
        raise SkillInstallError("Installed Skill name is invalid.")
    if not isinstance(request.force, bool):
        raise SkillInstallError("Skill install force must be a boolean.")
    ref = _validate_ref(request.ref) if request.ref is not None else None
    subdirectory = (
        _validate_subdirectory(request.subdirectory)
        if request.subdirectory is not None
        else None
    )
    return SkillInstallRequest(
        source=request.source,
        scope=request.scope,
        ref=ref,
        subdirectory=subdirectory,
        name=request.name,
        force=request.force,
    )


def _validate_subdirectory(value: str) -> str:
    if not value or len(value) > 1_024 or "\\" in value or any(ord(item) < 32 for item in value):
        raise SkillInstallError("Skill source path is invalid.")
    path = PurePosixPath(value)
    invalid_part = any(part in {"", ".", ".git"} for part in path.parts)
    if path.is_absolute() or ".." in path.parts or invalid_part or value.startswith("-"):
        raise SkillInstallError("Skill source path must be a safe relative path.")
    return path.as_posix()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":  # pragma: no cover - directory fsync is POSIX-only
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _install_usage() -> str:
    return (
        "Usage: /skill install SOURCE [--user|--project] [--ref REF] [--path PATH] "
        "[--name NAME] [--force]"
    )
