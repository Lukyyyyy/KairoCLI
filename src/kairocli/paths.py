from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .brand import PROJECT_DIR_NAME, PROJECT_LOCAL_MEMORY_FILE, PROJECT_MEMORY_FILE


def reject_symlink_components(path: Path, label: str) -> None:
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise ValueError(
                f"{label} cannot use a symlink component: {candidate.name or candidate}"
            )


@dataclass(frozen=True, slots=True)
class KairoPaths:
    home: Path
    workspace: Path

    @classmethod
    def discover(cls, workspace: Path | None = None, home: Path | None = None) -> KairoPaths:
        return cls((home or Path.home()).resolve(), (workspace or Path.cwd()).resolve())

    @property
    def user_dir(self) -> Path:
        return self.home / ".kairocli"

    @property
    def user_dotenv(self) -> Path:
        return self.user_dir / ".env"

    @property
    def project_dir(self) -> Path:
        return self.workspace / PROJECT_DIR_NAME

    @property
    def config_file(self) -> Path:
        return self.user_dir / "config.json"

    @property
    def pricing_file(self) -> Path:
        return self.user_dir / "pricing.json"

    @property
    def memory_file(self) -> Path:
        return self.user_dir / "memory" / "long_term_memory.json"

    @property
    def memory_embeddings_file(self) -> Path:
        return self.user_dir / "memory" / "embeddings.db"

    @property
    def task_database(self) -> Path:
        return self.user_dir / "tasks" / "tasks.db"

    @property
    def session_database(self) -> Path:
        return self.user_dir / "sessions" / "sessions.db"

    @property
    def runtime_dir(self) -> Path:
        return self.user_dir / "runtime"

    @property
    def audit_dir(self) -> Path:
        return self.user_dir / "audit"

    @property
    def history_file(self) -> Path:
        return self.user_dir / "history" / "input.history"

    @property
    def export_dir(self) -> Path:
        return self.user_dir / "exports"

    @property
    def snapshot_dir(self) -> Path:
        parent = self.workspace.parent
        parent_digest = hashlib.sha256(str(parent).encode()).hexdigest()[:16]
        workspace_digest = hashlib.sha256(str(self.workspace).encode()).hexdigest()[:16]
        return self.user_dir / "snapshots" / parent_digest / workspace_digest / ".git"

    def project_memory_candidates(self) -> tuple[Path, ...]:
        return (
            self.user_dir / PROJECT_MEMORY_FILE,
            self.workspace / PROJECT_MEMORY_FILE,
            self.project_dir / PROJECT_MEMORY_FILE,
            self.workspace / PROJECT_LOCAL_MEMORY_FILE,
            self.project_dir / PROJECT_LOCAL_MEMORY_FILE,
        )
