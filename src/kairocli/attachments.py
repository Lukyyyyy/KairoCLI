from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from .paths import reject_symlink_components

MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024
MAX_ATTACHMENT_SUMMARY_CHARS = 500
ATTACHMENT_ID_RE = re.compile(r"img_[0-9a-f]{24}\Z")
ATTACHMENT_REF_RE = re.compile(r'\bid="(img_[0-9a-f]{24})"')
CURRENT_ATTACHMENTS_RE = re.compile(
    r"\n*<current_attachments>.*?</current_attachments>", re.DOTALL
)
DATA_IMAGE_RE = re.compile(
    r"data:(image/[A-Za-z0-9.+-]+);base64,([A-Za-z0-9+/=]+)\Z"
)


@dataclass(frozen=True, slots=True)
class ImageAttachment:
    id: str
    media_type: str
    size_bytes: int
    summary: str = ""


class AttachmentStore:
    def __init__(self, root: Path, workspace: Path) -> None:
        self.root = root
        self.workspace_id = hashlib.sha256(str(workspace.resolve()).encode()).hexdigest()[:24]
        self._allowed: set[str] = set()
        self._lock = threading.Lock()

    def remember_data_url(self, data_url: str) -> ImageAttachment:
        match = DATA_IMAGE_RE.fullmatch(data_url)
        if match is None:
            raise ValueError("Attachment is not a supported image data URL")
        try:
            data = base64.b64decode(match.group(2), validate=True)
        except ValueError as exc:
            raise ValueError("Attachment contains invalid base64 data") from exc
        if not data or len(data) > MAX_ATTACHMENT_BYTES:
            raise ValueError("Attachment exceeds the 5 MiB stored-image limit")
        attachment_id = "img_" + hashlib.sha256(data).hexdigest()[:24]
        with self._lock:
            self._ensure_root()
            blob = self.root / f"{attachment_id}.bin"
            metadata = self.root / f"{attachment_id}.json"
            if not blob.exists():
                self._atomic_write(blob, data)
            if not metadata.exists():
                self._write_metadata(
                    metadata,
                    {
                        "id": attachment_id,
                        "media_type": match.group(1),
                        "size_bytes": len(data),
                        "summary": "",
                        "workspace_id": self.workspace_id,
                    },
                )
            self._allowed.add(attachment_id)
            return self._load(attachment_id)

    def authorize(self, attachment_ids: list[str]) -> None:
        with self._lock:
            for attachment_id in attachment_ids:
                if not ATTACHMENT_ID_RE.fullmatch(attachment_id):
                    continue
                try:
                    self._load(attachment_id)
                except (OSError, ValueError):
                    continue
                self._allowed.add(attachment_id)

    def save_summaries(self, values: list[dict[str, str]]) -> int:
        saved = 0
        with self._lock:
            for value in values:
                attachment_id = value.get("attachment_id", "")
                summary = " ".join(value.get("summary", "").split())
                if attachment_id not in self._allowed:
                    raise ValueError(
                        f"Attachment is not available in this conversation: {attachment_id}"
                    )
                if not summary or len(summary) > MAX_ATTACHMENT_SUMMARY_CHARS:
                    raise ValueError("Attachment summary must contain 1 to 500 characters")
                record = self._load_dict(attachment_id)
                if str(record.get("summary", "")).strip():
                    continue
                record["summary"] = summary
                self._write_metadata(self.root / f"{attachment_id}.json", record)
                saved += 1
        return saved

    def read_data_url(self, attachment_id: str) -> tuple[ImageAttachment, str]:
        with self._lock:
            if attachment_id not in self._allowed:
                raise ValueError(
                    f"Attachment is not available in this conversation: {attachment_id}"
                )
            record = self._load(attachment_id)
            blob = self.root / f"{attachment_id}.bin"
            if blob.is_symlink() or not blob.is_file():
                raise ValueError("Attachment data is unavailable")
            data = blob.read_bytes()
            if len(data) != record.size_bytes or len(data) > MAX_ATTACHMENT_BYTES:
                raise ValueError("Attachment data failed integrity validation")
            expected = "img_" + hashlib.sha256(data).hexdigest()[:24]
            if expected != attachment_id:
                raise ValueError("Attachment data failed integrity validation")
            encoded = base64.b64encode(data).decode("ascii")
            return record, f"data:{record.media_type};base64,{encoded}"

    def references_markup(
        self, attachments: list[ImageAttachment], tag: str = "historical_attachments"
    ) -> str:
        lines = [f"<{tag}>"]
        for index, attachment in enumerate(attachments, 1):
            summary = attachment.summary or "Summary unavailable; inspect the image if needed."
            lines.append(
                f'  <image id="{attachment.id}" label="Image {index}" '
                f'type="{html.escape(attachment.media_type, quote=True)}">'
                f"{html.escape(summary, quote=False)}</image>"
            )
        lines.append(f"</{tag}>")
        return "\n".join(lines)

    def _load(self, attachment_id: str) -> ImageAttachment:
        record = self._load_dict(attachment_id)
        size_bytes = record["size_bytes"]
        if type(size_bytes) is not int:
            raise ValueError("Attachment metadata is invalid")
        return ImageAttachment(
            attachment_id,
            str(record["media_type"]),
            size_bytes,
            str(record.get("summary", "")),
        )

    def _load_dict(self, attachment_id: str) -> dict[str, object]:
        if not ATTACHMENT_ID_RE.fullmatch(attachment_id):
            raise ValueError("Invalid attachment identifier")
        path = self.root / f"{attachment_id}.json"
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 4_096:
            raise ValueError("Attachment metadata is unavailable")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(raw, dict)
            or raw.get("id") != attachment_id
            or raw.get("workspace_id") != self.workspace_id
            or not str(raw.get("media_type", "")).startswith("image/")
            or type(raw.get("size_bytes")) is not int
            or not 0 < int(raw["size_bytes"]) <= MAX_ATTACHMENT_BYTES
        ):
            raise ValueError("Attachment metadata is invalid")
        return raw

    def _ensure_root(self) -> None:
        reject_symlink_components(self.root, "Attachment storage")
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            self.root.chmod(0o700)
        except OSError:
            pass

    def _write_metadata(self, path: Path, value: dict[str, object]) -> None:
        self._atomic_write(
            path,
            json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(),
        )

    def _atomic_write(self, path: Path, data: bytes) -> None:
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=self.root)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
