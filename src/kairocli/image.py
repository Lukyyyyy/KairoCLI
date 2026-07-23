from __future__ import annotations

import asyncio
import base64
import io
import mimetypes
import os
import re
import shutil
import signal
import stat
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from .policy import PathGuard
from .text_safety import safe_text

API_IMAGE_MAX_BASE64_SIZE = 5 * 1024 * 1024
MAX_SOURCE_IMAGE_BYTES = 50 * 1024 * 1024
MAX_SOURCE_IMAGE_PIXELS = 40_000_000
IMAGE_MAX_WIDTH = 2_000
IMAGE_MAX_HEIGHT = 2_000
IMAGE_REFERENCE_PATTERN = re.compile(
    r"@image:(<[^>]+>|[^\s<>\u2010-\u206f\u3000-\u303f\uff00-\uffef]+)"
    r"|@clipboard(?!\w)"
)


@dataclass(frozen=True, slots=True)
class ProcessedImage:
    data: str
    media_type: str
    source_bytes: int
    output_bytes: int
    original_size: tuple[int, int] | None = None
    display_size: tuple[int, int] | None = None
    reencoded: bool = False
    source_path: Path | None = None

    def data_url(self) -> str:
        return f"data:{self.media_type};base64,{self.data}"

    def metadata(self) -> str:
        if self.original_size is None or self.display_size is None:
            return f"bytes={self.output_bytes}"
        original = f"{self.original_size[0]}x{self.original_size[1]}"
        display = f"{self.display_size[0]}x{self.display_size[1]}"
        resized = f", display={display}" if display != original else ""
        return f"dimensions={original}{resized}, bytes={self.output_bytes}"

    def prompt_metadata(self) -> str | None:
        pieces: list[str] = []
        if self.source_path is not None:
            pieces.append(f"source: {self.source_path}")
        if self.original_size and self.display_size:
            original_width, original_height = self.original_size
            display_width, display_height = self.display_size
            if self.original_size != self.display_size:
                x_scale = original_width / max(1, display_width)
                y_scale = original_height / max(1, display_height)
                pieces.append(
                    f"original {original_width}x{original_height}, displayed at "
                    f"{display_width}x{display_height}; multiply displayed x/y coordinates "
                    f"by {x_scale:.2f}/{y_scale:.2f} to map to the original"
                )
            elif self.reencoded:
                pieces.append("re-encoded for API compatibility without resizing")
        if not pieces:
            return None
        return f"[Image: {', '.join(pieces)}]"


@dataclass(slots=True)
class ImageReference:
    path: Path
    media_type: str

    def process(self, max_bytes: int = MAX_SOURCE_IMAGE_BYTES) -> ProcessedImage:
        if max_bytes < 1:
            raise ValueError("Image size limit must be positive")
        metadata = self.path.stat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"Image is not a regular file: {self.path.name}")
        if metadata.st_size > max_bytes:
            raise ValueError("Image exceeds size limit")
        with self.path.open("rb") as source:
            data = source.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError("Image exceeds size limit")
        processed = process_image_bytes(data, self.media_type)
        return ProcessedImage(
            processed.data,
            processed.media_type,
            processed.source_bytes,
            processed.output_bytes,
            processed.original_size,
            processed.display_size,
            processed.reencoded,
            self.path,
        )

    def data_url(self, max_bytes: int = MAX_SOURCE_IMAGE_BYTES) -> str:
        return self.process(max_bytes).data_url()


@dataclass(frozen=True, slots=True)
class PreparedImageInput:
    text: str
    image_urls: tuple[str, ...]
    errors: tuple[str, ...] = ()


class _ImageSafetyError(ValueError):
    pass


def process_base64_image(data: str, media_type: str) -> ProcessedImage:
    if not data.strip():
        raise ValueError("Image data is empty")
    if len(data) > ((MAX_SOURCE_IMAGE_BYTES + 2) // 3) * 4 + 4:
        raise ValueError("Image exceeds the 50 MiB processing limit")
    try:
        decoded = base64.b64decode(data, validate=True)
    except ValueError as exc:
        raise ValueError("Image data is not valid base64") from exc
    return process_image_bytes(decoded, media_type)


def process_image_bytes(data: bytes, media_type: str) -> ProcessedImage:
    if not data:
        raise ValueError("Image data is empty")
    if len(data) > MAX_SOURCE_IMAGE_BYTES:
        raise ValueError("Image exceeds the 50 MiB processing limit")
    normalized_type = media_type.casefold() if media_type.startswith("image/") else "image/png"
    if normalized_type == "image/jpg":
        normalized_type = "image/jpeg"
    encoded = base64.b64encode(data).decode()
    try:
        from PIL import Image, ImageOps
    except ImportError:
        if len(encoded) <= API_IMAGE_MAX_BASE64_SIZE:
            return ProcessedImage(encoded, normalized_type, len(data), len(data))
        raise ValueError(
            "Pillow is required to compress images above the 5 MiB API limit"
        ) from None

    try:
        with Image.open(io.BytesIO(data)) as opened:
            width, height = opened.size
            if width * height > MAX_SOURCE_IMAGE_PIXELS:
                raise _ImageSafetyError(
                    f"Image exceeds the {MAX_SOURCE_IMAGE_PIXELS:,}-pixel safety limit"
                )
            image = ImageOps.exif_transpose(opened)
            image.load()
    except (Image.DecompressionBombError, _ImageSafetyError) as exc:
        raise ValueError("Image dimensions exceed safety limits: " + safe_text(exc)) from None
    except Exception as exc:
        if len(encoded) <= API_IMAGE_MAX_BASE64_SIZE:
            return ProcessedImage(encoded, normalized_type, len(data), len(data))
        raise ValueError(
            "Image cannot be decoded for compression and exceeds the 5 MiB API limit"
        ) from exc

    original_size = image.size
    has_alpha = image.mode in {"RGBA", "LA"} or "transparency" in image.info
    if len(encoded) <= API_IMAGE_MAX_BASE64_SIZE and not has_alpha:
        return ProcessedImage(
            encoded,
            normalized_type,
            len(data),
            len(data),
            original_size,
            original_size,
        )

    if has_alpha and len(encoded) <= API_IMAGE_MAX_BASE64_SIZE:
        rgba = image.convert("RGBA")
        flattened = Image.new("RGB", rgba.size, "white")
        flattened.paste(rgba, mask=rgba.getchannel("A"))
        image = flattened
        png = _encode_image(image, "PNG")
        png_base64 = base64.b64encode(png).decode()
        if len(png_base64) <= API_IMAGE_MAX_BASE64_SIZE:
            return ProcessedImage(
                png_base64,
                "image/png",
                len(data),
                len(png),
                original_size,
                image.size,
                True,
            )

    image.thumbnail((IMAGE_MAX_WIDTH, IMAGE_MAX_HEIGHT))
    display_size = image.size
    png = _encode_image(image, "PNG")
    png_base64 = base64.b64encode(png).decode()
    if len(png_base64) <= API_IMAGE_MAX_BASE64_SIZE:
        return ProcessedImage(
            png_base64,
            "image/png",
            len(data),
            len(png),
            original_size,
            display_size,
            True,
        )

    rgb = image.convert("RGB")
    for quality in (85, 70, 55, 40, 25):
        jpeg = _encode_image(rgb, "JPEG", quality=quality)
        jpeg_base64 = base64.b64encode(jpeg).decode()
        if len(jpeg_base64) <= API_IMAGE_MAX_BASE64_SIZE:
            return ProcessedImage(
                jpeg_base64,
                "image/jpeg",
                len(data),
                len(jpeg),
                original_size,
                display_size,
                True,
            )
    if image.width > 512 or image.height > 512:
        image.thumbnail((1_200, 1_200))
        rgb = image.convert("RGB")
        for quality in (85, 70, 55, 40, 25):
            jpeg = _encode_image(rgb, "JPEG", quality=quality)
            jpeg_base64 = base64.b64encode(jpeg).decode()
            if len(jpeg_base64) <= API_IMAGE_MAX_BASE64_SIZE:
                return ProcessedImage(
                    jpeg_base64,
                    "image/jpeg",
                    len(data),
                    len(jpeg),
                    original_size,
                    image.size,
                    True,
                )
    raise ValueError("Image remains above the 5 MiB API limit after compression")


def _encode_image(image: object, image_format: str, **options: object) -> bytes:
    output = io.BytesIO()
    image.save(output, format=image_format, **options)  # type: ignore[attr-defined]
    return output.getvalue()


def parse_image_references(text: str, workspace: Path) -> list[ImageReference]:
    guard = PathGuard(workspace)
    result: list[ImageReference] = []
    for match in IMAGE_REFERENCE_PATTERN.finditer(text):
        raw = match.group(1)
        if raw is None:
            continue
        value = raw[1:-1] if raw.startswith("<") and raw.endswith(">") else raw
        path = _resolve_image_path(value, guard)
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if media_type == "image/jpg":
            media_type = "image/jpeg"
        if not media_type.startswith("image/"):
            raise ValueError(f"Not an image: {path.name}")
        result.append(ImageReference(path, media_type))
    return result


async def prepare_image_input(
    text: str,
    workspace: Path,
    clipboard_dir: Path,
) -> PreparedImageInput:
    references: list[tuple[str, ImageReference | None, str | None]] = []
    guard = PathGuard(workspace)
    for match in IMAGE_REFERENCE_PATTERN.finditer(text):
        raw = match.group(1)
        label = raw or "clipboard"
        try:
            if raw is None:
                reference = await capture_clipboard_image(clipboard_dir)
            else:
                value = raw[1:-1] if raw.startswith("<") and raw.endswith(">") else raw
                path = _resolve_image_path(value, guard)
                media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                if not media_type.startswith("image/"):
                    raise ValueError(f"Not an image: {path.name}")
                reference = ImageReference(path, media_type)
            references.append((label, reference, None))
        except (OSError, ValueError) as exc:
            references.append((label, None, safe_text(exc)))

    cleaned = IMAGE_REFERENCE_PATTERN.sub("", text)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned).strip()
    urls: list[str] = []
    notes: list[str] = []
    errors: list[str] = []
    for label, attached_reference, error in references:
        if error is not None or attached_reference is None:
            detail = error or "unknown error"
            note = f"[Invalid image reference: {label}; reason: {detail}]"
            notes.append(note)
            errors.append(note)
            continue
        try:
            processed = await asyncio.to_thread(attached_reference.process)
        except (OSError, ValueError) as exc:
            note = f"[Invalid image reference: {label}; reason: {safe_text(exc)}]"
            notes.append(note)
            errors.append(note)
            continue
        urls.append(processed.data_url())
        metadata = processed.prompt_metadata()
        if metadata:
            notes.append(metadata)

    if urls:
        cleaned = cleaned or "Analyze the attached image(s)."
        notes.insert(
            0,
            "[Images are attached to this turn. Inspect them directly; current image content "
            "takes precedence over historical context. Do not re-read an Image source path with "
            "filesystem, browser, or MCP tools. If attachments are not visible, say so instead "
            "of guessing from paths or history.]",
        )
    if notes:
        cleaned = f"{cleaned}\n\n" if cleaned else ""
        cleaned += "\n".join(notes)
    return PreparedImageInput(cleaned, tuple(urls), tuple(errors))


def _resolve_image_path(value: str, guard: PathGuard) -> Path:
    if value.casefold().startswith("file://"):
        parsed = urlparse(value)
        raw_path = unquote(parsed.path)
        value = raw_path or unquote(value[7:])
    return guard.resolve(value, must_exist=True)


async def capture_clipboard_image(output_dir: Path) -> ImageReference:
    await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)
    target = output_dir / f"clipboard-{uuid.uuid4().hex}.png"
    try:
        from PIL import ImageGrab

        captured = await asyncio.to_thread(ImageGrab.grabclipboard)
        if captured is not None and hasattr(captured, "save"):
            await asyncio.to_thread(captured.save, target, "PNG")
            return ImageReference(target, "image/png")
    except Exception:
        pass
    if sys.platform == "darwin" and shutil.which("osascript"):
        if await _capture_macos_clipboard(target):
            return ImageReference(target, "image/png")
    command: list[str] | None = None
    if shutil.which("pngpaste"):
        command = ["pngpaste", str(target)]
    elif shutil.which("wl-paste"):
        command = ["wl-paste", "--no-newline", "--type", "image/png"]
    if command:
        if command[0] == "wl-paste":
            returncode, stdout, _ = await _run_bounded_process(command)
            if returncode == 0 and stdout:
                await asyncio.to_thread(target.write_bytes, stdout)
        else:
            await _run_bounded_process(command)
        exists_with_data = await asyncio.to_thread(
            lambda: target.is_file() and target.stat().st_size > 0
        )
        if exists_with_data:
            return ImageReference(target, "image/png")
    raise ValueError(
        "Clipboard does not contain an image or no clipboard image helper is available"
    )


async def _capture_macos_clipboard(target: Path) -> bool:
    png_script = b"""
on run argv
  set outputPath to item 1 of argv
  set imageData to (the clipboard as \xc2\xabclass PNGf\xc2\xbb)
  set fh to open for access (POSIX file outputPath as string) with write permission
  try
    set eof of fh to 0
    write imageData to fh
    close access fh
  on error errMsg
    try
      close access fh
    end try
    error errMsg
  end try
end run
"""
    code, _, _ = await _run_bounded_process(["/usr/bin/osascript", "-", str(target)], png_script)
    if code == 0 and await asyncio.to_thread(_has_file_data, target):
        return True
    await asyncio.to_thread(target.unlink, missing_ok=True)
    tiff = target.with_suffix(".tiff")
    tiff_script = png_script.replace(b"PNGf", b"TIFF")
    try:
        code, _, _ = await _run_bounded_process(["/usr/bin/osascript", "-", str(tiff)], tiff_script)
        if code != 0 or not await asyncio.to_thread(_has_file_data, tiff):
            return False
        code, _, _ = await _run_bounded_process(
            ["/usr/bin/sips", "-s", "format", "png", str(tiff), "--out", str(target)]
        )
        return code == 0 and await asyncio.to_thread(_has_file_data, target)
    finally:
        await asyncio.to_thread(tiff.unlink, missing_ok=True)


async def _run_bounded_process(
    command: list[str], stdin: bytes | None = None
) -> tuple[int, bytes, bytes]:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE if stdin is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=os.name == "posix",
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(stdin), 8)
    except TimeoutError:
        await _terminate_process(process)
        return -1, b"", b"clipboard helper timed out"
    except asyncio.CancelledError:
        await _terminate_process(process)
        raise
    return process.returncode or 0, stdout, stderr


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), 1)
        return
    except TimeoutError:
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        return
    await process.wait()


def _has_file_data(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0
