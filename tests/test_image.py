import base64
import io
from pathlib import Path

import pytest
from PIL import Image

import kairocli.image as image_module
from kairocli.image import (
    ImageReference,
    ProcessedImage,
    parse_image_references,
    prepare_image_input,
    process_base64_image,
)
from kairocli.policy import PolicyDenied


def test_image_reference_to_data_url(tmp_path: Path) -> None:
    image = tmp_path / "pixel.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    reference = parse_image_references("review @image:pixel.png", tmp_path)[0]
    assert reference.media_type == "image/png"
    assert reference.data_url().startswith("data:image/png;base64,")


def test_image_reference_respects_workspace(tmp_path: Path) -> None:
    with pytest.raises(PolicyDenied):
        parse_image_references("@image:../outside.png", tmp_path)


def test_image_reference_reads_only_enough_bytes_to_enforce_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "oversized.png"
    image.write_bytes(b"x")
    requested: list[int] = []

    class TrackingReader(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            requested.append(size)
            return super().read(size)

    def bounded_open(
        _path: Path, mode: str = "r", *args: object, **kwargs: object
    ) -> TrackingReader:
        assert mode == "rb"
        return TrackingReader(b"x" * 100)

    monkeypatch.setattr(Path, "open", bounded_open)
    with pytest.raises(ValueError, match="exceeds size limit"):
        ImageReference(image, "image/png").process(max_bytes=8)

    assert requested == [9]


def test_image_reference_rejects_known_oversize_without_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "oversized.png"
    image.write_bytes(b"x" * 9)

    def unexpected_open(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("known oversized image was opened")

    monkeypatch.setattr(Path, "open", unexpected_open)
    with pytest.raises(ValueError, match="exceeds size limit"):
        ImageReference(image, "image/png").process(max_bytes=8)


def test_image_reference_rejects_non_regular_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a regular file"):
        ImageReference(tmp_path, "image/png").process(max_bytes=8)


def test_alpha_image_is_flattened_and_keeps_dimensions() -> None:
    source = io.BytesIO()
    Image.new("RGBA", (3, 2), (255, 0, 0, 0)).save(source, "PNG")
    processed = process_base64_image(
        base64.b64encode(source.getvalue()).decode(), "image/png"
    )
    assert processed.media_type == "image/png"
    assert processed.original_size == (3, 2)
    assert processed.display_size == (3, 2)
    assert processed.reencoded
    with Image.open(io.BytesIO(base64.b64decode(processed.data))) as flattened:
        assert flattened.mode == "RGB"


def test_image_pixel_budget_is_rejected_instead_of_using_raw_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = io.BytesIO()
    Image.new("RGB", (3, 2), "white").save(source, "PNG")
    monkeypatch.setattr(image_module, "MAX_SOURCE_IMAGE_PIXELS", 5)

    with pytest.raises(ValueError, match="dimensions exceed safety limits"):
        process_base64_image(base64.b64encode(source.getvalue()).decode(), "image/png")


def test_invalid_base64_image_is_rejected() -> None:
    with pytest.raises(ValueError, match="valid base64"):
        process_base64_image("not-base64%%%", "image/png")


def test_angle_file_uri_and_full_width_punctuation_are_parsed(tmp_path: Path) -> None:
    spaced = tmp_path / "中文 image.png"
    spaced.write_bytes(b"\x89PNG\r\n\x1a\n")
    plain = tmp_path / "shot.png"
    plain.write_bytes(b"\x89PNG\r\n\x1a\n")
    angle = parse_image_references(f"look @image:<file://{spaced}>", tmp_path)
    punctuated = parse_image_references(f"look @image:{plain}。这是什么？", tmp_path)
    assert angle[0].path == spaced
    assert punctuated[0].path == plain


async def test_prepare_image_input_keeps_bad_refs_as_notes_and_honors_boundaries(
    tmp_path: Path,
) -> None:
    prepared = await prepare_image_input(
        "keep @clipboardfoo; inspect @image:missing.png。继续",
        tmp_path,
        tmp_path / "clipboard",
    )
    assert prepared.image_urls == ()
    assert "@clipboardfoo" in prepared.text
    assert "。继续" in prepared.text
    assert "Invalid image reference" in prepared.text
    assert len(prepared.errors) == 1


async def test_prepare_image_input_adds_direct_inspection_instructions(tmp_path: Path) -> None:
    image = tmp_path / "pixel.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    prepared = await prepare_image_input(
        "@image:pixel.png",
        tmp_path,
        tmp_path / "clipboard",
    )
    assert len(prepared.image_urls) == 1
    assert "Analyze the attached image" in prepared.text
    assert "takes precedence over historical context" in prepared.text
    assert "source:" in prepared.text


def test_resized_image_metadata_explains_coordinate_mapping() -> None:
    processed = ProcessedImage(
        "data",
        "image/jpeg",
        10,
        4,
        (4_000, 2_000),
        (2_000, 1_000),
        True,
        Path("diagram.png"),
    )
    metadata = processed.prompt_metadata()
    assert metadata is not None
    assert "multiply displayed x/y coordinates by 2.00/2.00" in metadata
