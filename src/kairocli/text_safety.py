from __future__ import annotations


def safe_text(value: object, *, fallback: str | None = None) -> str:
    """Return valid Unicode even when an object's string conversion is hostile."""

    if isinstance(value, str):
        text = value
    else:
        try:
            text = str(value)
        except BaseException:
            text = fallback or f"{type(value).__name__} text unavailable"
    return text.encode("utf-8", errors="replace").decode("utf-8")


def bound_utf8(value: object, maximum_bytes: int, marker: str) -> str:
    """Repair and truncate text without splitting UTF-8 or exceeding the byte budget."""

    if maximum_bytes <= 0:
        return ""
    repaired = safe_text(value)
    encoded = repaired.encode("utf-8")
    if len(encoded) <= maximum_bytes:
        return repaired
    marker_bytes = safe_text(marker).encode("utf-8")
    if len(marker_bytes) >= maximum_bytes:
        return marker_bytes[:maximum_bytes].decode("utf-8", errors="ignore")
    available = maximum_bytes - len(marker_bytes)
    return encoded[:available].decode("utf-8", errors="ignore") + marker_bytes.decode("utf-8")
