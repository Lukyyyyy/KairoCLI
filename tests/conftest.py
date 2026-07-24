"""Shared pytest configuration."""

from __future__ import annotations

import hashlib
from typing import Any

_MAX_PARAMETER_ID_LENGTH = 120


def pytest_make_parametrize_id(config: Any, val: object, argname: str) -> str | None:
    """Keep adversarial boundary values from flooding CI logs."""
    del config
    if not isinstance(val, (str, bytes)) or len(val) <= _MAX_PARAMETER_ID_LENGTH:
        return None
    encoded = val.encode("utf-8", errors="replace") if isinstance(val, str) else val
    digest = hashlib.sha256(encoded).hexdigest()[:12]
    return f"{argname}=<{type(val).__name__}:{len(val)}:{digest}>"
