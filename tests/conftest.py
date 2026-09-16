"""Shared pytest configuration."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

_MAX_PARAMETER_ID_LENGTH = 120
_EMBEDDING_ENVIRONMENT = (
    "KAIROCLI_MEMORY_SEMANTIC_SEARCH",
    "KAIROCLI_EMBEDDING_PROVIDER",
    "KAIROCLI_EMBEDDING_MODEL",
    "KAIROCLI_EMBEDDING_BASE_URL",
    "KAIROCLI_EMBEDDING_API_KEY",
    "KAIROCLI_EMBEDDING_DIMENSIONS",
    "KAIROCLI_EMBEDDING_MAX_BATCH_SIZE",
    "DASHSCOPE_API_KEY",
)


@pytest.fixture(autouse=True)
def isolate_embedding_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep local embedding credentials and endpoints out of unit tests."""
    for name in _EMBEDDING_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)


def pytest_make_parametrize_id(config: Any, val: object, argname: str) -> str | None:
    """Keep adversarial boundary values from flooding CI logs."""
    del config
    if not isinstance(val, (str, bytes)) or len(val) <= _MAX_PARAMETER_ID_LENGTH:
        return None
    encoded = val.encode("utf-8", errors="replace") if isinstance(val, str) else val
    digest = hashlib.sha256(encoded).hexdigest()[:12]
    return f"{argname}=<{type(val).__name__}:{len(val)}:{digest}>"
