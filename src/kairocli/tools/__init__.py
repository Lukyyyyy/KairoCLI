"""Built-in tool registry and implementations for Kairo CLI."""

from .filesystem import (
    _read_file_range as _read_file_range,
)
from .registry import (
    SERIALIZED_WORKSPACE_MUTATION_TOOLS,
    ToolDefinition,
    ToolRegistry,
)
from .registry import (
    _capture_diff_text as _capture_diff_text,
)
from .validation import ToolArgumentsError

__all__ = [
    "SERIALIZED_WORKSPACE_MUTATION_TOOLS",
    "ToolArgumentsError",
    "ToolDefinition",
    "ToolRegistry",
]
