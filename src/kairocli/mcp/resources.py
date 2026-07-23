from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .protocol import (
    format_tool_result as format_tool_result,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ResourceMention:
    server: str
    uri: str
    start: int
    end: int
    raw: str


_RESOURCE_MENTION = re.compile(r"@([A-Za-z][\w-]*):([a-z]+)://([^\s@]+)")


def parse_resource_mentions(value: str) -> list[ResourceMention]:
    result: list[ResourceMention] = []
    for match in _RESOURCE_MENTION.finditer(value):
        if _inside_quotes(value, match.start()):
            continue
        result.append(
            ResourceMention(
                match.group(1),
                f"{match.group(2)}://{match.group(3)}",
                match.start(),
                match.end(),
                match.group(0),
            )
        )
    return result


def _inside_quotes(value: str, offset: int) -> bool:
    single = False
    double = False
    escaped = False
    for char in value[:offset]:
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "'" and not double:
            single = not single
        elif char == '"' and not single:
            double = not double
    return single or double
