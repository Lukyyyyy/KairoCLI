"""Bounded subprocess stream handling for Kairo CLI tools."""

from __future__ import annotations

import asyncio
import os
import signal
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _BoundedStream:
    preview: bytes
    total_bytes: int
    truncated: bool


async def _read_bounded_stream(stream: asyncio.StreamReader, limit: int) -> _BoundedStream:
    head_limit = max(1, limit * 2 // 3)
    tail_limit = max(0, limit - head_limit)
    head = bytearray()
    tail = bytearray()
    total = 0
    while chunk := await stream.read(65_536):
        total += len(chunk)
        if len(head) < head_limit:
            take = min(head_limit - len(head), len(chunk))
            head.extend(chunk[:take])
            chunk = chunk[take:]
        if chunk and tail_limit:
            tail.extend(chunk)
            if len(tail) > tail_limit:
                del tail[: len(tail) - tail_limit]
    truncated = total > limit
    if truncated:
        preview = bytes(head) + b"\n...[output truncated; middle omitted]...\n" + bytes(tail)
    else:
        preview = bytes(head + tail)
    return _BoundedStream(preview, total, truncated)


async def _terminate_process_tree(process: asyncio.subprocess.Process) -> None:
    if os.name == "posix":
        # The shell may have already exited while a background child still owns its
        # pipes. Its process group remains addressable by the original leader PID.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        for _ in range(20):
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return
            await asyncio.sleep(0.05)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        if process.returncode is None:
            await process.wait()
        return
    if process.returncode is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), 1.0)
        return
    except TimeoutError:
        pass
    try:
        process.kill()
    except ProcessLookupError:
        return
    await process.wait()
