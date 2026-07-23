from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any, TypeVar

T = TypeVar("T")
CANCELLATION_GRACE_SECONDS = 0.1
_DETACHED_CANCELLATIONS: set[asyncio.Future[Any]] = set()


class AgentCanceled(RuntimeError):
    """Raised when a user-requested cancellation reaches an execution boundary."""


def raise_if_canceled(cancel_event: asyncio.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise AgentCanceled("Task canceled")


async def wait_with_cancellation(awaitable: Awaitable[T], cancel_event: asyncio.Event | None) -> T:
    """Await work while making an asyncio.Event an active cancellation signal."""
    if cancel_event is None:
        work = asyncio.ensure_future(awaitable)
        try:
            return await asyncio.shield(work)
        except asyncio.CancelledError:
            await _cancel_bounded(work)
            raise
    raise_if_canceled(cancel_event)
    work = asyncio.ensure_future(awaitable)
    cancellation = asyncio.create_task(cancel_event.wait())
    try:
        await asyncio.wait({work, cancellation}, return_when=asyncio.FIRST_COMPLETED)
        if cancel_event.is_set():
            await _cancel_bounded(work)
            raise AgentCanceled("Task canceled")
        return await work
    except asyncio.CancelledError:
        await _cancel_bounded(work)
        raise
    finally:
        cancellation.cancel()
        await asyncio.gather(cancellation, return_exceptions=True)


async def _cancel_bounded(work: asyncio.Future[Any]) -> None:
    if work.done():
        await asyncio.gather(work, return_exceptions=True)
        return
    work.cancel()
    done, _ = await asyncio.wait({work}, timeout=CANCELLATION_GRACE_SECONDS)
    if work in done:
        await asyncio.gather(work, return_exceptions=True)
        return
    _DETACHED_CANCELLATIONS.add(work)

    def consume(completed: asyncio.Future[Any]) -> None:
        _DETACHED_CANCELLATIONS.discard(completed)
        try:
            completed.exception()
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    work.add_done_callback(consume)
