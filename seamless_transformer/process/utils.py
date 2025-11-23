"""Shared helpers for the process management primitives."""

from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable


async def run_handler(handler: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Execute ``handler`` and await the result when necessary."""

    result = handler(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result  # type: ignore[return-value]
    return result


def ensure_memoryview(buffer: Any) -> memoryview:
    """Return a contiguous memoryview representation of ``buffer``."""

    view = memoryview(buffer)
    if not view.contiguous:
        view = memoryview(view.tobytes())
    if view.format != "B":
        view = view.cast("B")
    return view
