"""Scoped output routing for locally rendered streaming events."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import IO, Iterator


_stream_output: ContextVar[IO[str] | None] = ContextVar(
    "seamless_stream_output", default=None
)


def get_stream_output_override() -> IO[str] | None:
    """Return the active output target for streamed text, if one is set."""
    return _stream_output.get()


@contextmanager
def stream_output_to(output: IO[str]) -> Iterator[None]:
    """Route streamed text events in this execution context to ``output``."""
    token = _stream_output.set(output)
    try:
        yield
    finally:
        _stream_output.reset(token)
