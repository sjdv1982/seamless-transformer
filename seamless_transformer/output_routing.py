"""Route worker output to the transformation running in the current thread."""

from __future__ import annotations

import contextvars
import functools
import threading
from contextlib import contextmanager
from typing import Any, Iterator, TextIO


_CURRENT_ROUTE: contextvars.ContextVar["_OutputRoute | None"] = contextvars.ContextVar(
    "seamless_worker_output_route", default=None
)
_ROUTED_START_MARKER = "_seamless_output_route_wrapper"


class _OutputRoute:
    def __init__(
        self,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
        *,
        tqdm_patch: Any = None,
        parent: "_OutputRoute | None" = None,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.parent = parent
        self._tqdm_patch = tqdm_patch
        self._active = True
        self._lock = threading.RLock()

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    @property
    def tqdm_patch(self) -> Any:
        with self._lock:
            return self._tqdm_patch if self._active else None

    def _sink(self, stream_name: str) -> TextIO | None:
        return self.stdout if stream_name == "stdout" else self.stderr

    def write(self, stream_name: str, value: str, fallback: TextIO) -> int:
        with self._lock:
            if not self._active:
                return fallback.write(value)
            sink = self._sink(stream_name)
            if sink is not None:
                return sink.write(value)
            if self.parent is not None:
                return self.parent.write(stream_name, value, fallback)
            return fallback.write(value)

    def write_bytes(self, stream_name: str, value: bytes, fallback: TextIO) -> int:
        with self._lock:
            if not self._active:
                return _write_fallback_bytes(fallback, value)
            sink = self._sink(stream_name)
            if sink is not None:
                binary_sink = getattr(sink, "buffer", None)
                if binary_sink is not None:
                    return binary_sink.write(value)
                encoding = getattr(sink, "encoding", None) or "utf-8"
                errors = getattr(sink, "errors", None) or "replace"
                return sink.write(value.decode(encoding, errors=errors))
            if self.parent is not None:
                return self.parent.write_bytes(stream_name, value, fallback)
            return _write_fallback_bytes(fallback, value)

    def flush(self, stream_name: str, fallback: TextIO) -> None:
        with self._lock:
            if not self._active:
                fallback.flush()
                return
            sink = self._sink(stream_name)
            if sink is not None:
                sink.flush()
            elif self.parent is not None:
                self.parent.flush(stream_name, fallback)
            else:
                fallback.flush()

    def stream(self, stream_name: str, fallback: TextIO) -> TextIO:
        with self._lock:
            if not self._active:
                return fallback
            sink = self._sink(stream_name)
            if sink is not None:
                return sink
            if self.parent is not None:
                return self.parent.stream(stream_name, fallback)
            return fallback

    def deactivate(self) -> None:
        with self._lock:
            self._active = False


def _write_fallback_bytes(fallback: TextIO, value: bytes) -> int:
    binary_fallback = getattr(fallback, "buffer", None)
    if binary_fallback is not None:
        return binary_fallback.write(value)
    encoding = getattr(fallback, "encoding", None) or "utf-8"
    errors = getattr(fallback, "errors", None) or "replace"
    return fallback.write(value.decode(encoding, errors=errors))


class _RoutedBinaryStream:
    def __init__(self, stream: "_RoutingStream") -> None:
        self._stream = stream

    def write(self, value: bytes | bytearray | memoryview) -> int:
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise TypeError("a bytes-like object is required")
        data = bytes(value)
        route = _CURRENT_ROUTE.get()
        if route is None:
            return _write_fallback_bytes(self._stream._fallback, data)
        return route.write_bytes(
            self._stream._stream_name, data, self._stream._fallback
        )

    def flush(self) -> None:
        self._stream.flush()

    def writable(self) -> bool:
        return True

    @property
    def closed(self) -> bool:
        return False


class _RoutingStream:
    def __init__(self, fallback: TextIO, stream_name: str) -> None:
        self._fallback = fallback
        self._stream_name = stream_name
        self._binary_stream = _RoutedBinaryStream(self)

    @property
    def encoding(self) -> str | None:
        route = _CURRENT_ROUTE.get()
        stream = (
            self._fallback
            if route is None
            else route.stream(self._stream_name, self._fallback)
        )
        return getattr(stream, "encoding", None)

    @property
    def errors(self) -> str | None:
        route = _CURRENT_ROUTE.get()
        stream = (
            self._fallback
            if route is None
            else route.stream(self._stream_name, self._fallback)
        )
        return getattr(stream, "errors", None)

    @property
    def buffer(self) -> _RoutedBinaryStream:
        return self._binary_stream

    def write(self, value: str) -> int:
        if not isinstance(value, str):
            raise TypeError("write() argument must be str")
        route = _CURRENT_ROUTE.get()
        if route is None:
            return self._fallback.write(value)
        return route.write(self._stream_name, value, self._fallback)

    def writelines(self, lines: Any) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        route = _CURRENT_ROUTE.get()
        if route is None:
            self._fallback.flush()
        else:
            route.flush(self._stream_name, self._fallback)

    def isatty(self) -> bool:
        route = _CURRENT_ROUTE.get()
        stream = (
            self._fallback
            if route is None
            else route.stream(self._stream_name, self._fallback)
        )
        return bool(getattr(stream, "isatty", lambda: False)())

    def fileno(self) -> int:
        route = _CURRENT_ROUTE.get()
        stream = (
            self._fallback
            if route is None
            else route.stream(self._stream_name, self._fallback)
        )
        return stream.fileno()

    def writable(self) -> bool:
        return True

    @property
    def closed(self) -> bool:
        return False

    def close(self) -> None:
        # The router is process-wide and must not close its fallback or sinks.
        self.flush()


@contextmanager
def output_route(
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    *,
    tqdm_patch: Any = None,
) -> Iterator[_OutputRoute]:
    parent = _CURRENT_ROUTE.get()
    if parent is not None and not parent.active:
        parent = None
    route = _OutputRoute(
        stdout,
        stderr,
        tqdm_patch=tqdm_patch,
        parent=parent,
    )
    token = _CURRENT_ROUTE.set(route)
    try:
        yield route
    finally:
        route.deactivate()
        _CURRENT_ROUTE.reset(token)


def current_tqdm_patch() -> Any:
    route = _CURRENT_ROUTE.get()
    if route is None:
        return None
    return route.tqdm_patch


def install_output_routing() -> None:
    """Install stable stdout/stderr routers and child-only thread inheritance."""
    import sys

    if not isinstance(sys.stdout, _RoutingStream):
        sys.stdout = _RoutingStream(sys.stdout, "stdout")
    if not isinstance(sys.stderr, _RoutingStream):
        sys.stderr = _RoutingStream(sys.stderr, "stderr")

    current_start = threading.Thread.start
    if getattr(current_start, _ROUTED_START_MARKER, False):
        return

    @functools.wraps(current_start)
    def routed_start(thread: threading.Thread, *args: Any, **kwargs: Any) -> Any:
        route = _CURRENT_ROUTE.get()
        if route is None or not route.active:
            return current_start(thread, *args, **kwargs)

        original_run = thread.run

        @functools.wraps(original_run)
        def routed_run(*run_args: Any, **run_kwargs: Any) -> Any:
            token = _CURRENT_ROUTE.set(route)
            try:
                return original_run(*run_args, **run_kwargs)
            finally:
                _CURRENT_ROUTE.reset(token)

        thread.run = routed_run  # type: ignore[method-assign]
        try:
            return current_start(thread, *args, **kwargs)
        except BaseException:
            thread.run = original_run  # type: ignore[method-assign]
            raise

    setattr(routed_start, _ROUTED_START_MARKER, True)
    setattr(threading.Thread, "start", routed_start)
