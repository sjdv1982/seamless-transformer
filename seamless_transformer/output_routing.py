"""Route worker output to the transformation running in the current thread."""

from __future__ import annotations

import contextvars
import functools
import io
import threading
from contextlib import contextmanager
from typing import Any, Iterator, TextIO


_CURRENT_ROUTE: contextvars.ContextVar["_OutputRoute | None"] = contextvars.ContextVar(
    "seamless_worker_output_route", default=None
)
_ROUTED_START_MARKER = "_seamless_output_route_wrapper"
_PROCESS_OUTPUT_LOCK = threading.RLock()
_PROCESS_OUTPUT_USERS = 0
_PROCESS_OUTPUT_STATE: (
    tuple[TextIO, TextIO, "_RoutingStream", "_RoutingStream"] | None
) = None


class _OutputRoute:
    def __init__(
        self,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
        *,
        tqdm_patch: Any = None,
        parent: "_OutputRoute | None" = None,
        thread_only: bool = False,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.parent = parent
        self._tqdm_patch = tqdm_patch
        self._thread_only = thread_only
        self._owner_thread_id = threading.get_ident() if thread_only else None
        self._active = True
        self._lock = threading.RLock()

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    @property
    def applies_to_current_thread(self) -> bool:
        with self._lock:
            return self._active and (
                not self._thread_only
                or self._owner_thread_id == threading.get_ident()
            )

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


def _current_output_route() -> _OutputRoute | None:
    route = _CURRENT_ROUTE.get()
    if route is None or not route.applies_to_current_thread:
        return None
    return route


class _RoutedBinaryStream:
    def __init__(self, stream: "_RoutingStream") -> None:
        self._stream = stream

    def write(self, value: bytes | bytearray | memoryview) -> int:
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise TypeError("a bytes-like object is required")
        data = bytes(value)
        route = _current_output_route()
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
        route = _current_output_route()
        stream = (
            self._fallback
            if route is None
            else route.stream(self._stream_name, self._fallback)
        )
        return getattr(stream, "encoding", None)

    @property
    def errors(self) -> str | None:
        route = _current_output_route()
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
        route = _current_output_route()
        if route is None:
            return self._fallback.write(value)
        return route.write(self._stream_name, value, self._fallback)

    def writelines(self, lines: Any) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        route = _current_output_route()
        if route is None:
            self._fallback.flush()
        else:
            route.flush(self._stream_name, self._fallback)

    def isatty(self) -> bool:
        route = _current_output_route()
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
    thread_only: bool = False,
) -> Iterator[_OutputRoute]:
    parent = _current_output_route()
    route = _OutputRoute(
        stdout,
        stderr,
        tqdm_patch=tqdm_patch,
        parent=parent,
        thread_only=thread_only,
    )
    token = _CURRENT_ROUTE.set(route)
    try:
        yield route
    finally:
        route.deactivate()
        _CURRENT_ROUTE.reset(token)


def current_tqdm_patch() -> Any:
    route = _current_output_route()
    if route is None:
        return None
    return route.tqdm_patch


class _TeeStream:
    def __init__(self, live: TextIO, capture: TextIO) -> None:
        self._live = live
        self._capture = capture
        self._binary_stream = _TeeBinaryStream(self)

    @property
    def encoding(self) -> str | None:
        return getattr(self._live, "encoding", None)

    @property
    def errors(self) -> str | None:
        return getattr(self._live, "errors", None)

    @property
    def buffer(self) -> "_TeeBinaryStream":
        return self._binary_stream

    @property
    def closed(self) -> bool:
        return False

    def write(self, value: str) -> int:
        if not isinstance(value, str):
            raise TypeError("write() argument must be str")
        try:
            written = self._live.write(value)
        finally:
            self._capture.write(value)
        return len(value) if written is None else written

    def writelines(self, lines: Any) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        try:
            self._live.flush()
        finally:
            self._capture.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._live, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self._live.fileno()

    def writable(self) -> bool:
        return True

    def close(self) -> None:
        self.flush()


class _TeeBinaryStream:
    def __init__(self, stream: _TeeStream) -> None:
        self._stream = stream

    def write(self, value: bytes | bytearray | memoryview) -> int:
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise TypeError("a bytes-like object is required")
        data = bytes(value)
        encoding = self._stream.encoding or "utf-8"
        try:
            live_buffer = getattr(self._stream._live, "buffer", None)
            if live_buffer is None:
                written = self._stream._live.write(
                    data.decode(
                        encoding, errors=self._stream.errors or "replace"
                    )
                )
            else:
                written = live_buffer.write(data)
        finally:
            self._stream._capture.write(data.decode(encoding, errors="replace"))
        return len(data) if written is None else written

    def flush(self) -> None:
        live_buffer = getattr(self._stream._live, "buffer", None)
        try:
            if live_buffer is None:
                self._stream._live.flush()
            else:
                live_buffer.flush()
        finally:
            self._stream._capture.flush()

    def writable(self) -> bool:
        return True

    @property
    def closed(self) -> bool:
        return False


def _underlying_stream(stream: TextIO) -> TextIO:
    while isinstance(stream, _RoutingStream):
        stream = stream._fallback
    return stream


@contextmanager
def process_output_route() -> Iterator[tuple[TextIO, TextIO]]:
    """Capture one in-process transformation while teeing to caller streams.

    The router is installed for the lifetime of all overlapping captures, but
    the route itself belongs only to the thread that enters this context.
    """
    import sys

    global _PROCESS_OUTPUT_USERS, _PROCESS_OUTPUT_STATE

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    with _PROCESS_OUTPUT_LOCK:
        if _PROCESS_OUTPUT_USERS == 0:
            original_stdout, original_stderr = sys.stdout, sys.stderr
            stdout_router = _RoutingStream(original_stdout, "stdout")
            stderr_router = _RoutingStream(original_stderr, "stderr")
            sys.stdout = stdout_router
            sys.stderr = stderr_router
            _PROCESS_OUTPUT_STATE = (
                original_stdout,
                original_stderr,
                stdout_router,
                stderr_router,
            )
        else:
            assert _PROCESS_OUTPUT_STATE is not None
            original_stdout, original_stderr, _, _ = _PROCESS_OUTPUT_STATE
        _PROCESS_OUTPUT_USERS += 1

    try:
        with output_route(
            _TeeStream(_underlying_stream(original_stdout), stdout_capture),
            _TeeStream(_underlying_stream(original_stderr), stderr_capture),
            thread_only=True,
        ):
            yield stdout_capture, stderr_capture
    finally:
        with _PROCESS_OUTPUT_LOCK:
            _PROCESS_OUTPUT_USERS -= 1
            if _PROCESS_OUTPUT_USERS == 0:
                assert _PROCESS_OUTPUT_STATE is not None
                original_stdout, original_stderr, stdout_router, stderr_router = (
                    _PROCESS_OUTPUT_STATE
                )
                if sys.stdout is stdout_router:
                    sys.stdout = original_stdout
                if sys.stderr is stderr_router:
                    sys.stderr = original_stderr
                _PROCESS_OUTPUT_STATE = None


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
