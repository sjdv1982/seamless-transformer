"""Streaming stdout/stderr capture for child transformation processes."""

from __future__ import annotations

import io
import os
import threading
import time
from typing import Any, Callable


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except Exception:
        return default
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, *, minimum: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except Exception:
        return default
    return max(minimum, value)


DEFAULT_MAX_PAYLOAD_BYTES = _env_int(
    "SEAMLESS_STREAM_MAX_PAYLOAD_BYTES", 8192, minimum=1, maximum=10240
)
DEFAULT_MIN_INTERVAL_SECONDS = _env_float(
    "SEAMLESS_STREAM_MIN_INTERVAL_SECONDS", 2.0, minimum=0.05
)


class _StreamingTap(io.TextIOBase):
    """Mirror writes to a local sink and emit bounded streaming chunks."""

    def __init__(
        self,
        *,
        stream_name: str,
        sink: io.TextIOBase,
        notifier: Callable[[dict[str, Any]], None],
        max_payload: int = DEFAULT_MAX_PAYLOAD_BYTES,
        min_interval: float = DEFAULT_MIN_INTERVAL_SECONDS,
    ) -> None:
        self.stream_name = stream_name
        self._sink = sink
        self._notifier = notifier
        self._max_payload = max(1, min(10240, int(max_payload)))
        self._min_interval = max(0.05, float(min_interval))
        self._pending = bytearray()
        self._seq = 0
        self._last_flush = 0.0
        self._closed = False
        self._lock = threading.RLock()
        _STREAM_FLUSHER.add(self)

    @property
    def encoding(self) -> str:
        return getattr(self._sink, "encoding", None) or "utf-8"

    @property
    def errors(self) -> str:
        return getattr(self._sink, "errors", None) or "replace"

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        try:
            return bool(self._sink.isatty())
        except Exception:
            return False

    def write(self, s: str) -> int:
        if not isinstance(s, str):
            s = str(s)
        with self._lock:
            if self._closed:
                return 0
            written = self._sink.write(s)
            try:
                self._sink.flush()
            except Exception:
                pass
            encoded = s.encode(self.encoding, errors=self.errors)
            self._pending.extend(encoded)
            keep = max(self._max_payload * 4, self._max_payload)
            if len(self._pending) > keep:
                del self._pending[: len(self._pending) - keep]
            self._maybe_flush_locked(force=False)
        return written if isinstance(written, int) else len(s)

    def flush(self) -> None:
        with self._lock:
            try:
                self._sink.flush()
            finally:
                self._maybe_flush_locked(force=True)

    def update_throttle(self, *, max_payload: int, min_interval: float) -> None:
        with self._lock:
            self._max_payload = max(1, min(10240, int(max_payload)))
            self._min_interval = max(0.05, float(min_interval))

    def maybe_flush(self) -> None:
        with self._lock:
            self._maybe_flush_locked(force=False)

    def _maybe_flush_locked(self, *, force: bool) -> None:
        if not self._pending:
            return
        now = time.time()
        if not force and now - self._last_flush < self._min_interval:
            return
        data = bytes(self._pending)
        self._pending.clear()
        dropped = 0
        if len(data) > self._max_payload:
            dropped = len(data) - self._max_payload
            data = data[-self._max_payload :]
        text = data.decode(self.encoding, errors="replace")
        self._seq += 1
        self._last_flush = now
        self._notifier(
            {
                "kind": "stream",
                "stream": self.stream_name,
                "text": text,
                "truncated_head_bytes": dropped,
                "seq": self._seq,
                "ts": now,
            }
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._maybe_flush_locked(force=True)
            finally:
                self._closed = True
                _STREAM_FLUSHER.discard(self)
        super().close()


class _StreamFlusher:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._taps: set[_StreamingTap] = set()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def add(self, tap: _StreamingTap) -> None:
        with self._lock:
            self._taps.add(tap)
            if self._thread is None or not self._thread.is_alive():
                self._stop.clear()
                self._thread = threading.Thread(
                    target=self._run, name="seamless-stream-flusher", daemon=True
                )
                self._thread.start()

    def discard(self, tap: _StreamingTap) -> None:
        with self._lock:
            self._taps.discard(tap)

    def _run(self) -> None:
        while not self._stop.wait(0.5):
            with self._lock:
                taps = list(self._taps)
            for tap in taps:
                try:
                    tap.maybe_flush()
                except Exception:
                    pass


_STREAM_FLUSHER = _StreamFlusher()

