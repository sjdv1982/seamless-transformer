"""Bounded, throttled stdout/stderr capture for streaming transformations."""

from __future__ import annotations

import io
import logging
import threading
import time
from typing import Any, Callable

_LOGGER = logging.getLogger(__name__)
_TAPS: set["_StreamingTap"] = set()
_TAPS_LOCK = threading.RLock()
_FLUSHER: threading.Thread | None = None
_BUFFER_MULTIPLIER = 4


def _bounded_payload_size(value: Any) -> int:
    try:
        size = int(value)
    except (TypeError, ValueError, OverflowError):
        size = 8192
    return min(max(size, 1), 10_240)


def _bounded_interval(value: Any) -> float:
    try:
        interval = float(value)
    except (TypeError, ValueError, OverflowError):
        interval = 2.0
    if interval != interval or interval == float("inf"):
        return 2.0
    return min(max(interval, 0.0), 3600.0)


def _drop_partial_utf8_prefix(data: bytes) -> tuple[bytes, int]:
    """Remove continuation bytes at the head after a byte-bounded truncation."""
    start = 0
    while start < len(data) and (data[start] & 0xC0) == 0x80:
        start += 1
    return data[start:], start


class _StreamingTap(io.TextIOBase):
    """Capture complete output while emitting rate-limited, bounded text chunks."""

    def __init__(
        self,
        *,
        stream_name: str = "stdout",
        sink: io.BufferedIOBase | None = None,
        notifier: Callable[[dict[str, Any]], Any] | None = None,
        max_payload: int = 8192,
        min_interval: float = 2.0,
    ) -> None:
        super().__init__()
        self.stream_name = str(stream_name)
        self.sink = sink if sink is not None else io.BytesIO()
        self.notifier = notifier or (lambda _chunk: None)
        self._max_payload = _bounded_payload_size(max_payload)
        self._min_interval = _bounded_interval(min_interval)
        self._pending = bytearray()
        self._dropped_head_bytes = 0
        self._last_flush = time.monotonic()
        self._seq = 0
        self._lock = threading.RLock()
        # Keep sequence extraction and notifier delivery in one order. In
        # particular, close() must wait for an earlier cadence flush to finish.
        self._send_lock = threading.Lock()
        self._tap_closed = False
        with _TAPS_LOCK:
            _TAPS.add(self)
            _ensure_flusher_locked()

    @property
    def encoding(self) -> str:
        return "utf-8"

    @property
    def errors(self) -> str:
        return "replace"

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def write(self, s: str) -> int:
        if not isinstance(s, str):
            raise TypeError("write() argument must be str")
        encoded = s.encode("utf-8", errors="replace")
        with self._lock:
            if self._tap_closed:
                raise ValueError("I/O operation on closed stream")
            if encoded:
                self.sink.write(encoded)
                self._pending.extend(encoded)
                max_buffer = self._max_payload * _BUFFER_MULTIPLIER
                overflow = len(self._pending) - max_buffer
                if overflow > 0:
                    del self._pending[:overflow]
                    self._dropped_head_bytes += overflow
        return len(s)

    def flush(self) -> None:
        flush = getattr(self.sink, "flush", None)
        if flush is not None:
            flush()

    def update_throttle(self, *, max_payload: int, min_interval: float) -> None:
        """Apply a scheduler update to this active tap."""
        with self._lock:
            if self._tap_closed:
                return
            self._max_payload = _bounded_payload_size(max_payload)
            self._min_interval = _bounded_interval(min_interval)
            max_buffer = self._max_payload * _BUFFER_MULTIPLIER
            overflow = len(self._pending) - max_buffer
            if overflow > 0:
                del self._pending[:overflow]
                self._dropped_head_bytes += overflow

    def _maybe_flush(self, *, force: bool = False) -> None:
        chunk: dict[str, Any] | None = None
        with self._send_lock:
            with self._lock:
                if self._tap_closed or not self._pending:
                    return
                now = time.monotonic()
                if not force and now - self._last_flush < self._min_interval:
                    return

                raw = bytes(self._pending)
                dropped = self._dropped_head_bytes
                self._pending.clear()
                self._dropped_head_bytes = 0
                if len(raw) > self._max_payload:
                    overflow = len(raw) - self._max_payload
                    raw = raw[overflow:]
                    dropped += overflow
                raw, partial_prefix = _drop_partial_utf8_prefix(raw)
                dropped += partial_prefix
                text = raw.decode("utf-8", errors="replace")
                self._last_flush = now
                chunk = {
                    "stream": self.stream_name,
                    "text": text,
                    "truncated_head_bytes": dropped,
                    "seq": self._seq,
                    "ts": time.time(),
                }
                self._seq += 1

            try:
                self.notifier(chunk)
            except Exception:
                _LOGGER.debug("Failed to forward a streamed output chunk", exc_info=True)

    def close(self) -> None:
        with self._send_lock:
            with self._lock:
                if self._tap_closed:
                    return
                # Mark closed in the same critical section as extracting the final
                # bytes so a concurrent write cannot be stranded after this flush.
                self._tap_closed = True
                now = time.monotonic()
                raw = bytes(self._pending)
                dropped = self._dropped_head_bytes
                self._pending.clear()
                self._dropped_head_bytes = 0
                if len(raw) > self._max_payload:
                    overflow = len(raw) - self._max_payload
                    raw = raw[overflow:]
                    dropped += overflow
                raw, partial_prefix = _drop_partial_utf8_prefix(raw)
                dropped += partial_prefix
                chunk = None
                if raw:
                    chunk = {
                        "stream": self.stream_name,
                        "text": raw.decode("utf-8", errors="replace"),
                        "truncated_head_bytes": dropped,
                        "seq": self._seq,
                        "ts": time.time(),
                    }
                    self._seq += 1
                self._last_flush = now

            with _TAPS_LOCK:
                _TAPS.discard(self)
            if chunk is not None:
                try:
                    self.notifier(chunk)
                except Exception:
                    _LOGGER.debug("Failed to forward a final output chunk", exc_info=True)
            try:
                self.flush()
            finally:
                super().close()


def update_active_throttles(*, max_payload: int, min_interval: float) -> None:
    """Update all taps in this child process after a parent notification."""
    with _TAPS_LOCK:
        taps = tuple(_TAPS)
    for tap in taps:
        tap.update_throttle(max_payload=max_payload, min_interval=min_interval)


def _ensure_flusher_locked() -> None:
    global _FLUSHER
    if _FLUSHER is not None and _FLUSHER.is_alive():
        return
    _FLUSHER = threading.Thread(
        target=_flush_loop,
        name="seamless-stream-flusher",
        daemon=True,
    )
    _FLUSHER.start()


def _flush_loop() -> None:
    global _FLUSHER
    while True:
        with _TAPS_LOCK:
            taps = tuple(_TAPS)
            if not taps:
                _FLUSHER = None
                return
        for tap in taps:
            tap._maybe_flush()
        intervals = []
        for tap in taps:
            with tap._lock:
                if not tap._tap_closed:
                    intervals.append(tap._min_interval)
        delay = min((max(interval / 2.0, 0.01) for interval in intervals), default=0.5)
        time.sleep(min(delay, 0.5))
