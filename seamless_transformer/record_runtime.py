"""Runtime helpers for execution-record mode."""

from __future__ import annotations

_CACHED_RECORD_MODE: bool | None = None


def get_record_mode() -> bool:
    global _CACHED_RECORD_MODE
    if _CACHED_RECORD_MODE is None:
        from seamless_config.select import get_record

        _CACHED_RECORD_MODE = bool(get_record())
    return _CACHED_RECORD_MODE


def invalidate_record_mode_cache() -> None:
    global _CACHED_RECORD_MODE
    _CACHED_RECORD_MODE = None


__all__ = ["get_record_mode", "invalidate_record_mode_cache"]
