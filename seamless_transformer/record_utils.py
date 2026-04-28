"""Shared helpers for execution-record plumbing."""

from __future__ import annotations

from datetime import datetime, timezone
import os
import resource
import sys

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _memory_peak_bytes() -> int | None:
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        peak = int(usage.ru_maxrss)
    except Exception:
        return None
    if peak <= 0:
        return None
    if sys.platform.startswith("linux"):
        return peak * 1024
    return peak


def _process_create_time_epoch() -> float | None:
    try:
        import psutil
    except Exception:
        return None
    try:
        return float(psutil.Process().create_time())
    except Exception:
        return None


def _resolve_remote_target(
    execution: str,
    *,
    get_remote=None,
    get_selected_cluster=None,
    check_remote_redundancy=None,
) -> str | None:
    if execution != "remote":
        return None
    if (
        get_remote is None
        or get_selected_cluster is None
        or check_remote_redundancy is None
    ):
        from seamless_config.select import (
            check_remote_redundancy as _check_remote_redundancy,
            get_remote as _get_remote,
            get_selected_cluster as _get_selected_cluster,
        )

        if get_remote is None:
            get_remote = _get_remote
        if get_selected_cluster is None:
            get_selected_cluster = _get_selected_cluster
        if check_remote_redundancy is None:
            check_remote_redundancy = _check_remote_redundancy
    remote_target = get_remote()
    if remote_target is not None:
        return remote_target
    cluster = get_selected_cluster()
    if cluster is None:
        return None
    try:
        return check_remote_redundancy(cluster)
    except Exception:
        return None


__all__ = [
    "_memory_peak_bytes",
    "_process_create_time_epoch",
    "_resolve_remote_target",
    "_utcnow_iso",
]
