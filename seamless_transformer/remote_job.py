"""Helpers for remote job materialization control flow."""

from __future__ import annotations

from typing import Any

REMOTE_JOB_META_KEY = "write_remote_job"
_REMOTE_JOB_WRITTEN_PREFIX = "__SEAMLESS_REMOTE_JOB_WRITTEN__:"


class RemoteJobWritten(RuntimeError):
    """Raised when execution stops after writing a remote job directory."""

    def __init__(self, directory: str):
        super().__init__(f"Remote job written to '{directory}'")
        self.directory = directory


def get_write_remote_job(meta: Any) -> str | None:
    """Extract the requested remote job directory from transformation metadata."""

    if not isinstance(meta, dict):
        return None
    directory = meta.get(REMOTE_JOB_META_KEY)
    if not directory:
        return None
    if not isinstance(directory, str):
        raise TypeError(
            f"Transformation meta '{REMOTE_JOB_META_KEY}' must be a string path"
        )
    return directory


def encode_remote_job_written(directory: str) -> str:
    """Encode a successful remote-job write result for transport layers."""

    return _REMOTE_JOB_WRITTEN_PREFIX + directory


def parse_remote_job_written(value: Any) -> str | None:
    """Decode a remote-job write transport marker."""

    if not isinstance(value, str):
        return None
    if not value.startswith(_REMOTE_JOB_WRITTEN_PREFIX):
        return None
    return value[len(_REMOTE_JOB_WRITTEN_PREFIX) :]


__all__ = [
    "REMOTE_JOB_META_KEY",
    "RemoteJobWritten",
    "get_write_remote_job",
    "encode_remote_job_written",
    "parse_remote_job_written",
]
