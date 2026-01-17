"""Register buffers:
- Write them remotely (if configured)
- Keep them in the local buffer cache"""

from __future__ import annotations

import asyncio
import os

from seamless import Buffer, Checksum
from seamless.checksum.calculate_checksum import calculate_checksum
from seamless.checksum.json_ import json_dumps_bytes
from seamless.caching.buffer_cache import get_buffer_cache


def _ensure_cached(buffer: Buffer) -> None:
    """Ensure a buffer stays available for subsequent resolve() calls."""
    buffer.get_checksum()
    buffer.tempref()


def _write_buffer_remote(buffer: Buffer) -> None:
    """Best-effort write to a remote buffer server, if configured."""
    try:
        import seamless_remote.buffer_remote as buffer_remote
    except Exception:
        return
    try:
        has_write = buffer_remote.has_write_server()
    except Exception:
        has_write = False
    if not has_write:
        return
    coro = buffer.write()
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        future.result()
    else:
        asyncio.run(coro)


def register_buffer(
    buffer: bytes, destination_folder: str | None = None, dry_run: bool = False
) -> str:
    """Register a buffer locally; optionally write it to destination or remote."""
    buf = Buffer(buffer)
    checksum = buf.get_checksum()
    _ensure_cached(buf)

    if dry_run:
        return checksum.hex()

    if destination_folder is not None:
        filename = os.path.join(destination_folder, checksum.hex())
        with open(filename, "wb") as f:
            f.write(buffer)
    else:
        _write_buffer_remote(buf)

    return checksum.hex()


def register_dict(
    data: dict, destination_folder: str | None = None, dry_run: bool = False
) -> str:
    """Register the buffer underlying a dict (celltype="plain")."""
    buffer = json_dumps_bytes(data) + b"\n"
    return register_buffer(
        buffer, destination_folder=destination_folder, dry_run=dry_run
    )


def check_file(filename: str) -> tuple[bool, str, int]:
    """Check if a file needs to be written remotely.
    Return the result and the checksum, and the length of the file buffer.
    """
    with open(filename, "rb") as f:
        buffer = f.read()
    result, checksum = check_buffer(buffer)
    return result, checksum, len(buffer)


def register_file(
    filename: str, destination_folder: str | None = None, hardlink: bool = False
) -> str:
    """Calculate a file checksum and register its contents.

    destination_folder: instead of uploading to a buffer server, write to this folder
    """
    with open(filename, "rb") as f:
        buffer = f.read()

    checksum_hex = calculate_checksum(buffer)
    if hardlink and destination_folder is not None:
        destlink = os.path.join(destination_folder, checksum_hex)
        os.link(filename, destlink)
        return checksum_hex
    return register_buffer(buffer, destination_folder=destination_folder)


def check_buffer(buffer: bytes) -> tuple[bool, str]:
    """Check if a buffer is present remotely (best-effort).
    Return the result and the checksum.
    """
    checksum_hex = calculate_checksum(buffer)
    checksum = Checksum(checksum_hex)

    cache = get_buffer_cache()
    if cache.get(checksum) is not None:
        return True, checksum_hex

    try:
        checksum.resolve()
    except Exception:
        return False, checksum_hex
    return True, checksum_hex
