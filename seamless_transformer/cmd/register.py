"""Register buffers:
- Write them remotely (if configured)
- Keep them in the local buffer cache"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from http.client import HTTPConnection
from typing import Literal
from urllib.parse import urlsplit

from seamless import Buffer, Checksum
from seamless.checksum.calculate_checksum import calculate_checksum
from seamless.checksum.json_ import json_dumps_bytes
from seamless.caching.buffer_cache import get_buffer_cache
from .message import message as msg

TransferMode = Literal["copy", "hardlink", "symlink"]
DestinationEntryPolicy = Literal["skip", "verify", "repair"]


class DestinationEntryError(RuntimeError):
    """Raised when a destination entry violates the requested integrity policy."""


def _ensure_cached(buffer: Buffer) -> None:
    """Ensure a buffer stays available for subsequent resolve() calls."""
    buffer.get_checksum()
    buffer.tempref()


def _ensure_dir_permissions(path: str, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except Exception:
        pass


def _has_hashserver_prefix(directory: str) -> bool:
    return os.path.exists(os.path.join(directory, ".HASHSERVER_PREFIX"))


def _resolve_destination_path(
    destination_folder: str, checksum_hex: str, *, create_dirs: bool = True
) -> str:
    if _has_hashserver_prefix(destination_folder):
        prefix = checksum_hex[:2]
        target_dir = os.path.join(destination_folder, prefix)
        if create_dirs:
            os.makedirs(target_dir, exist_ok=True)
            _ensure_dir_permissions(target_dir, 0o3775)
        return os.path.join(target_dir, checksum_hex)
    return os.path.join(destination_folder, checksum_hex)


def _write_buffer_to_destination(
    buffer: bytes,
    destination_folder: str,
    checksum_hex: str,
    *,
    overwrite: bool = False,
) -> None:
    actual_checksum = calculate_checksum(buffer)
    if actual_checksum != checksum_hex:
        raise DestinationEntryError(
            f"Refusing to write checksum '{checksum_hex}': source bytes have checksum '{actual_checksum}'"
        )
    path = _resolve_destination_path(destination_folder, checksum_hex, create_dirs=True)
    if os.path.lexists(path) and not overwrite:
        _ensure_dir_permissions(path, 0o444)
        return
    target_dir = os.path.dirname(path)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=checksum_hex + "-", dir=target_dir)
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(buffer)
        os.replace(tmp_path, path)
        _ensure_dir_permissions(path, 0o444)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass


def _destination_entry_mismatch_message(path: str, checksum_hex: str) -> str:
    return f"Destination entry '{path}' does not match checksum '{checksum_hex}'"


def _existing_destination_entry_valid(path: str, checksum_hex: str) -> bool:
    try:
        with open(path, "rb") as f:
            buffer = f.read()
    except Exception:
        return False
    return calculate_checksum(buffer) == checksum_hex


def _handle_existing_destination_entry(
    destination_folder: str,
    checksum_hex: str,
    *,
    existing_entry_policy: DestinationEntryPolicy = "skip",
    repair_buffer: bytes | None = None,
) -> bool:
    path = _resolve_destination_path(destination_folder, checksum_hex, create_dirs=False)
    if not os.path.lexists(path):
        return False
    if existing_entry_policy == "skip":
        return True
    if _existing_destination_entry_valid(path, checksum_hex):
        return True
    if existing_entry_policy == "verify":
        raise DestinationEntryError(
            _destination_entry_mismatch_message(path, checksum_hex)
        )
    if repair_buffer is None:
        return False
    _write_buffer_to_destination(
        repair_buffer, destination_folder, checksum_hex, overwrite=True
    )
    return True


def _ensure_url_scheme(url: str) -> str:
    if "://" in url:
        return url
    return "http://" + url


def _has_sync(url: str, checksums: list[str]) -> set[str]:
    if not checksums:
        return set()
    url = _ensure_url_scheme(url)
    parts = urlsplit(url)
    host = parts.hostname
    if host is None:
        return set()
    port = parts.port or (80 if parts.scheme == "http" else 443)
    try:
        conn = HTTPConnection(host, port, timeout=1.0)
        body = json.dumps(checksums)
        conn.request(
            "GET", "/has", body=body, headers={"Content-Type": "application/json"}
        )
        resp = conn.getresponse()
        if not (200 <= resp.status < 300):
            conn.close()
            return set()
        data = resp.read()
        conn.close()
        result = json.loads(data)
        if not isinstance(result, list) or len(result) != len(checksums):
            return set()
        return {cs for cs, ok in zip(checksums, result) if bool(ok)}
    except Exception:
        return set()


def _buffers_exist_remote(checksum_hexes: list[str]) -> set[str]:
    try:
        import seamless_remote.buffer_remote as buffer_remote
    except Exception:
        return set()

    def _ensure_client(client) -> None:
        init_sync = getattr(client, "ensure_initialized_sync", None)
        if callable(init_sync):
            try:
                init_sync(skip_healthcheck=True)
            except Exception:
                pass

    present = set()

    try:
        read_folders = list(buffer_remote._read_folders_clients)
    except Exception:
        read_folders = []
    for client in read_folders:
        _ensure_client(client)
        directory = getattr(client, "directory", None)
        if not directory:
            continue
        for checksum_hex in checksum_hexes:
            if checksum_hex in present:
                continue
            filename = os.path.join(directory, checksum_hex)
            if os.path.exists(filename):
                present.add(checksum_hex)
                continue
            subdir = os.path.join(directory, checksum_hex[:2])
            if os.path.isdir(subdir):
                if os.path.exists(os.path.join(subdir, checksum_hex)):
                    present.add(checksum_hex)

    remaining = [cs for cs in checksum_hexes if cs not in present]
    try:
        read_servers = list(buffer_remote._read_server_clients)
    except Exception:
        read_servers = []
    for client in read_servers:
        if not remaining:
            break
        _ensure_client(client)
        url = getattr(client, "url", None)
        if not url:
            continue
        present.update(_has_sync(url, remaining))
        remaining = [cs for cs in remaining if cs not in present]

    return present


def _buffers_exist_local(
    destination_folder: str,
    checksum_hexes: list[str],
    *,
    existing_entry_policy: DestinationEntryPolicy = "skip",
) -> set[str]:
    if not checksum_hexes:
        return set()
    present = set()
    for checksum_hex in checksum_hexes:
        if checksum_hex in present:
            continue
        path = _resolve_destination_path(
            destination_folder, checksum_hex, create_dirs=False
        )
        if not os.path.lexists(path):
            continue
        if existing_entry_policy == "skip":
            present.add(checksum_hex)
            continue
        if _existing_destination_entry_valid(path, checksum_hex):
            present.add(checksum_hex)
            continue
        if existing_entry_policy == "verify":
            raise DestinationEntryError(
                _destination_entry_mismatch_message(path, checksum_hex)
            )
    return present


def _buffer_exists_remote(checksum: Checksum) -> bool:
    return checksum.hex() in _buffers_exist_remote([checksum.hex()])


def check_checksums_present(
    checksum_hexes: list[str],
    destination_folder: str | None = None,
    *,
    existing_entry_policy: DestinationEntryPolicy = "skip",
) -> set[str]:
    present = set()
    cache = get_buffer_cache()
    remaining = []
    for checksum_hex in dict.fromkeys(checksum_hexes):
        try:
            checksum = Checksum(checksum_hex)
        except Exception:
            continue
        if cache.get(checksum) is not None:
            present.add(checksum_hex)
        else:
            remaining.append(checksum_hex)
    if remaining:
        if destination_folder is not None:
            present.update(
                _buffers_exist_local(
                    destination_folder,
                    remaining,
                    existing_entry_policy=existing_entry_policy,
                )
            )
        else:
            present.update(_buffers_exist_remote(remaining))
    return present


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
    buffer: bytes,
    destination_folder: str | None = None,
    dry_run: bool = False,
    *,
    transfer_mode: TransferMode | None = None,
    existing_entry_policy: DestinationEntryPolicy = "skip",
) -> str:
    """Register a buffer locally; optionally write it to destination or remote."""
    buf = Buffer(buffer)
    checksum = buf.get_checksum()
    _ensure_cached(buf)

    if dry_run:
        return checksum.hex()

    if destination_folder is not None:
        if transfer_mode is None:
            transfer_mode = "copy"
        try:
            already_present = _handle_existing_destination_entry(
                destination_folder,
                checksum.hex(),
                existing_entry_policy=existing_entry_policy,
                repair_buffer=buffer,
            )
            if not already_present:
                _write_buffer_to_destination(buffer, destination_folder, checksum.hex())
        except DestinationEntryError:
            raise
        except Exception as exc:
            msg(
                0,
                "WARNING: could not write buffer to destination folder "
                f"'{destination_folder}': {exc}; falling back to remote write",
            )
            _write_buffer_remote(buf)
    else:
        _write_buffer_remote(buf)

    return checksum.hex()


def register_dict(
    data: dict,
    destination_folder: str | None = None,
    dry_run: bool = False,
    *,
    transfer_mode: TransferMode | None = None,
    existing_entry_policy: DestinationEntryPolicy = "skip",
) -> str:
    """Register the buffer underlying a dict (celltype="plain")."""
    buffer = json_dumps_bytes(data) + b"\n"
    return register_buffer(
        buffer,
        destination_folder=destination_folder,
        dry_run=dry_run,
        transfer_mode=transfer_mode,
        existing_entry_policy=existing_entry_policy,
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
    filename: str,
    destination_folder: str | None = None,
    hardlink: bool = False,
    *,
    transfer_mode: TransferMode | None = None,
    existing_entry_policy: DestinationEntryPolicy = "skip",
) -> str:
    """Calculate a file checksum and register its contents.

    destination_folder: instead of uploading to a buffer server, write to this folder
    """
    with open(filename, "rb") as f:
        buffer = f.read()

    checksum_hex = calculate_checksum(buffer)
    if transfer_mode is None and hardlink:
        transfer_mode = "hardlink"
    if destination_folder is not None:
        if transfer_mode is None:
            transfer_mode = "copy"
        if _handle_existing_destination_entry(
            destination_folder,
            checksum_hex,
            existing_entry_policy=existing_entry_policy,
            repair_buffer=buffer,
        ):
            return checksum_hex
        destlink = _resolve_destination_path(
            destination_folder, checksum_hex, create_dirs=True
        )
        if transfer_mode == "copy":
            _write_buffer_to_destination(buffer, destination_folder, checksum_hex)
        elif transfer_mode == "hardlink":
            os.link(filename, destlink)
        elif transfer_mode == "symlink":
            os.symlink(os.path.abspath(filename), destlink)
        else:
            raise ValueError(transfer_mode)
        return checksum_hex
    return register_buffer(
        buffer,
        destination_folder=destination_folder,
        transfer_mode=transfer_mode,
        existing_entry_policy=existing_entry_policy,
    )


def check_buffer(buffer: bytes) -> tuple[bool, str]:
    """Check if a buffer is present remotely (best-effort).
    Return the result and the checksum.
    """
    checksum_hex = calculate_checksum(buffer)
    checksum = Checksum(checksum_hex)

    cache = get_buffer_cache()
    if cache.get(checksum) is not None:
        return True, checksum_hex

    if _buffer_exists_remote(checksum):
        return True, checksum_hex
    return False, checksum_hex
