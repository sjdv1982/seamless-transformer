import asyncio
import functools
import json
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from seamless import Checksum, CacheMissError
from seamless.caching.buffer_cache import get_buffer_cache
from seamless.checksum.calculate_checksum import calculate_file_checksum

from .exceptions import SeamlessSystemExit
from .confirm import confirm_yna
from .message import message as msg, message_and_exit as err
from .bytes2human import bytes2human
from .register import _resolve_destination_path

stdout_lock = threading.Lock()

TransferMode = Literal["copy", "hardlink", "symlink"]
DestinationEntryPolicy = Literal["skip", "verify", "repair"]


def _run_coro_in_thread(coro):
    result = {}
    error = {}

    def _runner():
        try:
            result["value"] = asyncio.run(coro)
        except Exception as exc:
            error["exc"] = exc

    thread = threading.Thread(target=_runner, name="download-buffer-lengths")
    thread.start()
    thread.join()
    if error:
        return None
    return result.get("value")


def exists_file(filename, download_checksum):
    try:
        download_checksum = Checksum(download_checksum)
    except Exception:
        return False

    try:
        current_checksum = calculate_file_checksum(filename)
    except Exception:
        return False

    return current_checksum == download_checksum.hex()


def _buffer_source_path(source_directory: str, checksum_hex: str) -> str | None:
    path = _resolve_destination_path(
        source_directory, checksum_hex, create_dirs=False
    )
    if os.path.lexists(path):
        return path
    return None


def _get_buffer_length_direct(source_directory: str, checksum_hex: str) -> int | None:
    path = _buffer_source_path(source_directory, checksum_hex)
    if path is None:
        return None
    try:
        return os.path.getsize(path)
    except Exception:
        return None


def _remove_path(path: str) -> None:
    if not os.path.lexists(path):
        return
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    else:
        os.unlink(path)


def _handle_existing_target(
    filename: str,
    checksum_hex: str,
    *,
    existing_entry_policy: DestinationEntryPolicy,
) -> bool:
    if not os.path.lexists(filename):
        return False
    if existing_entry_policy == "skip":
        return True
    if exists_file(filename, checksum_hex):
        return True
    if existing_entry_policy == "verify":
        raise SeamlessSystemExit(
            f"Destination entry '{filename}' does not match checksum '{checksum_hex}'"
        )
    _remove_path(filename)
    return False


def get_buffer_length(checksums, *, source_directory: str | None = None):
    cache = get_buffer_cache()
    results = {}
    missing = []
    for checksum in checksums:
        try:
            checksum_obj = Checksum(checksum)
        except Exception:
            results[checksum] = None
            continue
        if not checksum_obj:
            results[checksum] = None
            continue
        buf = cache.get(checksum_obj)
        if buf is not None:
            results[checksum] = len(buf.content)
        else:
            results[checksum] = None
            missing.append((checksum, checksum_obj))

    if source_directory is not None:
        for checksum, _checksum_obj in list(missing):
            length = _get_buffer_length_direct(source_directory, checksum)
            if length is not None:
                results[checksum] = length
        missing = [(cs, obj) for cs, obj in missing if results[cs] is None]

    if missing:
        try:
            import seamless_remote.buffer_remote as buffer_remote
        except Exception:
            buffer_remote = None
        if buffer_remote is not None:
            coro = buffer_remote.get_buffer_lengths(
                [checksum_obj for _, checksum_obj in missing]
            )
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                lengths = _run_coro_in_thread(coro)
            else:
                try:
                    lengths = asyncio.run(coro)
                except Exception:
                    lengths = None
            if lengths is None:
                try:
                    coro.close()
                except Exception:
                    pass
            if isinstance(lengths, list) and len(lengths) == len(missing):
                for (checksum, _checksum_obj), length in zip(missing, lengths):
                    results[checksum] = length
    return results


def _download_file_remote(filename, file_checksum):
    try:
        file_checksum = Checksum(file_checksum)
    except Exception:
        return
    try:
        cache = get_buffer_cache()
        file_buffer = cache.get(file_checksum)
        if file_buffer is None:
            file_buffer = file_checksum.resolve()
        if file_buffer is None:
            raise CacheMissError(file_checksum)
    except CacheMissError:
        with stdout_lock:
            msg(
                0,
                f"Cannot download contents of file '{filename}, checksum {file_checksum}'",
            )
        return
    try:
        if filename == "/dev/stdout":
            sys.stdout.buffer.write(file_buffer.content)
        elif filename == "/dev/stderr":
            sys.stderr.buffer.write(file_buffer.content)
        else:
            with open(filename, "wb") as f:
                f.write(file_buffer.content)
    except Exception:
        with stdout_lock:
            msg(0, f"Cannot download file '{filename}'")
        return


def _download_file_direct(
    filename: str,
    file_checksum: str,
    *,
    source_directory: str,
    transfer_mode: TransferMode,
    existing_entry_policy: DestinationEntryPolicy,
) -> None:
    try:
        file_checksum = Checksum(file_checksum)
    except Exception:
        return
    checksum_hex = file_checksum.hex()
    if _handle_existing_target(
        filename, checksum_hex, existing_entry_policy=existing_entry_policy
    ):
        return
    source_path = _buffer_source_path(source_directory, checksum_hex)
    if source_path is None:
        with stdout_lock:
            msg(
                0,
                f"Cannot download contents of file '{filename}', checksum '{checksum_hex}'",
            )
        return
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    try:
        if transfer_mode == "copy":
            shutil.copyfile(source_path, filename)
        elif transfer_mode == "hardlink":
            os.link(source_path, filename)
        elif transfer_mode == "symlink":
            os.symlink(os.path.abspath(source_path), filename)
        else:
            raise ValueError(transfer_mode)
    except Exception:
        with stdout_lock:
            msg(0, f"Cannot download file '{filename}'")
        return


def download_file(
    filename,
    file_checksum,
    *,
    source_directory: str | None = None,
    transfer_mode: TransferMode = "copy",
    existing_entry_policy: DestinationEntryPolicy = "skip",
):
    if source_directory is not None:
        return _download_file_direct(
            filename,
            file_checksum,
            source_directory=source_directory,
            transfer_mode=transfer_mode,
            existing_entry_policy=existing_entry_policy,
        )
    return _download_file_remote(filename, file_checksum)


def download_index(index_checksum: Checksum, dirname):
    index_checksum = Checksum(index_checksum)
    cache = get_buffer_cache()
    index_buffer = cache.get(index_checksum)
    if index_buffer is None:
        err(
            f"Cannot download directory '{dirname}' index '{index_checksum}', CacheMissError"
        )
    try:
        index_data = json.loads(index_buffer.content.decode())
    except (json.JSONDecodeError, UnicodeDecodeError):
        err(
            f"Cannot load directory '{dirname}' index from '{index_checksum}': invalid index"
        )
    return index_data, index_buffer.content


def download(
    files,
    directories,
    *,
    checksum_dict,
    max_download_size,
    max_download_files,
    auto_confirm,
    index_checksums=None,
    source_directory: str | None = None,
    transfer_mode: TransferMode = "copy",
    existing_entry_policy: DestinationEntryPolicy = "skip",
):
    checksum_dict_original = checksum_dict
    checksum_dict = checksum_dict.copy()
    skipped_targets = []
    for target, checksum in list(checksum_dict.items()):
        try:
            skip = _handle_existing_target(
                target, checksum, existing_entry_policy=existing_entry_policy
            )
        except SeamlessSystemExit:
            raise
        if skip:
            skipped_targets.append(target)
            checksum_dict.pop(target)
            if target in files:
                files.remove(target)
    if skipped_targets:
        msg(2, f"Skip {len(skipped_targets)} files that already exist")

    if len(checksum_dict):
        msg(2, f"Download {len(checksum_dict)} files")
    checksums = set(checksum_dict.values())
    buffer_lengths = get_buffer_length(checksums, source_directory=source_directory)

    size_load_per_file = 100000
    size_load_per_unknown_file = 10000000000
    processed_downloads = {}

    for download_target in files:
        buffer_length = buffer_lengths[checksum_dict[download_target]]
        if buffer_length is None:
            size_load = size_load_per_unknown_file
            buffer_length = 0
            unknown = 1
        else:
            size_load = size_load_per_file + buffer_length
            unknown = 0
        curr_download = checksum_dict[download_target]
        processed_downloads[download_target] = (
            size_load,
            buffer_length,
            curr_download,
            unknown,
        )

    for download_target in directories:
        curr_files = [f for f in checksum_dict if f.startswith(download_target + "/")]
        if not curr_files:
            curr_files_original = [
                f for f in checksum_dict_original if f.startswith(download_target + "/")
            ]
            if not curr_files_original:
                continue
        buffer_length = 0
        unknown_buffer_lengths = 0
        for f in curr_files:
            bl = buffer_lengths[checksum_dict[f]]
            if bl is None:
                unknown_buffer_lengths += 1
            else:
                buffer_length += bl
        size_load = size_load_per_file * len(curr_files) + buffer_length
        size_load += size_load_per_unknown_file * unknown_buffer_lengths
        striplen = len(download_target) + 1
        curr_download = {f[striplen:]: checksum_dict[f] for f in curr_files}
        processed_downloads[download_target] = (
            size_load,
            buffer_length,
            curr_download,
            unknown_buffer_lengths,
        )

    def write_checksum(filename, file_checksum):
        try:
            file_checksum = Checksum(file_checksum)
        except Exception:
            return
        try:
            with open(filename + ".CHECKSUM", "w") as f:
                f.write(file_checksum.hex() + "\n")
        except Exception:
            msg(0, f"Cannot write checksum to file '{filename}.CHECKSUM'")
            return

    confirm_all = False
    if auto_confirm == "yes":
        confirm_all = True
    for download_target in sorted(
        processed_downloads, key=lambda k: -processed_downloads[k][0]
    ):
        size_load, buffer_length, curr_download, unknown = processed_downloads[
            download_target
        ]
        buffer_length_str = bytes2human(buffer_length)
        need_confirm = False
        if buffer_length + size_load_per_unknown_file * unknown > max_download_size:
            need_confirm = True
        if isinstance(curr_download, dict):
            nfiles = len(curr_download)
            if unknown:
                download_msg = f"'{download_target}', {nfiles} files, {buffer_length_str}, {unknown} files of length unknown"
            else:
                download_msg = (
                    f"'{download_target}', {nfiles} files, {buffer_length_str}"
                )
        else:
            nfiles = 1
            if unknown:
                download_msg = f"'{download_target}', length unknown"
            else:
                download_msg = f"'{download_target}', {buffer_length_str}"
        if nfiles > max_download_files:
            need_confirm = True
        if confirm_all:
            need_confirm = False
        if need_confirm:
            if auto_confirm == "no":
                msg(1, f"Skip download of {download_msg}")
                continue
            cs = checksum_dict[download_target]
            try:
                Checksum(cs).resolve()
            except Exception:
                msg(
                    0,
                    f"Cannot download contents of file '{download_target}', checksum '{cs}'",
                )
                continue
            try:
                confirmation = confirm_yna(f"Confirm download of {download_msg}?")
            except SeamlessSystemExit as exc:
                err(*exc.args)
            if confirmation == "no":
                msg(1, f"Skip download of {download_target}")
                continue
            if confirmation == "all":
                confirm_all = True
        if isinstance(curr_download, dict):
            os.makedirs(download_target, exist_ok=True)
            subdirs = {os.path.dirname(k) for k in curr_download}
            for subdir in subdirs:
                os.makedirs(os.path.join(download_target, subdir), exist_ok=True)
            curr_checksum_dict = {
                os.path.join(download_target, k): v for k, v in curr_download.items()
            }
            downloader = functools.partial(
                download_file,
                source_directory=source_directory,
                transfer_mode=transfer_mode,
                existing_entry_policy=existing_entry_policy,
            )
            with ThreadPoolExecutor(max_workers=20) as executor:
                list(
                    executor.map(
                        downloader,
                        curr_checksum_dict.keys(),
                        curr_checksum_dict.values(),
                    )
                )
            if index_checksums is not None:
                write_checksum(download_target, index_checksums[download_target])
        else:
            download_file(
                download_target,
                curr_download,
                source_directory=source_directory,
                transfer_mode=transfer_mode,
                existing_entry_policy=existing_entry_policy,
            )
