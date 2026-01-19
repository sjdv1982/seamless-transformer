"""Namespace utilities for transformer execution."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Tuple

from seamless import CacheMissError, Checksum

from .code_manager import get_code_manager
from .transformation_utils import unpack_deep_structure, is_deep_celltype


def _buffer_path_candidates(directory: str, checksum_hex: str) -> tuple[str, str]:
    return (
        os.path.join(directory, checksum_hex),
        os.path.join(directory, checksum_hex[:2], checksum_hex),
    )


def _get_read_folder_directories() -> list[str]:
    try:
        import seamless_remote.buffer_remote as buffer_remote
    except Exception:
        return []
    try:
        read_folders = list(buffer_remote._read_folders_clients)
    except Exception:
        return []

    directories: list[str] = []
    for client in read_folders:
        init_sync = getattr(client, "ensure_initialized_sync", None)
        if callable(init_sync):
            try:
                init_sync(skip_healthcheck=True)
            except Exception:
                pass
        directory = getattr(client, "directory", None)
        if directory:
            directories.append(os.path.expanduser(directory))
    return directories


def _find_filesystem_path(
    checksum: Checksum, mode: str, directories: list[str]
) -> str | None:
    if not directories:
        return None
    checksum = Checksum(checksum)
    if not checksum:
        return None
    checksum_hex = checksum.hex()
    for directory in directories:
        for candidate in _buffer_path_candidates(directory, checksum_hex):
            if mode == "file":
                if os.path.isfile(candidate):
                    return candidate
            elif mode == "directory":
                if os.path.isdir(candidate):
                    return candidate
    return None


def build_transformation_namespace_sync(
    transformation: Dict[str, Any],
) -> Tuple[Any, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    namespace = {
        "__name__": "transformer",
        "__package__": "transformer",
    }
    code = None
    deep_structures_to_unpack: Dict[str, Tuple[Any, str]] = {}
    namespace["PINS"] = {}
    namespace["OUTPUTPIN"] = transformation["__output__"][1]
    modules_to_build: Dict[str, Any] = {}
    as_ = transformation.get("__as__", {})
    format_section = transformation.get("__format__", {})
    FILESYSTEM: Dict[str, Any] = {}
    if isinstance(format_section, dict):
        for pinname, fmt in format_section.items():
            filesystem = fmt.get("filesystem") if isinstance(fmt, dict) else None
            if filesystem is None:
                continue
            fs_entry = dict(filesystem)
            fs_entry.setdefault("filesystem", False)
            FILESYSTEM[pinname] = fs_entry
    namespace["FILESYSTEM"] = FILESYSTEM
    read_folder_directories = _get_read_folder_directories()
    code_manager = get_code_manager()
    fallback_syntactic = transformation.get("__code_checksum__")
    fallback_code_text = transformation.get("__code_text__")
    language = transformation.get("__language__", "python")

    for pinname in sorted(transformation.keys()):
        if pinname in (
            "__compilers__",
            "__languages__",
            "__env__",
            "__as__",
            "__meta__",
            "__format__",
        ):
            continue
        if pinname in (
            "__language__",
            "__output__",
            "__code_checksum__",
            "__code_text__",
        ):
            continue

        celltype, subcelltype, checksum_value = transformation[pinname]
        if checksum_value is None:
            continue

        try:
            checksum = Checksum(checksum_value)
        except Exception as exc:
            raise RuntimeError(
                "Invalid checksum for pin "
                f"{pinname}: {checksum_value!r} ({type(checksum_value).__name__})"
            ) from exc
        if pinname == "code" and language == "python":
            if fallback_syntactic:
                try:
                    checksum = Checksum(fallback_syntactic)
                except Exception as exc:
                    raise RuntimeError(
                        "Invalid fallback code checksum: "
                        f"{fallback_syntactic!r} ({type(fallback_syntactic).__name__})"
                    ) from exc
            else:
                syntactic_options = code_manager.get_syntactic_checksums(checksum)
                if syntactic_options:
                    checksum = syntactic_options[0]

        from_filesystem = False
        fs_entry = FILESYSTEM.get(pinname)
        if fs_entry is not None:
            fs_mode = fs_entry.get("mode")
            if fs_mode:
                fs_result = _find_filesystem_path(
                    checksum, fs_mode, read_folder_directories
                )
                if fs_result is not None:
                    fs_entry["filesystem"] = True
                    value = fs_result
                    from_filesystem = True
                else:
                    fs_entry["filesystem"] = False
                    if fs_mode == "file" and not fs_entry.get("optional", False):
                        msg = f"{pinname}: could not find file for {checksum.hex()}"
                        raise CacheMissError(msg)
        if from_filesystem:
            pinname_as = as_.get(pinname, pinname)
            namespace["PINS"][pinname_as] = value
            namespace[pinname_as] = value
            continue

        try:
            buffer = checksum.resolve()
        except Exception:
            if pinname == "code" and fallback_code_text is not None:
                code = fallback_code_text
                continue
            raise
        if buffer is None:
            raise CacheMissError(checksum.hex())

        if pinname == "code":
            if language != "python":
                target_celltype = celltype or "text"
                code = buffer.get_value(target_celltype)
                continue
            value = buffer.get_value("python")
            if isinstance(value, str) and len(value) == 64:
                if os.environ.get("SEAMLESS_DEBUG_HEXCODE"):
                    import pprint

                    print(
                        "[transformer code hex]",
                        "fallback_syntactic",
                        fallback_syntactic,
                        "fallback_text",
                        fallback_code_text is not None,
                        "transformation_keys",
                        sorted(transformation.keys()),
                        file=sys.stderr,
                        flush=True,
                    )
                    pprint.pprint(transformation, stream=sys.stderr)
                candidates: list[str] = []
                if fallback_syntactic:
                    candidates.append(fallback_syntactic)
                candidates.append(value)
                for candidate in candidates:
                    try:
                        buf2 = Checksum(candidate).resolve()
                    except Exception:
                        continue
                    if buf2 is None:
                        continue
                    try:
                        value = buf2.get_value("python")
                    except Exception:
                        continue
                    break
                else:
                    raise CacheMissError(value)
                if isinstance(value, str) and len(value) == 64:
                    if fallback_code_text is not None:
                        value = fallback_code_text
                    else:
                        raise CacheMissError(value)
            code = value
            continue

        target_celltype = celltype or "mixed"
        if is_deep_celltype(celltype):
            deep_structure = buffer.get_value("plain")
            value = unpack_deep_structure(deep_structure, celltype)
            pinname_as = as_.get(pinname, pinname)
            namespace["PINS"][pinname_as] = value
            namespace[pinname_as] = value
            continue

        value = buffer.get_value(target_celltype)

        if (celltype, subcelltype) == ("plain", "module"):
            modules_to_build[pinname] = value
        else:
            pinname_as = as_.get(pinname, pinname)
            namespace["PINS"][pinname_as] = value
            namespace[pinname_as] = value

    return code, namespace, modules_to_build, deep_structures_to_unpack


__all__ = ["build_transformation_namespace_sync"]
