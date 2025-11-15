"""Namespace utilities for transformer execution."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from seamless import CacheMissError, Checksum

from .code_manager import get_code_manager
from .transformation_utils import unpack_deep_structure, is_deep_celltype


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
    namespace["OUTPUTPIN"] = transformation["__output__"][1], None
    modules_to_build: Dict[str, Any] = {}
    as_ = transformation.get("__as__", {})
    namespace["FILESYSTEM"] = {}
    code_manager = get_code_manager()

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
        if pinname in ("__language__", "__output__", "__code_checksum__"):
            continue

        celltype, subcelltype, checksum_value = transformation[pinname]
        if checksum_value is None:
            continue

        checksum = Checksum(checksum_value)
        if pinname == "code":
            syntactic_options = code_manager.get_syntactic_checksums(checksum)
            checksum = syntactic_options[0] if syntactic_options else checksum

        buffer = checksum.resolve()
        if buffer is None:
            raise CacheMissError(checksum.hex())

        if pinname == "code":
            value = buffer.get_value("python")
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
