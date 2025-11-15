"""Helpers for working with transformation dicts."""

from __future__ import annotations

from typing import Dict, Any

from seamless import Buffer, Checksum


DEEP_CELLTYPES = ("deepcell", "deepfolder")


def tf_get_buffer(transformation: Dict[str, Any]) -> Buffer:
    """Serialize a transformation dict into a Buffer."""

    assert isinstance(transformation, dict)
    result: Dict[str, Any] = {}
    for key, value in transformation.items():
        if key in (
            "__compilers__",
            "__languages__",
            "__meta__",
            "__env__",
            "__code_checksum__",
        ):
            continue
        if key in ("__language__", "__output__", "__as__", "__format__"):
            result[key] = value
            continue
        if key.startswith("SPECIAL__"):
            continue

        celltype, subcelltype, checksum = value
        if isinstance(checksum, Checksum):
            checksum = checksum.hex()
        result[key] = (celltype, subcelltype, checksum)

    return Buffer(result, celltype="plain")


def is_deep_celltype(celltype: str | None) -> bool:
    return celltype in DEEP_CELLTYPES


def unpack_deep_structure(structure, celltype: str):
    """Resolve a deep structure (dict/list of checksums) into concrete values."""

    if not is_deep_celltype(celltype) or structure is None:
        return structure

    def _convert(value):
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_convert(v) for v in value]
        checksum = Checksum(value)
        buffer = checksum.resolve()
        if celltype == "deepfolder":
            content = buffer.content
            if content is None:
                content = buffer.get_value("plain")
        else:
            content = buffer.get_value("mixed")
        return content

    return _convert(structure)


def pack_deep_structure(structure, celltype: str):
    """Convert a deep value into a dict/list of checksum hex strings."""

    if not is_deep_celltype(celltype) or structure is None:
        return structure

    def _pack(value):
        if isinstance(value, dict):
            return {k: _pack(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_pack(v) for v in value]
        if isinstance(value, str) and len(value) == 64:
            return value
        buffer = Buffer(value, "mixed" if celltype == "deepcell" else None)
        checksum = buffer.get_checksum()
        return checksum.hex()

    return _pack(structure)


__all__ = [
    "tf_get_buffer",
    "is_deep_celltype",
    "unpack_deep_structure",
    "pack_deep_structure",
]
