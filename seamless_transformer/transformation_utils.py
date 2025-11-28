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
        ):
            continue
        if key in (
            "__language__",
            "__output__",
            "__as__",
            "__format__",
            "__code_checksum__",
        ):
            # TODO:
            # __code_checksum__ is not really part of the transformation dict
            #   two transformations with the same "code" but different __code_checksum__
            #   have a different syntactic Python source buffer but the same AST (semantic buffer)
            #   (this is the case for code after whitespace reformatting)
            #     => __code_checksum__ must go to tf_dunder and be stripped from tf_get_buffer
            #   HOWEVER: this will break linecache tracebacks for spawn and jobserver
            #     (which surely must already be the case for nested transformations: test needed)
            #   => sem2syn needs to be added to database.
            #           Keep a per-client cache list of all written sem2syn, but otherwise do a blocking await
            #      Maybe setup sem2syn messaging between spawn process and main, but probably YAGNI
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
        assert isinstance(buffer, Buffer)
        if celltype == "deepfolder":
            pass
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
