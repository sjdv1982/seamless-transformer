"""Helpers for working with transformation dicts."""

from __future__ import annotations

from typing import Dict, Any

from seamless import Buffer, Checksum


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


__all__ = ["tf_get_buffer"]
