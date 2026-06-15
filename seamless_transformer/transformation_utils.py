"""Helpers for working with transformation dicts."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Any

from seamless import Buffer, Checksum


DEEP_CELLTYPES = ("deepcell", "deepfolder", "folder")
TRANSFORMATION_LOAD_BEARING_DUNDER_KEYS = {
    "__language__",
    "__output__",
    "__as__",
    "__format__",
    "__schema__",
}
TRANSFORMATION_ORTHOGONAL_DUNDER_KEYS = {
    "__meta__",
    "__env__",
    "__compilation__",
    "__record_probe__",
    "__code_checksum__",
    "__code_text__",
    "__compilers__",
    "__languages__",
}
TRANSFORMATION_DERIVED_DUNDER_KEYS = {
    "__compiled__",
    "__header__",
    "__deps__",
}
# Backward-compatible export names used by older modules. "Core" now means
# checksum-defining dunders; "execution" means dunders carried outside identity.
TRANSFORMATION_CORE_KEYS = TRANSFORMATION_LOAD_BEARING_DUNDER_KEYS
TRANSFORMATION_LOCAL_DUNDER_KEYS = {"__meta__", "__env__"}
TRANSFORMATION_EXECUTION_DUNDER_KEYS = (
    TRANSFORMATION_ORTHOGONAL_DUNDER_KEYS | TRANSFORMATION_DERIVED_DUNDER_KEYS
)


def tf_get_buffer(transformation: Dict[str, Any]) -> Buffer:
    """Serialize the checksum-defining payload of a transformation dict."""

    assert isinstance(transformation, dict)
    result: Dict[str, Any] = {}
    for key, value in transformation.items():
        if key in TRANSFORMATION_LOAD_BEARING_DUNDER_KEYS:
            result[key] = value
            continue
        if key in TRANSFORMATION_ORTHOGONAL_DUNDER_KEYS:
            continue
        if key in TRANSFORMATION_DERIVED_DUNDER_KEYS:
            continue
        if key.startswith("META__"):
            continue
        if key.startswith("__"):
            raise ValueError(f"Unknown transformation dunder key: {key}")

        celltype, subcelltype, checksum = value
        if isinstance(checksum, Checksum):
            checksum = checksum.hex()
        result[key] = (celltype, subcelltype, checksum)

    return Buffer(result, celltype="plain")


def json_null_checksum() -> Checksum:
    """Return the canonical checksum used for JSON null optional-pin absence."""

    return Buffer(None, "plain").get_checksum()


def normalize_optional_pins_for_construction(
    transformation_dict: Dict[str, Any], optional_pins
) -> Dict[str, Any]:
    """Drop connected optional pins that resolved to canonical JSON null.

    Optional pin metadata is intentionally external to the checksum-defining
    transformation payload. Connected optional pins still resolve normally; only
    a successful plain/mixed JSON-null result is canonicalized to pin absence.
    """

    optional_pin_names = frozenset(optional_pins or ())
    if not optional_pin_names:
        return transformation_dict
    null_checksum = json_null_checksum()
    null_checksum_hex = null_checksum.hex()
    for pinname in optional_pin_names:
        value = transformation_dict.get(pinname)
        if value is None:
            continue
        celltype, _subcelltype, checksum = value
        if checksum is None:
            continue
        if isinstance(checksum, Checksum):
            checksum_hex = checksum.hex()
        else:
            checksum_hex = checksum
        if checksum_hex != null_checksum_hex:
            continue
        if celltype not in ("plain", "mixed"):
            raise TypeError(
                f"Optional pin '{pinname}' with celltype '{celltype}' "
                "cannot use JSON null as absence"
            )
        transformation_dict.pop(pinname, None)
    return transformation_dict


def sufficiently_connected(required_pins, optional_pins, wired_pins) -> bool:
    """Return true when every required pin is wired.

    Connected optional pins are represented by membership in ``wired_pins`` and
    therefore gate through normal dependency construction. Unwired optional pins
    are acceptable and absent.
    """

    required = set(required_pins or ()) - set(optional_pins or ())
    return required.issubset(set(wired_pins or ()))


def extract_tf_dunder(transformation: Dict[str, Any]) -> Dict[str, Any]:
    """Return non-identity dunder payload for worker/jobserver transport."""

    return {
        key: deepcopy(value)
        for key, value in transformation.items()
        if key in TRANSFORMATION_EXECUTION_DUNDER_KEYS or key.startswith("META__")
    }


def extract_job_dunder(transformation: Dict[str, Any]) -> Dict[str, Any]:
    """Return dunder payload required by standalone job directories."""

    return {
        key: deepcopy(value)
        for key, value in transformation.items()
        if (
            key.startswith("META__")
            or key in TRANSFORMATION_EXECUTION_DUNDER_KEYS
            and key != "__env__"
        )
    }


def merge_transformation_meta(
    transformation: Dict[str, Any], tf_dunder: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """Merge intrinsic transformation metadata with execution-side overrides."""

    meta: Dict[str, Any] = {}
    meta0 = transformation.get("__meta__")
    if isinstance(meta0, dict):
        meta.update(deepcopy(meta0))
    if isinstance(tf_dunder, dict):
        meta1 = tf_dunder.get("__meta__")
        if isinstance(meta1, dict):
            meta.update(deepcopy(meta1))
    return meta


def resolve_env_checksum(
    transformation: Dict[str, Any], tf_dunder: Dict[str, Any] | None = None
):
    """Resolve the env checksum, preferring the transformation dict."""

    env_checksum = transformation.get("__env__")
    if env_checksum is None and isinstance(tf_dunder, dict):
        env_checksum = tf_dunder.get("__env__")
    return env_checksum


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
        if celltype == "deepcell":
            content = buffer.get_value("mixed")
        else:
            content = buffer.content
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
        buffer.tempref()
        return checksum.hex()

    return _pack(structure)


__all__ = [
    "tf_get_buffer",
    "json_null_checksum",
    "normalize_optional_pins_for_construction",
    "sufficiently_connected",
    "extract_tf_dunder",
    "extract_job_dunder",
    "merge_transformation_meta",
    "resolve_env_checksum",
    "is_deep_celltype",
    "unpack_deep_structure",
    "pack_deep_structure",
    "TRANSFORMATION_LOAD_BEARING_DUNDER_KEYS",
    "TRANSFORMATION_ORTHOGONAL_DUNDER_KEYS",
    "TRANSFORMATION_DERIVED_DUNDER_KEYS",
    "TRANSFORMATION_EXECUTION_DUNDER_KEYS",
]
