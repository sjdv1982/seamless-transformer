"""Internal helpers for values crossing a language or process boundary."""

from __future__ import annotations

import io
import json
import re
from collections.abc import Mapping

from seamless import Checksum


_CHECKSUM_RE = re.compile(r"^[0-9a-f]{64}$")
_FILE_ENCODINGS = {".json", ".bytes", ".npy", ".jsonarray", ".mixed"}


def _checksum_entry(value, label):
    if not isinstance(value, Mapping) or set(value) != {"__checksum__"}:
        return None
    checksum = value["__checksum__"]
    if not isinstance(checksum, str) or _CHECKSUM_RE.fullmatch(checksum) is None:
        raise ValueError(f"Malformed checksum entry {label!r}")
    return Checksum(checksum)


def _decode_file(filename, files, label):
    if not isinstance(filename, str):
        raise ValueError(f"Malformed file entry {label!r}: expected a file name")
    if "/" in filename or "\\" in filename or filename in ("", ".", ".."):
        raise ValueError(f"Malformed file entry {label!r}: unsafe file name {filename!r}")
    suffix = next((suffix for suffix in _FILE_ENCODINGS if filename.endswith(suffix)), None)
    if suffix is None:
        raise ValueError(f"Malformed file entry {label!r}: unknown encoding in {filename!r}")
    if filename not in files:
        raise ValueError(f"Malformed file entry {label!r}: file {filename!r} is missing")
    data = files[filename]
    if not isinstance(data, bytes):
        raise TypeError(f"Wire file {filename!r} for {label!r} must contain bytes")
    try:
        if suffix == ".bytes":
            return data
        if suffix == ".json":
            return json.loads(data.decode("utf-8"))
        if suffix == ".npy":
            import numpy as np

            return np.load(io.BytesIO(data), allow_pickle=False)
        if suffix == ".jsonarray":
            return decode_jsonarray(json.loads(data.decode("utf-8")))
        if suffix == ".mixed":
            from seamless.util.mixed.io import deserialize

            return deserialize(data)[0]
    except Exception as exc:
        raise ValueError(f"Cannot decode wire entry {label!r} from {filename!r}: {exc}") from exc
    raise AssertionError(suffix)


def decode_jsonarray(value):
    """Decode the R/Julia column-major array payload, with or without its marker."""
    if not isinstance(value, Mapping):
        raise ValueError("Malformed JSON array payload: expected an object")
    required = {"dtype", "shape", "values"}
    if value.get("__seamless_array__") is True:
        missing = required - set(value)
        if missing:
            raise ValueError(
                "Malformed JSON array payload: missing required fields "
                f"{sorted(missing)!r}"
            )
        dtype = value.get("dtype")
        shape = value.get("shape")
        values = value.get("values")
    elif required.issubset(value):
        dtype = value.get("dtype")
        shape = value.get("shape")
        values = value.get("values")
    else:
        missing = required - set(value)
        raise ValueError(
            "Malformed JSON array payload: missing required fields "
            f"{sorted(missing)!r}"
        )
    import numpy as np

    dtypes = {
        "bool": np.bool_,
        "int64": np.int64,
        "uint64": np.uint64,
        "float64": np.float64,
    }
    if dtype not in dtypes:
        raise ValueError(f"Unknown JSON array dtype {dtype!r}")
    if not isinstance(shape, (list, tuple)) or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 0
        for size in shape
    ):
        raise ValueError(f"Malformed JSON array shape {shape!r}")
    if not isinstance(values, list):
        raise ValueError("Malformed JSON array values")
    array = np.asarray(values, dtype=dtypes[dtype])
    try:
        return np.ascontiguousarray(array.reshape(tuple(shape), order="F"))
    except ValueError as exc:
        raise ValueError("JSON array shape does not match its values") from exc


def decode_result_payload(value):
    """Decode the recursive JSON markers used by R and Julia result payloads."""
    if isinstance(value, Mapping):
        if value.get("__seamless_bytes__") is True and set(value) == {
            "__seamless_bytes__",
            "values",
        }:
            raw_values = value["values"]
            if not isinstance(raw_values, list) or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or item < 0
                or item > 255
                for item in raw_values
            ):
                raise ValueError("Malformed result bytes payload")
            return bytes(raw_values)
        if value.get("__seamless_array__") is True:
            return decode_jsonarray(value)
        return {key: decode_result_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_result_payload(item) for item in value]
    return value


def read_wire_directory(files, output_celltype):
    """Decode an index and its named files from an in-memory file mapping."""
    if not isinstance(files, Mapping):
        raise TypeError("wire directory must be a mapping from names to bytes")
    if "index.json" not in files:
        raise ValueError("Malformed wire entry 'index.json': file is missing")
    raw_index = files["index.json"]
    if not isinstance(raw_index, bytes):
        raise TypeError("Wire file 'index.json' must contain bytes")
    try:
        index = json.loads(raw_index.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Malformed wire entry 'index.json': {exc}") from exc
    if not isinstance(index, Mapping):
        raise ValueError("Malformed wire entry 'index.json': expected a JSON object")
    unknown_top_keys = set(index) - {"members", "value", "json"}
    if unknown_top_keys:
        key = sorted(unknown_top_keys, key=str)[0]
        raise ValueError(f"Malformed wire entry {key!r}: unknown top-level key")
    has_members = "members" in index
    has_value = "value" in index
    if has_members == has_value:
        raise ValueError("Malformed wire entry 'index.json': expected exactly one of members or value")
    is_deep_result = output_celltype in ("folder", "deepcell")
    if has_members and not is_deep_result:
        raise ValueError("Malformed wire entry 'members': result celltype is not deep")
    if has_value and is_deep_result:
        raise ValueError("Malformed wire entry 'value': result celltype is deep")

    if has_value:
        if "json" in index:
            raise ValueError("Malformed wire entry 'json': only allowed with members")
        value_entry = index["value"]
        checksum = _checksum_entry(value_entry, "value")
        if checksum is not None:
            return checksum
        return _decode_file(value_entry, files, "value")

    members = index["members"]
    if not isinstance(members, Mapping):
        raise ValueError("Malformed wire entry 'members': expected an object")
    result = {}
    for key, entry in members.items():
        if not isinstance(key, str):
            raise ValueError(f"Malformed wire entry {key!r}: member key must be a string")
        checksum = _checksum_entry(entry, key)
        if checksum is not None:
            result[key] = checksum
        elif isinstance(entry, str):
            result[key] = _decode_file(entry, files, key)
        else:
            raise ValueError(f"Malformed wire entry {key!r}: expected checksum or file name")

    if "json" in index:
        inline = _decode_file(index["json"], files, "json")
        if not isinstance(inline, Mapping):
            raise ValueError("Malformed wire entry 'json': file must contain a JSON object")
        for key, value in inline.items():
            if not isinstance(key, str):
                raise ValueError(f"Malformed wire entry {key!r}: member key must be a string")
            if key in result:
                raise ValueError(f"Malformed wire entry {key!r}: key appears in members and json")
            result[key] = value
    return result


def write_deepcell_pin_index(value):
    """Return the JSON index that helper A reads for a deepcell pin."""
    if not isinstance(value, Mapping):
        raise TypeError("deepcell pin must be a mapping")
    members = {}
    for key, checksum in value.items():
        if not isinstance(key, str):
            raise TypeError(f"Deepcell pin member key {key!r} must be a string")
        if not isinstance(checksum, Checksum):
            raise TypeError(f"Deepcell pin member {key!r} must be a Checksum")
        members[key] = {"__checksum__": checksum.hex()}
    return json.dumps({"members": members}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def write_checksum_pin_index(value):
    """Return the JSON index that helper A reads for a checksum pin."""
    if not isinstance(value, Checksum):
        raise TypeError("checksum pin must be a Checksum")
    return json.dumps(
        {"value": {"__checksum__": value.hex()}},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


__all__ = [
    "decode_jsonarray",
    "read_wire_directory",
    "write_deepcell_pin_index",
    "write_checksum_pin_index",
]
