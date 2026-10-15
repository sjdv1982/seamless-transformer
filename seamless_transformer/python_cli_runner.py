"""Runner staged into each Python CLI transformation job."""

from __future__ import annotations

from collections.abc import Mapping
from io import BytesIO
import importlib
import json
import math
import os
from pathlib import Path
import sys


def _read_bytes(path):
    with open(path, "rb") as handle:
        return handle.read()


def _read_folder(path):
    result = {}
    for directory, dirs, filenames in os.walk(path):
        dirs.sort()
        filenames.sort()
        for filename in filenames:
            full_path = os.path.join(directory, filename)
            member = os.path.relpath(full_path, path).replace(os.sep, "/")
            result[member] = _read_bytes(full_path)
    return result


def _decode_pin(spec):
    filename = spec["file"]
    encoding = spec["encoding"]
    if encoding == "file":
        return str(Path(filename).resolve())
    if encoding == "folder":
        return _read_folder(filename)
    data = _read_bytes(filename)
    if encoding == "json":
        return json.loads(data.decode("utf-8"))
    if encoding == "text":
        return data.decode("utf-8")
    if encoding == "bytes":
        return data
    if encoding == "npy":
        import numpy as np

        return np.load(BytesIO(data), allow_pickle=False)
    if encoding == "mixed":
        helper = importlib.import_module("seamless_core_helper")
        return helper.decode_mixed(data)
    if encoding in ("deepcell", "checksum"):
        helper = importlib.import_module("seamless_core_helper")
        return helper.decode_checksum_pin(data, encoding)
    raise ValueError(f"Unsupported Python CLI pin encoding {encoding!r}")


def _json_value(value):
    if value is None or type(value) in (bool, int, str):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_json_value(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _json_value(item) for key, item in value.items())
    return False


def _json_bytes(value, label):
    if not _json_value(value):
        raise TypeError(f"Python CLI result {label!r} is not JSON-encodable")
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _checksum_hex(value):
    module = sys.modules.get("seamless")
    checksum_class = getattr(module, "Checksum", None) if module is not None else None
    if isinstance(checksum_class, type) and isinstance(value, checksum_class):
        return value.hex()
    return None


def _structured_dtype_has_padding(dtype):
    subdtype = getattr(dtype, "subdtype", None)
    if subdtype is not None:
        return _structured_dtype_has_padding(subdtype[0])

    fields = getattr(dtype, "fields", None)
    if fields is None:
        return False

    offset = 0
    for name in dtype.names or ():
        field_dtype, field_offset = fields[name][:2]
        if field_offset != offset or _structured_dtype_has_padding(field_dtype):
            return True
        offset = field_offset + field_dtype.itemsize
    return offset != dtype.itemsize


def _contains_checksum(value):
    if _checksum_hex(value) is not None:
        return True
    if isinstance(value, Mapping):
        return any(_contains_checksum(child) for child in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_checksum(child) for child in value)
    return False


def _contains_padded_structured_array(value):
    numpy_module = sys.modules.get("numpy")
    ndarray_type = getattr(numpy_module, "ndarray", None)
    if isinstance(ndarray_type, type) and isinstance(value, ndarray_type):
        return (
            value.dtype.fields is not None
            and _structured_dtype_has_padding(value.dtype)
        )
    if isinstance(value, Mapping):
        return any(
            _contains_padded_structured_array(child)
            for child in value.values()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_padded_structured_array(child) for child in value)
    return False


def _write_result_value(value, label, files, counter):
    checksum = _checksum_hex(value)
    if checksum is not None:
        return {"__checksum__": checksum}

    if _contains_padded_structured_array(value):
        raise TypeError(
            f"Python CLI result {label!r} contains a structured NumPy "
            "array with padding that cannot be checksum-preserved"
        )

    if isinstance(value, bytes):
        suffix = ".bytes"
        data = value
    else:
        numpy_module = sys.modules.get("numpy")
        ndarray_type = getattr(numpy_module, "ndarray", None)
        if isinstance(ndarray_type, type) and isinstance(value, ndarray_type):
            if value.dtype.fields is not None:
                # NumPy's .npy loader drops dtype.isalignedstruct, which
                # changes the Seamless checksum of aligned structured arrays.
                helper = importlib.import_module("seamless_core_helper")
                suffix = ".mixed"
                data = helper.encode_mixed(value)
            else:
                np = numpy_module
                suffix = ".npy"
                buffer = BytesIO()
                np.save(buffer, value, allow_pickle=False)
                data = buffer.getvalue()
        elif _json_value(value):
            return None
        else:
            helper = importlib.import_module("seamless_core_helper")
            suffix = ".mixed"
            data = helper.encode_mixed(value)

    if not isinstance(data, bytes):
        raise TypeError(f"Python CLI result {label!r} did not encode to bytes")
    filename = f"m{counter[0]:06d}{suffix}"
    counter[0] += 1
    files[filename] = data
    return filename


def _write_result(result, output_celltype):
    if output_celltype not in ("deepcell", "folder", "checksum") and _contains_checksum(result):
        raise TypeError(
            f"A Checksum can only be returned under a deep or checksum result, "
            f"not {output_celltype!r}"
        )

    files = {}
    counter = [0]
    if output_celltype in ("folder", "deepcell"):
        if not isinstance(result, dict):
            raise TypeError(
                f"Python CLI {output_celltype} result must be a dict"
            )
        members = {}
        inline = {}
        for key, value in result.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"Python CLI {output_celltype} result key {key!r} must be a string"
                )
            encoded = _write_result_value(value, key, files, counter)
            if encoded is None:
                inline[key] = value
            else:
                members[key] = encoded
        index = {"members": members}
        if inline:
            filename = f"m{counter[0]:06d}.json"
            counter[0] += 1
            files[filename] = _json_bytes(inline, "members")
            index["json"] = filename
    else:
        encoded = _write_result_value(result, "value", files, counter)
        if encoded is None:
            filename = f"m{counter[0]:06d}.json"
            counter[0] += 1
            files[filename] = _json_bytes(result, "value")
            encoded = filename
        index = {"value": encoded}
    files["index.json"] = json.dumps(
        index,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return files


def _configure_remote_clients():
    if "SEAMLESS_REMOTE_CLIENTS" not in os.environ:
        return
    try:
        from seamless_config.extern_clients import set_remote_clients_from_env
    except ImportError:
        return
    set_remote_clients_from_env(include_dask=True)


def main():
    with open("manifest.json", encoding="utf-8") as handle:
        manifest = json.load(handle)

    namespace = {
        "__name__": "transformer",
        "__package__": "transformer",
        "PINS": {},
        "FILESYSTEM": manifest["filesystem"],
        "OUTPUTPIN": manifest["outputpin"],
    }
    for spec in manifest["pins"]:
        value = _decode_pin(spec)
        namespace["PINS"][spec["name"]] = value
        namespace[spec["name"]] = value

    _configure_remote_clients()

    with open("program.py", encoding="utf-8") as handle:
        program = handle.read()
    code = compile(program, "program.py", "exec")
    exec(code, namespace)
    output_name = manifest["output"]
    try:
        result = namespace[output_name]
    except KeyError:
        raise RuntimeError(f"Output variable name '{output_name}' undefined") from None

    files = _write_result(result, manifest["output_celltype"])
    result_dir = Path("RESULT")
    result_dir.mkdir()
    for filename, data in files.items():
        with open(result_dir / filename, "wb") as handle:
            handle.write(data)


if __name__ == "__main__":
    main()
