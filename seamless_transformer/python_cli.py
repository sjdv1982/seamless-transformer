"""Host side of the Python CLI backend."""

from __future__ import annotations

from collections.abc import Mapping
from io import BytesIO
import json
import os
from pathlib import Path

from seamless import Buffer, Checksum

from .cached_compile import build_exec_code
from .wire_format import (
    read_wire_directory,
    write_checksum_pin_index,
    write_deepcell_pin_index,
)


_SKIP_PIN_NAMES = {
    "code",
    "objects",
    "__language__",
    "__output__",
    "__code_checksum__",
    "__code_text__",
    "__compilers__",
    "__languages__",
    "__env__",
    "__as__",
    "__meta__",
    "__format__",
}


def _json_value(value):
    if value is None or type(value) in (bool, int, str):
        return True
    if type(value) is float:
        import math

        return math.isfinite(value)
    if type(value) is list:
        return all(_json_value(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _json_value(item) for key, item in value.items())
    return False


def _json_bytes(value, *, label):
    if not _json_value(value):
        raise TypeError(f"Python CLI {label} is not JSON-encodable")
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _buffer_bytes(value):
    if isinstance(value, Buffer):
        return value.content
    if isinstance(value, bytes):
        return value
    raise TypeError(f"Expected a bytes buffer, got {type(value).__name__}")


def _as_checksum(value):
    if isinstance(value, Checksum):
        return value
    return Checksum(value)


def _stage_bytes(stage_pins, filename, data):
    stage_pins["META__FILE__" + filename] = data


def _copy_filesystem(filesystem):
    copied = {}
    for name, entry in filesystem.items():
        if not isinstance(name, str) or not isinstance(entry, Mapping):
            raise TypeError("Python CLI FILESYSTEM must map names to mappings")
        copied[name] = dict(entry)
    return copied


def _pin_specs(transformation, namespace, stage_pins, stage_filesystem):
    pins = namespace.get("PINS", {})
    filesystem = namespace.get("FILESYSTEM", {})
    as_map = transformation.get("__as__", {})
    specs = []
    numpy_needed = False
    core_needed = False

    for ordinal, pinname in enumerate(sorted(transformation)):
        if pinname in _SKIP_PIN_NAMES or pinname.startswith("__"):
            continue
        entry = transformation[pinname]
        if not isinstance(entry, (tuple, list)) or len(entry) != 3:
            continue
        celltype, _subcelltype, checksum_value = entry
        if checksum_value is None:
            continue
        pinname_as = as_map.get(pinname, pinname)
        if pinname_as not in pins:
            continue
        value = pins[pinname_as]
        input_file = f"pins/pin{ordinal:04d}"
        encoding = None

        fs_entry = filesystem.get(pinname, {})
        if (
            isinstance(fs_entry, Mapping)
            and fs_entry.get("mode") == "file"
            and fs_entry.get("filesystem", False)
        ):
            if not isinstance(value, (str, os.PathLike)):
                raise TypeError(f"Python CLI file pin {pinname!r} must be a path")
            stage_pins[input_file] = os.fspath(value)
            stage_filesystem[input_file] = {
                **dict(fs_entry),
                "mode": "file",
                "filesystem": True,
            }
            encoding = "file"
        elif celltype == "folder":
            if not isinstance(value, Mapping):
                raise TypeError(f"Python CLI folder pin {pinname!r} must be a mapping")
            stage_pins[input_file] = dict(value)
            stage_filesystem[input_file] = {
                "mode": "directory",
                "filesystem": False,
                "celltype": "folder",
                "members_are_checksums": False,
            }
            encoding = "folder"
        elif celltype == "deepfolder":
            if not isinstance(value, Mapping):
                raise TypeError(
                    f"Python CLI deepfolder pin {pinname!r} must be a mapping"
                )
            path_map = {}
            for member, path in value.items():
                if not isinstance(member, str) or not isinstance(path, (str, os.PathLike)):
                    raise TypeError(
                        f"Python CLI deepfolder pin {pinname!r} must map strings to paths"
                    )
                path_map[member] = os.fspath(path)
            stage_pins[input_file] = path_map
            # execute_cli uses this entry to check that container jobs can
            # follow every path. "metadata" keeps write_bash_job from
            # materializing the paths as a directory pin.
            stage_filesystem[input_file] = {
                "mode": "metadata",
                "filesystem": False,
                "celltype": "deepfolder",
            }
            encoding = "deepfolder"
        elif celltype == "deepcell":
            _stage_bytes(
                stage_pins,
                input_file,
                write_deepcell_pin_index(value),
            )
            encoding = "deepcell"
            core_needed = True
        elif celltype == "checksum":
            _stage_bytes(
                stage_pins,
                input_file,
                write_checksum_pin_index(_as_checksum(value)),
            )
            encoding = "checksum"
            core_needed = True
        elif celltype == "bytes":
            _stage_bytes(stage_pins, input_file, _buffer_bytes(value))
            encoding = "bytes"
        elif celltype == "binary":
            import numpy as np

            if not isinstance(value, np.ndarray):
                raise TypeError(f"Python CLI binary pin {pinname!r} must be a NumPy array")
            array_file = BytesIO()
            np.save(array_file, value, allow_pickle=False)
            _stage_bytes(stage_pins, input_file, array_file.getvalue())
            encoding = "npy"
            numpy_needed = True
        elif celltype == "text":
            if not isinstance(value, str):
                raise TypeError(f"Python CLI text pin {pinname!r} must be text")
            _stage_bytes(stage_pins, input_file, value.encode("utf-8"))
            encoding = "text"
        elif celltype == "array":
            import numpy as np

            if not isinstance(value, np.ndarray):
                raise TypeError(f"Python CLI array pin {pinname!r} must be a NumPy array")
            array_file = BytesIO()
            np.save(array_file, value, allow_pickle=False)
            _stage_bytes(stage_pins, input_file, array_file.getvalue())
            encoding = "npy"
            numpy_needed = True
        elif celltype == "mixed":
            raw = _as_checksum(checksum_value).resolve().content
            if raw.startswith(b"\x94SEAMLESS-MIXED"):
                _stage_bytes(stage_pins, input_file, raw)
                encoding = "mixed"
                core_needed = True
            elif raw.startswith(b"\x93NUMPY"):
                _stage_bytes(stage_pins, input_file, raw)
                encoding = "npy"
                numpy_needed = True
            elif _json_value(value):
                _stage_bytes(stage_pins, input_file, _json_bytes(value, label="pin"))
                encoding = "json"
            elif isinstance(value, bytes):
                _stage_bytes(stage_pins, input_file, value)
                encoding = "bytes"
            else:
                # Nonstandard mixed buffers are carried in their original form;
                # only Seamless-mixed containers require the helper.
                _stage_bytes(stage_pins, input_file, raw)
                encoding = "mixed"
                core_needed = True
        else:
            import numpy as np

            if isinstance(value, np.ndarray):
                array_file = BytesIO()
                np.save(array_file, value, allow_pickle=False)
                _stage_bytes(stage_pins, input_file, array_file.getvalue())
                encoding = "npy"
                numpy_needed = True
            elif isinstance(value, bytes):
                _stage_bytes(stage_pins, input_file, value)
                encoding = "bytes"
            elif isinstance(value, str):
                _stage_bytes(stage_pins, input_file, value.encode("utf-8"))
                encoding = "text"
            elif _json_value(value):
                _stage_bytes(stage_pins, input_file, _json_bytes(value, label="pin"))
                encoding = "json"
            else:
                raw = _as_checksum(checksum_value).resolve().content
                _stage_bytes(stage_pins, input_file, raw)
                encoding = "mixed"
                core_needed = True

        specs.append(
            {
                "name": pinname_as,
                "file": input_file,
                "encoding": encoding,
            }
        )

    return specs, numpy_needed, core_needed


def execute_python_cli(
    transformation,
    code,
    identifier,
    namespace,
    inputs,
    output_name,
    output_celltype,
    environment,
):
    """Run Python code in a CLI job and decode its wire-format result."""
    from .languages import execute_cli

    stage_pins = {}
    stage_filesystem = {}
    pin_specs, numpy_needed, core_needed = _pin_specs(
        transformation,
        namespace,
        stage_pins,
        stage_filesystem,
    )

    generated_code = build_exec_code(code, identifier, inputs, output_name)
    _stage_bytes(stage_pins, "program.py", generated_code.encode("utf-8"))
    _stage_bytes(
        stage_pins,
        "runner.py",
        Path(__file__).with_name("python_cli_runner.py").read_bytes(),
    )
    _stage_bytes(
        stage_pins,
        "seamless_core_helper.py",
        Path(__file__).with_name("seamless_core_helper.py").read_bytes(),
    )

    manifest = {
        "pins": pin_specs,
        "inputs": list(inputs),
        "output": output_name,
        "output_celltype": output_celltype,
        "outputpin": namespace.get("OUTPUTPIN", output_celltype),
        "filesystem": _copy_filesystem(namespace.get("FILESYSTEM", {})),
        "numpy_needed": numpy_needed,
        "core_needed": core_needed,
    }
    _stage_bytes(
        stage_pins,
        "manifest.json",
        json.dumps(
            manifest,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8"),
    )

    files = execute_cli(
        "python3 -u runner.py",
        sorted(stage_pins),
        stage_pins,
        stage_filesystem,
        "folder",
        environment,
    )
    return read_wire_directory(files, output_celltype)


__all__ = ["execute_python_cli"]
