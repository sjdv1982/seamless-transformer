"""CFFI wrapper generation for compiled transformers."""

from __future__ import annotations

import os
import re
import sysconfig
import tempfile


def _cdef_from_header(c_header: str) -> str:
    lines = []
    for line in c_header.splitlines():
        if line.lstrip().startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines)


def build_extension_cffi(
    full_module_name: str,
    binary_objects: dict[str, bytes],
    target: str,
    c_header: str,
    link_options: list[str],
    compiler_verbose: bool = False,
) -> bytes:
    """Build a Python extension .so from binary objects and a C header."""

    try:
        from cffi import FFI
    except ImportError:
        raise ImportError(
            "cffi is required for compiled transformers. Install it with: pip install cffi"
        ) from None

    ffi = FFI()
    ffi.cdef(_cdef_from_header(c_header))

    with tempfile.TemporaryDirectory() as build_dir:
        object_paths = []
        for name, data in binary_objects.items():
            path = os.path.join(build_dir, name)
            with open(path, "wb") as f:
                f.write(data)
            object_paths.append(path)

        ffi.set_source(
            full_module_name,
            c_header,
            extra_objects=object_paths,
            extra_link_args=list(link_options or []),
        )
        output_path = ffi.compile(tmpdir=build_dir, verbose=compiler_verbose)
        with open(output_path, "rb") as f:
            return f.read()


__all__ = ["build_extension_cffi"]
