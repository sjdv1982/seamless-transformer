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

        header_path = os.path.join(build_dir, "public.h")
        with open(header_path, "w") as f:
            f.write(c_header)

        ffi.set_source(
            full_module_name,
            '#include "public.h"',
            extra_objects=object_paths,
            extra_link_args=list(link_options or []),
            include_dirs=[build_dir],
        )
        ffi.compile(tmpdir=build_dir, verbose=compiler_verbose)

        suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        candidates = [
            os.path.join(build_dir, filename)
            for filename in os.listdir(build_dir)
            if filename.endswith(suffix)
            or re.match(rf"{re.escape(full_module_name)}.*\.so$", filename)
        ]
        if not candidates:
            for root, _dirs, files in os.walk(build_dir):
                for filename in files:
                    if filename.endswith(".so"):
                        candidates.append(os.path.join(root, filename))
        if not candidates:
            raise RuntimeError("CFFI build did not produce an extension module")
        with open(candidates[0], "rb") as f:
            return f.read()


__all__ = ["build_extension_cffi"]
