"""Build and call compiled transformer modules."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
import os
import tempfile
from types import ModuleType
from typing import Any

from .compile import complete, compile
from .cffi_wrapper import build_extension_cffi


_BINARY_CACHE: dict[str, dict[str, bytes]] = {}
_MODULE_CACHE: dict[str, ModuleType] = {}
_SO_CACHE_DIR = tempfile.mkdtemp(prefix="seamless-compiled-modules-")


def _stable_digest(value: Any) -> str:
    try:
        payload = json.dumps(value, sort_keys=True, default=repr).encode()
    except TypeError:
        payload = repr(value).encode()
    return hashlib.sha256(payload).hexdigest()


def _import_extension_module(full_module_name: str, so_bytes: bytes) -> ModuleType:
    path = os.path.join(_SO_CACHE_DIR, full_module_name + ".so")
    with open(path, "wb") as f:
        f.write(so_bytes)
    spec = importlib.util.spec_from_file_location(full_module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import extension module {full_module_name!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_compiled_module(
    module_definition: dict,
    *,
    module_name: str | None = None,
) -> ModuleType:
    """Build and import a compiled Python extension module."""

    completed = complete(module_definition)
    digest = _stable_digest(completed)
    if digest in _MODULE_CACHE:
        return _MODULE_CACHE[digest]

    if digest not in _BINARY_CACHE:
        _BINARY_CACHE[digest] = compile(completed)
    binary_objects = _BINARY_CACHE[digest]

    if module_name is None:
        module_name = "_seamless_compiled_" + digest[:24]
    so_bytes = build_extension_cffi(
        module_name,
        binary_objects,
        completed.get("target", "profile"),
        completed["public_header"]["code"],
        completed.get("link_options", []),
    )
    module = _import_extension_module(module_name, so_bytes)
    _MODULE_CACHE[digest] = module
    return module


__all__ = ["build_compiled_module", "complete", "compile"]
