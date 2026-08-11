"""Public Seamless transformer API.

Keep this module light: CLI subcommands import submodules under this package,
and eager public-API imports add noticeable startup time to short commands.
"""

from __future__ import annotations

import threading
from typing import Any


class SeamlessStreamTransformationError(RuntimeError):
    pass


global_lock = threading.Lock()


_LAZY_EXPORTS = {
    "direct": (".transformer_class", "direct"),
    "delayed": (".transformer_class", "delayed"),
    "Transformation": (".transformation_class", "Transformation"),
    "Environment": (".environment", "Environment"),
    "CompiledObject": (".compiled_transformer", "CompiledObject"),
    "CompiledTransformer": (".compiled_transformer", "CompiledTransformer"),
    "DirectCompiledTransformer": (
        ".compiled_transformer",
        "DirectCompiledTransformer",
    ),
    "parallel": (".multi", "parallel"),
    "parallel_async": (".multi", "parallel_async"),
    "TransformationIterableBase": (".multi", "TransformationIterableBase"),
    "TransformationList": (".multi", "TransformationList"),
    "spawn": (".worker", "spawn"),
    "has_spawned": (".worker", "has_spawned"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    from importlib import import_module

    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = [
    "direct",
    "delayed",
    "Transformation",
    "Environment",
    "CompiledObject",
    "CompiledTransformer",
    "DirectCompiledTransformer",
    "parallel",
    "parallel_async",
    "TransformationIterableBase",
    "TransformationList",
    "spawn",
    "has_spawned",
    "global_lock",
    "SeamlessStreamTransformationError",
]
