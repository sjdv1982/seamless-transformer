from .transformer_class import direct, delayed
from .transformation_class import Transformation
from .multi import parallel, parallel_async, TransformationIterableBase, TransformationList
from .worker import spawn, has_spawned
import threading


class SeamlessStreamTransformationError(RuntimeError):
    pass


global_lock = (
    threading.Lock()
)  # a global lock that can be useful to coordinate between in-process transformations

__all__ = [
    "direct",
    "delayed",
    "Transformation",
    "parallel",
    "parallel_async",
    "TransformationIterableBase",
    "TransformationList",
    "spawn",
    "has_spawned",
    "global_lock",
    "SeamlessStreamTransformationError",
]
