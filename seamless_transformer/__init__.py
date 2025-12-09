from .transformer_class import transformer
from .transformation_class import Transformation
from .worker import spawn, shutdown_workers, has_spawned
import threading

global_lock = (
    threading.Lock()
)  # a global lock that can be useful to coordinate between in-process transformations

__all__ = [
    "transformer",
    "Transformation",
    "spawn",
    "shutdown_workers",
    "has_spawned",
    "global_lock",
]
