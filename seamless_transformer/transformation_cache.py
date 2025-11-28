"""Transformation cache helpers."""

from typing import Any, Dict

import asyncio
import os

from seamless import CacheMissError, Checksum, is_worker

from .run import run_transformation_dict_in_process
from . import worker
from seamless_config.select import get_execution

try:
    from seamless.caching import buffer_writer as _buffer_writer
except ImportError:  # pragma: no cover - optional dependency
    _buffer_writer = None

try:
    from seamless_remote import database_remote
except ImportError:
    database_remote = None

try:
    from seamless_remote import jobserver_remote
except ImportError:
    jobserver_remote = None

try:
    from seamless_remote.client import close_all_clients as _close_all_clients
except ImportError:  # pragma: no cover - optional dependency
    _close_all_clients = None

# In-process cache of transformation results
_DEBUG = os.environ.get("SEAMLESS_DEBUG_TRANSFORMATION", "").lower() in (
    "1",
    "true",
    "yes",
)


def _debug(msg: str) -> None:
    if _DEBUG:
        print(f"[transformation_cache] {msg}", flush=True)


async def _await_buffer_writer(checksum: Checksum) -> None:
    if _buffer_writer is None:
        return
    try:
        await _buffer_writer.await_existing_task(checksum)
    except Exception:
        pass


class TransformationCache:
    """Singleton wrapper for transformation cache helpers."""

    def __init__(self) -> None:
        self._transformation_cache: dict[Checksum, Checksum] = {}

    async def run(
        self,
        transformation_dict: Dict[str, Any],
        *,
        tf_checksum,
        tf_dunder,
        scratch: bool,
        require_fingertip: bool,
    ) -> Checksum:
        assert (not scratch) or (not require_fingertip)
        tf_checksum = Checksum(tf_checksum)
        cached_result = self._transformation_cache.get(tf_checksum)
        if cached_result is not None:
            _debug(f"cache hit {tf_checksum.hex()}")
            if not scratch:
                cached_result.tempref()
            return cached_result

        if database_remote is not None and not is_worker():
            _debug(f"query remote db for {tf_checksum.hex()}")
            remote_result = await database_remote.get_transformation_result(tf_checksum)
            _debug(f"remote db result {remote_result}")
            if remote_result is not None:
                if require_fingertip:
                    try:
                        _debug("waiting for fingertip resolution")
                        # TODO: be more lazy and only evaluate the *potential* checksum resolution
                        await remote_result.resolution()
                    except CacheMissError:
                        _debug("fingertip resolution cache miss")
                        remote_result = None
                if remote_result is not None:
                    _debug("using remote result")
                    if not scratch:
                        remote_result.tempref()
                    return remote_result

        execution = get_execution()
        if execution == "remote":
            if jobserver_remote is None:
                raise RuntimeError(
                    "Remote execution requested but seamless_remote is not installed"
                )
            _debug("dispatching transformation to remote jobserver")

            ### NOTE: flushing the entire buffer_writer queue, just to be sure that
            ###   the jobserver has it available.
            ### TODO: flush only the buffers that are required by the transformation
            ### This is not trivial in case of deep checksums
            from seamless.caching import buffer_writer

            buffer_writer.flush()
            ### /NOTE

            result_checksum = await jobserver_remote.run_transformation(
                transformation_dict,
                tf_checksum=tf_checksum,
                tf_dunder=tf_dunder,
                scratch=scratch,
            )
            if isinstance(result_checksum, str):
                raise RuntimeError(result_checksum)
            result_checksum = Checksum(result_checksum)
        elif worker.has_spawned() and not is_worker():
            _debug("dispatching transformation to worker pool")
            result_checksum = await worker.dispatch_to_workers(
                transformation_dict,
                tf_checksum=tf_checksum,
                tf_dunder=tf_dunder,
                scratch=scratch,
            )
            if isinstance(result_checksum, str):
                raise RuntimeError(result_checksum)
            result_checksum = Checksum(result_checksum)
        elif is_worker():
            assert not worker.has_spawned()
            _debug("forwarding transformation request to parent")
            result_checksum = await worker.forward_to_parent(
                transformation_dict,
                tf_checksum=tf_checksum,
                tf_dunder=tf_dunder,
                scratch=scratch,
            )
            if isinstance(result_checksum, str):
                raise RuntimeError(result_checksum)
            result_checksum = Checksum(result_checksum)
        else:
            _debug("running transformation in-process")
            result_checksum = run_transformation_dict_in_process(
                transformation_dict, tf_checksum, tf_dunder, scratch
            )
            result_checksum = Checksum(result_checksum)

        if require_fingertip:
            try:
                _debug("ensuring fingertip value for result")
                await result_checksum.resolution()
            except Exception:
                _debug("fingertip resolution failed; will continue")

        if database_remote is not None and not is_worker():
            await database_remote.set_transformation_result(
                tf_checksum, result_checksum
            )

        if not scratch:
            result_checksum.tempref()
        self._transformation_cache[tf_checksum] = result_checksum
        await _await_buffer_writer(result_checksum)

        if database_remote is not None and not is_worker() and not scratch:
            try:
                buf = result_checksum.resolve()
            except Exception:
                buf = None
            if buf is not None:
                try:
                    await buf.write()
                except Exception:
                    pass
        return result_checksum

    def run_sync(
        self,
        transformation_dict: Dict[str, Any],
        *,
        tf_checksum,
        tf_dunder,
        scratch: bool,
        require_fingertip: bool,
    ) -> Checksum:
        tf_checksum = Checksum(tf_checksum)
        cached_result = self._transformation_cache.get(tf_checksum)
        if cached_result is not None:
            if not scratch:
                cached_result.tempref()
            return cached_result

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                self.run(
                    transformation_dict,
                    tf_checksum=tf_checksum,
                    tf_dunder=tf_dunder,
                    scratch=scratch,
                    require_fingertip=require_fingertip,
                ),
                loop,
            )
            return future.result()

        new_loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(
                self.run(
                    transformation_dict,
                    tf_checksum=tf_checksum,
                    tf_dunder=tf_dunder,
                    scratch=scratch,
                    require_fingertip=require_fingertip,
                )
            )
        finally:
            if _close_all_clients is not None:
                try:
                    _close_all_clients()
                except Exception:
                    pass
            asyncio.set_event_loop(None)
            try:
                new_loop.run_until_complete(new_loop.shutdown_asyncgens())
            except Exception:
                pass
            try:
                new_loop.run_until_complete(new_loop.shutdown_default_executor())
            except Exception:
                pass
            new_loop.close()


_transformation_cache_instance: TransformationCache | None = None


def get_transformation_cache() -> TransformationCache:
    global _transformation_cache_instance
    if _transformation_cache_instance is None:
        _transformation_cache_instance = TransformationCache()
    return _transformation_cache_instance


async def run(
    transformation_dict: Dict[str, Any],
    *,
    tf_checksum,
    tf_dunder,
    scratch: bool,
    require_fingertip: bool,
) -> Checksum:
    return await get_transformation_cache().run(
        transformation_dict,
        tf_checksum=tf_checksum,
        tf_dunder=tf_dunder,
        scratch=scratch,
        require_fingertip=require_fingertip,
    )


def run_sync(
    transformation_dict: Dict[str, Any],
    *,
    tf_checksum,
    tf_dunder,
    scratch: bool,
    require_fingertip: bool,
) -> Checksum:
    return get_transformation_cache().run_sync(
        transformation_dict,
        tf_checksum=tf_checksum,
        tf_dunder=tf_dunder,
        scratch=scratch,
        require_fingertip=require_fingertip,
    )


__all__ = [
    "TransformationCache",
    "get_transformation_cache",
    "run",
    "run_sync",
]
