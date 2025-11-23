"""Transformation cache helpers."""

from typing import Any, Dict

import asyncio
import os

from seamless import CacheMissError, Checksum

from .run import run_transformation_dict_in_process

try:
    from seamless.caching import buffer_writer as _buffer_writer
except ImportError:  # pragma: no cover - optional dependency
    _buffer_writer = None

try:
    from seamless_remote import database_remote
except ImportError:
    database_remote = None

try:
    from seamless_remote.client import close_all_clients as _close_all_clients
except ImportError:  # pragma: no cover - optional dependency
    _close_all_clients = None

# In-process cache of transformation results
_transformation_cache: dict[Checksum, Checksum] = {}
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


async def run(
    transformation_dict: Dict[str, Any],
    *,
    tf_checksum,
    tf_dunder,
    scratch: bool,
    require_fingertip: bool,
) -> Checksum:
    """Execute a transformation, with a shared checksum-to-result cache.

    require_fingertip: set this to True if a checksum is not enough,
     i.e. that the resulting buffer must be available
    """

    assert (not scratch) or (not require_fingertip)
    tf_checksum = Checksum(tf_checksum)
    cached_result = _transformation_cache.get(tf_checksum)
    if cached_result is not None:
        _debug(f"cache hit {tf_checksum.hex()}")
        if not scratch:
            cached_result.tempref()
        return cached_result

    if database_remote is not None:
        _debug(f"query remote db for {tf_checksum.hex()}")
        remote_result = await database_remote.get_transformation_result(tf_checksum)
        _debug(f"remote db result {remote_result}")
        if remote_result is not None:
            if require_fingertip:
                try:
                    _debug("waiting for fingertip resolution")
                    await remote_result.resolution()
                except CacheMissError:
                    _debug("fingertip resolution cache miss")
                    remote_result = None
            if remote_result is not None:
                _debug("using remote result")
                if not scratch:
                    remote_result.tempref()
                return remote_result

    _debug("running transformation in-process")
    result_checksum = run_transformation_dict_in_process(
        transformation_dict, tf_checksum, tf_dunder, scratch
    )
    result_checksum = Checksum(result_checksum)

    if database_remote is not None:
        # TODO: this could be done in the background, if we are a main Seamless client
        # Not a good idea if we are a jobslave!
        await database_remote.set_transformation_result(tf_checksum, result_checksum)

    if not scratch:
        result_checksum.tempref()
    _transformation_cache[tf_checksum] = result_checksum
    await _await_buffer_writer(result_checksum)
    return result_checksum


def run_sync(
    transformation_dict: Dict[str, Any],
    *,
    tf_checksum,
    tf_dunder,
    scratch: bool,
    require_fingertip: bool,
) -> Checksum:
    """Synchronous wrapper around :func:`run`."""

    tf_checksum = Checksum(tf_checksum)
    cached_result = _transformation_cache.get(tf_checksum)
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
            run(
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
            run(
                transformation_dict,
                tf_checksum=tf_checksum,
                tf_dunder=tf_dunder,
                scratch=scratch,
                require_fingertip=require_fingertip,
            )
        )
    finally:
        # Close any aiohttp sessions bound to this temporary loop to avoid
        # unclosed-connector warnings on shutdown.
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


__all__ = ["run", "run_sync"]
