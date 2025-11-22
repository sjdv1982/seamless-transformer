"""Transformation cache helpers."""

from typing import Any, Dict

import asyncio

from seamless import CacheMissError, Checksum

from .run import run_transformation_dict_forked, run_transformation_dict_in_process

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


async def run(
    transformation_dict: Dict[str, Any],
    *,
    tf_checksum,
    tf_dunder,
    scratch: bool,
    in_process: bool,
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
        if not scratch:
            cached_result.tempref()
        return cached_result

    if database_remote is not None:
        remote_result = await database_remote.get_transformation_result(tf_checksum)
        if remote_result is not None:
            if require_fingertip:
                try:
                    await remote_result.resolution()
                except CacheMissError:
                    remote_result = None
            if remote_result is not None:
                if not scratch:
                    remote_result.tempref()
                return remote_result

    if in_process:
        result_checksum = run_transformation_dict_in_process(
            transformation_dict, tf_checksum, tf_dunder, scratch
        )
    else:
        result_checksum = await run_transformation_dict_forked(
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
    return result_checksum


def run_sync(
    transformation_dict: Dict[str, Any],
    *,
    tf_checksum,
    tf_dunder,
    scratch: bool,
    in_process: bool,
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
                in_process=in_process,
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
                in_process=in_process,
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
        new_loop.close()


__all__ = ["run", "run_sync"]
