"""Transformation cache helpers."""

from typing import Any, Dict

import asyncio

from seamless import Checksum

from .run import run_transformation_dict_forked, run_transformation_dict_in_process

# In-process cache of transformation results
_transformation_cache: dict[Checksum, Checksum] = {}


async def run(
    transformation_dict: Dict[str, Any],
    *,
    tf_checksum,
    tf_dunder,
    scratch: bool,
    in_process: bool,
) -> Checksum:
    """Execute a transformation, with a shared checksum-to-result cache."""

    tf_checksum = Checksum(tf_checksum)
    cached_result = _transformation_cache.get(tf_checksum)
    if cached_result is not None:
        cached_result.tempref()
        return cached_result

    if in_process:
        result_checksum = run_transformation_dict_in_process(
            transformation_dict, tf_checksum, tf_dunder, scratch
        )
    else:
        result_checksum = await run_transformation_dict_forked(
            transformation_dict, tf_checksum, tf_dunder, scratch
        )
    result_checksum = Checksum(result_checksum)
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
) -> Checksum:
    """Synchronous wrapper around :func:`run`."""

    tf_checksum = Checksum(tf_checksum)
    cached_result = _transformation_cache.get(tf_checksum)
    if cached_result is not None:
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
            )
        )
    finally:
        asyncio.set_event_loop(None)
        new_loop.close()


__all__ = ["run", "run_sync"]
