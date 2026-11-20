"""Subprocess body for test_fingertip.py::test_fingertip_identity_cycle_terminates.

Kept out of test_fingertip.py, which selects the "fingertip" stage on import:
with that stage's hashserver, X would be fetched instead of recovered.
"""

import asyncio
import uuid

import seamless
from seamless import Buffer, CacheMissError, Checksum
from seamless.caching.buffer_cache import get_buffer_cache
from seamless.checksum.cached_calculate_checksum import checksum_cache
import seamless.checksum.expression as expression_mod
from seamless.transformer import delayed


def _drop_buffer(checksum: Checksum) -> None:
    cache = get_buffer_cache()
    with cache.lock:
        cache.weak_cache.pop(checksum, None)
        cache.strong_cache.pop(checksum, None)
    checksum_cache.pop(checksum, None)
    expression_mod._expression_result_buffers.pop(checksum, None)


def identity_cycle_fingertip() -> None:
    """Recover X whose only recorded producer is T(X) = X, in-process."""

    @delayed
    def identity(value):
        return value

    identity.local = True
    identity.allow_input_fingertip = True
    try:
        source = Buffer("identity cycle " + uuid.uuid4().hex, "mixed")
        checksum = source.get_checksum()
        handle = identity(checksum)
        assert handle.compute() == checksum, handle.exception
        handle._release_refholds()
        del source
        _drop_buffer(checksum)
        try:
            checksum.resolve()
        except CacheMissError:
            pass
        else:
            raise AssertionError("X must be missing before the fingertip")
        # T's input recovery runs in an executor thread. It must see that X is
        # already being recovered, or it re-enters T and never returns.
        try:
            asyncio.run(checksum.fingertip("mixed"))
        except CacheMissError:
            pass
        else:
            raise AssertionError("X has no other producer and cannot be recovered")
    finally:
        seamless.close()
