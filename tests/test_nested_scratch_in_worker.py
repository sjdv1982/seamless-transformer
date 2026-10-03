"""A scratch transformation nested inside a spawned worker.

The worker keeps no buffer of a scratch result, so ``.run()`` must fingertip it:
resolution misses, and the worker recomputes the result locally without
publishing it (contracts/internal/checksum-reference-lifecycle.md, §8,
*Fingertipping never publishes*). This needs mark_scratch to be a no-op inside
the worker, and the parent's buffer miss to reach the worker as a
CacheMissError rather than as a generic handler failure.
"""

import uuid

import pytest

from seamless import Buffer, Checksum
from seamless.caching import buffer_writer
from seamless.transformer import delayed, has_spawned, spawn
from seamless_transformer import worker


def _close_worker_manager():
    manager = worker._worker_manager
    if manager is not None:
        manager.close(wait=True)
    worker._worker_manager = None
    worker._set_has_spawned(False)


@pytest.fixture
def temporary_spawned_workers():
    spawned_here = False
    if not has_spawned():
        spawn(1)
        spawned_here = True
    yield
    if spawned_here:
        _close_worker_manager()


@pytest.fixture
def writes(monkeypatch):
    written: list[Checksum] = []
    monkeypatch.setattr(
        buffer_writer, "register", lambda buf: written.append(buf.get_checksum())
    )
    return written


def test_nested_scratch_run_in_worker_fingertips_without_publishing(
    temporary_spawned_workers, writes
):
    def outer(word):
        import os
        from seamless.transformer import delayed

        def inner(word):
            return word + "-inner"

        inner_builder = delayed(inner)
        inner_builder.scratch = True
        return {"pid": os.getpid(), "value": inner_builder(word).run()}

    word = f"nested-{uuid.uuid4().hex}"
    outer_builder = delayed(outer)
    outer_builder.local = True
    transformation = outer_builder(word)
    try:
        result = transformation.run()
    finally:
        transformation._release_refholds()

    import os

    assert result["pid"] != os.getpid(), "precondition: outer ran in a worker"
    assert result["value"] == word + "-inner"
    inner_result = Buffer(word + "-inner", "mixed").get_checksum()
    assert inner_result not in writes, "the scratch inner result was published"


def test_worker_forwards_automatic_irreproducible_observation(temporary_spawned_workers, monkeypatch):
    from seamless_remote import database_remote
    reports = []
    async def report(tf_checksum, result_checksum):
        reports.append((tf_checksum.hex(), result_checksum.hex()))
        return True
    monkeypatch.setattr(database_remote, "report_irreproducible_result", report)
    def outer(tf_hex, recorded_hex, observed_hex):
        import asyncio
        from seamless import Checksum
        from seamless_transformer.transformation_cache import get_transformation_cache
        cache = get_transformation_cache()
        cache._register_transformation_result(Checksum(tf_hex), Checksum(recorded_hex))
        outcome = asyncio.run(cache._record_transformation_result(Checksum(tf_hex), Checksum(observed_hex)))
        return {"outcome": outcome, "recorded": cache._transformation_cache[Checksum(tf_hex)].hex()}
    builder = delayed(outer)
    builder.local = True
    tf_hex, recorded_hex, observed_hex = "a" * 64, "b" * 64, "c" * 64
    tf = builder(tf_hex, recorded_hex, observed_hex)
    try:
        assert tf.run() == {"outcome": "MISMATCH", "recorded": recorded_hex}
        assert reports == [(tf_hex, observed_hex)]
    finally:
        tf._release_refholds()
