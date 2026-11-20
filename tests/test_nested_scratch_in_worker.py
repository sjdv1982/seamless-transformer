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


def _purged_scratch_payload(nonce):
    """A scratch producer's result, computed in-process and then purged.

    Returns the result checksum and the producer's builder. The builder holds
    the producer's code, which a recomputation needs.
    """
    from seamless import CacheMissError
    from seamless.caching.buffer_cache import get_buffer_cache
    from seamless_transformer.transformation_cache import get_transformation_cache

    def producer(nonce) -> bytes:
        return ("payload-" + nonce).encode()

    producer_builder = delayed(producer)
    producer_builder.scratch = True
    producer_builder.local = True
    handle = producer_builder(nonce)
    produced = handle.compute()
    assert isinstance(produced, Checksum), handle.exception
    handle._release_refholds()
    get_buffer_cache().purge_scratch(produced)
    with pytest.raises(CacheMissError):
        produced.resolve()
    candidates = get_transformation_cache().get_reverse_transformations(produced)
    assert list(candidates), "precondition: the parent knows a candidate"
    return produced, producer_builder


def _spawned_consumer(produced):
    def consumer(value, nonce):
        import os

        return [len(value), os.getpid()]

    consumer_builder = delayed(consumer)
    # Not scratch: run() must read the worker's result, not recompute it here.
    consumer_builder.local = True
    consumer_builder.celltypes.value = "bytes"
    consumer_builder.meta = {"allow_input_fingertip": True}
    return consumer_builder(produced, uuid.uuid4().hex)


def test_parent_recovers_direct_input_before_spawn_dispatch(
    temporary_spawned_workers,
):
    """Without Dask, the parent fingertips an absent direct input for its worker."""
    import os

    # The producer's own input must stay available for the recomputation.
    nonce = Buffer(uuid.uuid4().hex, "mixed")
    produced, _producer_builder = _purged_scratch_payload(nonce.get_checksum())
    transformation = _spawned_consumer(produced)
    try:
        length, pid = transformation.run()
    finally:
        transformation._release_refholds()
    assert pid != os.getpid(), "the consumer ran in a worker"
    assert length == len(produced.resolve().content)


def test_failed_parent_recovery_reports_cache_miss_with_category(
    temporary_spawned_workers,
):
    """A failed recovery before dispatch fails the job like an in-process one."""
    from seamless import CacheMissError, FingertipCategory

    # No buffer of the producer's input survives, so no candidate can recover it.
    produced, _producer_builder = _purged_scratch_payload(uuid.uuid4().hex)
    transformation = _spawned_consumer(produced)
    try:
        assert transformation.compute() is None
        exc = transformation._exception
    finally:
        transformation._release_refholds()
    assert isinstance(exc, CacheMissError), repr(exc)
    assert Checksum(exc.checksum) == produced
    assert exc.fingertip_category == FingertipCategory.MATERIALIZATION
    assert "NotImplementedError" not in transformation.exception


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
