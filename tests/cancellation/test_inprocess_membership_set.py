"""In-process awaiter-set behaviour (Part III.1-III.4).

Deterministic and service-free: a ``FakeRunner`` holds a transformation 'running'
while the test manipulates membership.

softcancel is expressed as **cancelling the awaiting asyncio task** (the participant
voluntarily leaving), exactly as in the design ("softcancel = just cancel my await").
"""

import asyncio

import pytest

from seamless_transformer import transformation_cache
from seamless_transformer.transformation_cache import TransformationCancelledError

from _harness import FakeRunner, cs

TD = {"code": "return 1"}
LOCAL = {"__meta__": {"local": True}}


def _submit(cache, tf_checksum, *, dunder=LOCAL):
    return asyncio.create_task(
        cache.run(
            TD, tf_checksum=tf_checksum, tf_dunder=dunder,
            scratch=False, require_value=False, force_local=True,
        )
    )


async def _running(cache, tf_checksum, *, timeout=2.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if cache.transformation_status(tf_checksum) == "running":
            return True
        await asyncio.sleep(0.02)
    return False


async def _settled_not_running(cache, tf_checksum, *, timeout=2.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if cache.transformation_status(tf_checksum) == "not-running":
            return True
        await asyncio.sleep(0.02)
    return False


def test_dedup_single_execution(inproc_cache, monkeypatch):
    """One membership set per tf_checksum: two awaiters => one execution."""
    cache = inproc_cache
    fake = FakeRunner("b")
    monkeypatch.setattr(transformation_cache, "run_transformation_dict", fake)

    async def main():
        tfc = cs("a")
        t1 = _submit(cache, tfc)
        await asyncio.to_thread(fake.started.wait, 5)
        t2 = _submit(cache, tfc, dunder={"__meta__": {"local": False}})
        await asyncio.sleep(0)
        assert cache.transformation_status(tfc) == "running"
        fake.release.set()
        assert await t1 == cs("b")
        assert await t2 == cs("b")

    asyncio.run(main())
    assert fake.calls == 1  # deduplicated: the work ran exactly once


def test_softcancel_one_member_peer_survives(inproc_cache, monkeypatch):
    """softcancel = deregister; above zero the run continues and the survivor still
    gets the result. No cancellation signal reaches the surviving member."""
    cache = inproc_cache
    fake = FakeRunner("2")
    monkeypatch.setattr(transformation_cache, "run_transformation_dict", fake)

    async def main():
        tfc = cs("1")
        t1 = _submit(cache, tfc)
        await asyncio.to_thread(fake.started.wait, 5)
        t2 = _submit(cache, tfc)
        await asyncio.sleep(0)

        t1.cancel()  # member 1 voluntarily leaves
        with pytest.raises(asyncio.CancelledError):
            await t1
        # The run is still alive for member 2.
        assert cache.transformation_status(tfc) == "running"
        fake.release.set()
        assert await t2 == cs("2")  # survivor undisturbed

    asyncio.run(main())
    assert fake.calls == 1  # no relaunch; the survivor reused the same run


def test_softcancel_middle_of_three_no_signal(inproc_cache, monkeypatch):
    """3 members, cancel one => the other two are undisturbed (pure deregistration,
    not a broadcast)."""
    cache = inproc_cache
    fake = FakeRunner("9")
    monkeypatch.setattr(transformation_cache, "run_transformation_dict", fake)

    async def main():
        tfc = cs("8")
        t1 = _submit(cache, tfc)
        await asyncio.to_thread(fake.started.wait, 5)
        t2 = _submit(cache, tfc)
        t3 = _submit(cache, tfc)
        await asyncio.sleep(0)

        t2.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t2
        assert cache.transformation_status(tfc) == "running"
        fake.release.set()
        assert await t1 == cs("9")
        assert await t3 == cs("9")

    asyncio.run(main())
    assert fake.calls == 1


def test_softcancel_last_member_cancels_underlying(inproc_cache, monkeypatch):
    """softcancel-at-zero: the sole awaiter leaves => the set empties => the
    underlying run is cancelled and its bookkeeping cleared."""
    cache = inproc_cache
    fake = FakeRunner("4")
    monkeypatch.setattr(transformation_cache, "run_transformation_dict", fake)

    async def main():
        tfc = cs("3")
        t1 = _submit(cache, tfc)
        await asyncio.to_thread(fake.started.wait, 5)
        assert cache.transformation_status(tfc) == "running"

        t1.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t1

        # Set empty => underlying cancelled; no lingering active submission.
        assert await _settled_not_running(cache, tfc), cache.transformation_status(tfc)
        assert tfc not in cache._active_submissions
        fake.release.set()  # let the stand-in thread unwind cleanly

    asyncio.run(main())


def test_softcancel_at_zero_then_resubmit_runs_again(inproc_cache, monkeypatch):
    """After the set empties, the next submit is a *fresh* run — proving the
    previous run was really dropped, not silently reused."""
    cache = inproc_cache

    # Two independent generations, each with its own gate.
    import threading

    gen = {"n": 0}
    started = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]

    def fake_run(*_args, **_kwargs):
        i = gen["n"]
        gen["n"] += 1
        started[i].set()
        assert release[i].wait(15)
        return cs("c")

    monkeypatch.setattr(transformation_cache, "run_transformation_dict", fake_run)

    async def main():
        tfc = cs("d")
        t1 = _submit(cache, tfc)
        await asyncio.to_thread(started[0].wait, 5)
        t1.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t1
        assert await _settled_not_running(cache, tfc)

        # Fresh submit => the runner is invoked a *second* time.
        t2 = _submit(cache, tfc)
        await asyncio.to_thread(started[1].wait, 5)
        release[0].set()
        release[1].set()
        assert await t2 == cs("c")

    asyncio.run(main())
    assert gen["n"] == 2


def test_cache_hit_is_never_a_member(inproc_cache):
    """A pure cache hit returns immediately and creates no active submission —
    refcount-neutral, never a member."""
    cache = inproc_cache
    tfc = cs("5")
    result = cs("6")
    cache._register_transformation_result(tfc, result)

    async def main():
        assert await cache.run(
            TD, tf_checksum=tfc, tf_dunder={},
            scratch=False, require_value=False, force_local=True,
        ) == result

    asyncio.run(main())
    assert cache._active_submissions == {}


def test_cancel_noops_on_unknown_and_completed(inproc_cache, monkeypatch):
    """Hard/soft cancel of a forgotten or completed checksum is a no-op."""
    cache = inproc_cache

    # Unknown checksum: nothing to cancel.
    assert cache.cancel_by_checksum(cs("0")) is False

    fake = FakeRunner("7")
    monkeypatch.setattr(transformation_cache, "run_transformation_dict", fake)

    async def main():
        tfc = cs("e")
        t1 = _submit(cache, tfc)
        await asyncio.to_thread(fake.started.wait, 5)
        fake.release.set()
        assert await t1 == cs("7")
        # Completed: cancel is a no-op (run already gone from the set).
        assert cache.cancel_by_checksum(tfc) is False

    asyncio.run(main())
