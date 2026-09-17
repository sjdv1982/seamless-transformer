"""In-process atomicity / races (Part III.4 constraint 3).

"remove last member -> empty -> cancel" races "a new submitter wants to latch."
A lost race merely re-submits a fresh run and is never corrupting — results always
come from the content-addressed cache, never from the membership set.
"""

import asyncio
import threading

import pytest

from seamless_transformer import transformation_cache

from _harness import cs

TD = {"code": "return 1"}
LOCAL = {"__meta__": {"local": True}}


def _submit(cache, tf_checksum):
    return asyncio.create_task(
        cache.run(
            TD, tf_checksum=tf_checksum, tf_dunder=LOCAL,
            scratch=False, require_value=False, force_local=True,
        )
    )


def test_softcancel_storm_keeps_one_holder_alive(inproc_cache, monkeypatch):
    """Many awaiters on one checksum, cancel all but the first while it runs: the
    run is never dropped (a holder always remains) and executes exactly once; every
    survivor gets the consistent result."""
    cache = inproc_cache
    started = threading.Event()
    release = threading.Event()
    calls = []

    def fake_run(*_args, **_kwargs):
        calls.append(1)
        started.set()
        assert release.wait(15)
        return cs("b")

    monkeypatch.setattr(transformation_cache, "run_transformation_dict", fake_run)

    async def main():
        tfc = cs("a")
        keep = _submit(cache, tfc)
        await asyncio.to_thread(started.wait, 5)
        transient = [_submit(cache, tfc) for _ in range(8)]
        await asyncio.sleep(0)

        for t in transient:
            t.cancel()
        for t in transient:
            with pytest.raises(asyncio.CancelledError):
                await t

        # The protected holder kept the run alive throughout.
        assert cache.transformation_status(tfc) == "running"
        release.set()
        assert await keep == cs("b")

    asyncio.run(main())
    assert calls == [1]  # exactly one execution despite the churn


def test_concurrent_submit_and_softcancel_no_corruption(inproc_cache, monkeypatch):
    """Interleave fresh submits and cancellations of the *last* holder so the set
    repeatedly empties and refills. No corruption: each completed run returns the
    canonical result and the execution count stays bounded."""
    cache = inproc_cache
    lock = threading.Lock()
    gates = []  # one (started, release) pair per execution

    def fake_run(*_args, **_kwargs):
        started = threading.Event()
        release = threading.Event()
        with lock:
            gates.append((started, release))
        started.set()
        assert release.wait(15)
        return cs("2")

    monkeypatch.setattr(transformation_cache, "run_transformation_dict", fake_run)

    async def main():
        tfc = cs("1")
        # Round 1: submit, wait running, cancel sole holder -> set empties.
        t1 = _submit(cache, tfc)
        while cache.transformation_status(tfc) != "running":
            await asyncio.sleep(0.01)
        t1.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t1
        while cache.transformation_status(tfc) != "not-running":
            await asyncio.sleep(0.01)

        # Round 2: resubmit two holders; cancel one; release; the other completes.
        a = _submit(cache, tfc)
        while cache.transformation_status(tfc) != "running":
            await asyncio.sleep(0.01)
        b = _submit(cache, tfc)
        await asyncio.sleep(0)
        a.cancel()
        with pytest.raises(asyncio.CancelledError):
            await a
        assert cache.transformation_status(tfc) == "running"
        for _started, release in gates:
            release.set()
        assert await b == cs("2")

    asyncio.run(main())
    # Two execution rounds at most one new run each: never an unbounded storm.
    assert 1 <= len(gates) <= 2
