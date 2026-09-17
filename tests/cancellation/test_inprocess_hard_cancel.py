"""In-process hard cancel + envelope orthogonality (Part III.3, III.5).

Hard ``cancel`` means "this run is *wrong* and must be killed and resubmitted." It
signals **every** member (kill-all), which is the deliberate escape hatch for the
``strict_dunder`` envelope-contention check: the same tf_checksum can be running under
the wrong (orthogonal) envelope for every latcher at once, so all of them must die.
"""

import asyncio
import threading

import pytest

from seamless_transformer import transformation_cache
from seamless_transformer.transformation_cache import TransformationCancelledError

from _harness import cs

TD = {"code": "return 1"}
LOCAL = {"__meta__": {"local": True}}


def _submit(cache, tf_checksum, *, dunder=LOCAL, strict=False):
    return asyncio.create_task(
        cache.run(
            TD, tf_checksum=tf_checksum, tf_dunder=dunder,
            scratch=False, require_value=False, force_local=True,
            strict_dunder=strict,
        )
    )


def test_hard_cancel_kills_all_members(inproc_cache, monkeypatch):
    """Hard cancel signals every member: both awaiters raise."""
    cache = inproc_cache
    started = threading.Event()
    release = threading.Event()

    def fake_run(*_args, **_kwargs):
        started.set()
        assert release.wait(15)
        return cs("b")

    monkeypatch.setattr(transformation_cache, "run_transformation_dict", fake_run)

    async def main():
        tfc = cs("a")
        t1 = _submit(cache, tfc)
        await asyncio.to_thread(started.wait, 5)
        t2 = _submit(cache, tfc)
        await asyncio.sleep(0)
        try:
            assert cache.cancel_by_checksum(tfc) is True
            # Every member is cancelled, not just the owner.
            with pytest.raises(TransformationCancelledError):
                await asyncio.wait_for(t1, timeout=2)
            with pytest.raises(TransformationCancelledError):
                await asyncio.wait_for(t2, timeout=2)
            assert cache.transformation_status(tfc) == "not-running"
        finally:
            release.set()

    asyncio.run(main())


def test_hard_cancel_returns_false_when_idle(inproc_cache):
    """Idempotent / no-op when nothing is running."""
    cache = inproc_cache
    assert cache.cancel_by_checksum(cs("f")) is False


def test_strict_dunder_rejects_during_active_run(inproc_cache, monkeypatch):
    """Same identity, *different envelope* under strict_dunder is rejected while a
    run is active — the envelope is orthogonal to the tf_checksum."""
    cache = inproc_cache
    started = threading.Event()
    release = threading.Event()

    def fake_run(*_args, **_kwargs):
        started.set()
        assert release.wait(15)
        return cs("d")

    monkeypatch.setattr(transformation_cache, "run_transformation_dict", fake_run)

    async def main():
        tfc = cs("c")
        task = _submit(cache, tfc, dunder={"__meta__": {"local": True}})
        await asyncio.to_thread(started.wait, 5)
        try:
            with pytest.raises(RuntimeError, match="different dunder envelope"):
                await cache.run(
                    TD, tf_checksum=tfc, tf_dunder={"__meta__": {"local": False}},
                    scratch=False, require_value=False, force_local=True,
                    strict_dunder=True,
                )
        finally:
            release.set()
        assert await task == cs("d")

    asyncio.run(main())


def test_hard_cancel_enables_strict_resubmission_distinct_generation(inproc_cache, monkeypatch):
    """The escape hatch: a hard cancel kills the wrong-envelope run, after which a
    strict resubmission succeeds as a *distinct generation* — the old-generation
    cancel cannot alias the new run."""
    cache = inproc_cache
    started = threading.Event()
    release = threading.Event()
    calls = []

    def fake_run(*_args, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            started.set()
            assert release.wait(15)
            return cs("f")
        return cs("e")

    monkeypatch.setattr(transformation_cache, "run_transformation_dict", fake_run)

    async def main():
        tfc = cs("e")
        task = _submit(cache, tfc, dunder={"__meta__": {"local": True}})
        await asyncio.to_thread(started.wait, 5)

        # Wrong envelope is detected; hard cancel clears it for resubmission.
        assert cache.cancel_by_checksum(tfc) is True
        assert cache.cancel_by_checksum(tfc) is False  # idempotent
        assert cache.transformation_status(tfc) == "not-running"

        # New generation under the corrected envelope.
        assert await cache.run(
            TD, tf_checksum=tfc, tf_dunder={"__meta__": {"local": False}},
            scratch=False, require_value=False, force_local=True, strict_dunder=True,
        ) == cs("e")

        # The original (old generation) is cancelled, not aliased to the new result.
        release.set()
        with pytest.raises(TransformationCancelledError):
            await task

    asyncio.run(main())
    assert len(calls) == 2
