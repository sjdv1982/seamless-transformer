"""Contract gaps in the in-process substrate (contracts/cancellation.md).

Service-free. Complements test_inprocess_*.py with rules they do not pin:

- "The API": ``softcancel_by_checksum(tf_checksum, member)`` is pure
  deregistration of *that* member; on a completed or forgotten checksum, or for
  a non-member, it is a no-op returning False.
- Constraint 1 (softness composes across layers): an emptied in-process set
  *leaves* the remote set through the remote soft API and never hard-cancels
  the jobserver or Dask. Here the jobserver/Dask clients are recording fakes, so
  the cascade is observable without a cluster.
- "The API": hard ``cancel_by_checksum`` reaches jobserver and Dask as a hard
  cancel.
- "The pattern": execution is owned by the deduplication site, never by the
  first caller -- including the first caller's *event loop*.
"""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from seamless import Checksum
from seamless_transformer import transformation_cache
from seamless_transformer.transformation_cache import TransformationCancelledError

from _harness import FakeRunner, cs

TD = {"code": "return 1"}
LOCAL = {"__meta__": {"local": True}}


def _submit(cache, tf_checksum, *, force_local=True):
    return asyncio.create_task(
        cache.run(
            TD, tf_checksum=tf_checksum, tf_dunder=LOCAL if force_local else {},
            scratch=False, require_value=False, force_local=force_local,
        )
    )


async def _members(cache, tf_checksum, n, timeout=5.0):
    """Wait until the set for tf_checksum has n members."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        active = cache._active_submissions.get(tf_checksum)
        if active is not None and len(active.awaiters) >= n:
            return active
        await asyncio.sleep(0.01)
    raise AssertionError(f"set never reached {n} members")


# --------------------------------------------------------------------------- #
# softcancel_by_checksum as an API
# --------------------------------------------------------------------------- #
def test_softcancel_by_checksum_noop_cases(inproc_cache, monkeypatch):
    """Unknown checksum, non-member, member=None, completed run: all no-ops that
    return False and never disturb a live run."""
    cache = inproc_cache
    fake = FakeRunner("2")
    monkeypatch.setattr(transformation_cache, "run_transformation_dict", fake)

    async def main():
        assert cache.softcancel_by_checksum(cs("0"), object()) is False
        assert await cache.softcancel_by_checksum_async(cs("0"), object()) is False
        tfc = cs("1")
        t1 = _submit(cache, tfc)
        await asyncio.to_thread(fake.started.wait, 5)
        await _members(cache, tfc, 1)
        # Not a member: nothing leaves, nothing is cancelled.
        assert cache.softcancel_by_checksum(tfc, object()) is False
        # No member given: a caller can only softcancel its *own* participation.
        assert cache.softcancel_by_checksum(tfc) is False
        assert await cache.softcancel_by_checksum_async(tfc, None) is False
        assert cache.transformation_status(tfc) == "running"
        fake.release.set()
        assert await t1 == cs("2")
        # Completed: no-op.
        assert cache.softcancel_by_checksum(tfc, object()) is False

    asyncio.run(main())
    assert fake.calls == 1


def test_softcancel_by_checksum_member_leaves_peer_survives(inproc_cache, monkeypatch):
    """Explicit checksum-addressed softcancel of one member: returns True ("I was
    registered and have now left"), the run continues for the other member, and
    no signal reaches anyone."""
    cache = inproc_cache
    fake = FakeRunner("4")
    monkeypatch.setattr(transformation_cache, "run_transformation_dict", fake)

    async def main():
        tfc = cs("3")
        t1 = _submit(cache, tfc)
        await asyncio.to_thread(fake.started.wait, 5)
        t2 = _submit(cache, tfc)
        active = await _members(cache, tfc, 2)
        member = next(iter(active.awaiters))
        assert cache.softcancel_by_checksum(tfc, member) is True
        assert cache.softcancel_by_checksum(tfc, member) is False  # already left
        assert cache.transformation_status(tfc) == "running"
        assert len(active.awaiters) == 1
        fake.release.set()
        assert await t1 == cs("4")
        assert await t2 == cs("4")

    asyncio.run(main())
    assert fake.calls == 1


# --------------------------------------------------------------------------- #
# Constraint 1: soft cascades soft (recording fakes for jobserver and Dask)
# --------------------------------------------------------------------------- #
class _RecordingJobserver:
    def __init__(self):
        self.run_member_ids = []
        self.soft = []
        self.hard = []
        self.started = asyncio.Event()

    async def run_transformation(
        self, transformation_dict, *, tf_checksum, tf_dunder, scratch,
        strict_dunder=False, member_id=None,
    ):
        self.run_member_ids.append(member_id)
        self.started.set()
        await asyncio.Event().wait()  # runs until cancelled

    async def softcancel_transformation_async(self, tf_checksum, member_id):
        self.soft.append((Checksum(tf_checksum), member_id))
        return True

    def softcancel_transformation(self, tf_checksum, member_id):
        self.soft.append((Checksum(tf_checksum), member_id))
        return True

    async def cancel_transformation_async(self, tf_checksum):
        self.hard.append(Checksum(tf_checksum))
        return True

    def cancel_transformation(self, tf_checksum):
        self.hard.append(Checksum(tf_checksum))
        return True


class _RecordingDask:
    def __init__(self):
        self.soft = []
        self.hard = []

    def softcancel_by_checksum(self, tf_checksum, member_id=None):
        self.soft.append((Checksum(tf_checksum), member_id))
        return True

    def cancel_by_checksum(self, tf_checksum):
        self.hard.append(Checksum(tf_checksum))
        return True


@pytest.fixture
def remote_fakes(monkeypatch):
    jobserver = _RecordingJobserver()
    dask = _RecordingDask()

    async def _no_result(_tf_checksum):
        return None

    async def _promise(_checksum):
        return None

    database = SimpleNamespace(
        has_write_server=lambda: True, get_transformation_result=_no_result
    )
    buffers = SimpleNamespace(has_write_server=lambda: True, promise=_promise)
    monkeypatch.setattr(transformation_cache, "jobserver_remote", jobserver)
    monkeypatch.setattr(transformation_cache, "database_remote", database)
    monkeypatch.setattr(transformation_cache, "buffer_remote", buffers)
    monkeypatch.setattr(transformation_cache, "get_execution", lambda: "remote")
    monkeypatch.setattr(transformation_cache, "is_worker", lambda: False)
    monkeypatch.setattr(transformation_cache.worker, "has_spawned", lambda: False)
    import seamless_dask.transformer_client as tc

    monkeypatch.setattr(tc, "get_seamless_dask_client", lambda: dask)
    return SimpleNamespace(
        cache=transformation_cache.TransformationCache(),
        jobserver=jobserver,
        dask=dask,
    )


def test_emptied_set_leaves_remote_sets_softly(remote_fakes):
    """An emptied in-process set means *leave* the jobserver/Dask set: the
    jobserver receives a softcancel for the member id this process registered
    with, and neither jobserver nor Dask ever receives a hard cancel."""
    f = remote_fakes
    cache = f.cache

    async def main():
        tfc = cs("5")
        t1 = _submit(cache, tfc, force_local=False)
        await asyncio.wait_for(f.jobserver.started.wait(), 5)
        t2 = _submit(cache, tfc, force_local=False)
        await _members(cache, tfc, 2)

        t1.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t1
        # Set not empty: the remote set is not touched at all.
        assert f.jobserver.soft == [] and f.dask.soft == []
        assert cache.transformation_status(tfc) == "running"

        t2.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t2
        for _ in range(100):
            if f.jobserver.soft:
                break
            await asyncio.sleep(0.01)

        assert len(f.jobserver.run_member_ids) == 1  # one remote submission
        member_id = f.jobserver.run_member_ids[0]
        assert member_id is not None
        assert (tfc, member_id) in f.jobserver.soft
        assert f.jobserver.hard == []
        assert f.dask.hard == []

    asyncio.run(main())


def test_hard_cancel_by_checksum_is_hard_at_every_layer(remote_fakes):
    f = remote_fakes
    cache = f.cache

    async def main():
        tfc = cs("6")
        t1 = _submit(cache, tfc, force_local=False)
        await asyncio.wait_for(f.jobserver.started.wait(), 5)
        assert cache.cancel_by_checksum(tfc) is True
        with pytest.raises(TransformationCancelledError):
            await asyncio.wait_for(t1, 5)
        assert tfc in f.jobserver.hard
        assert tfc in f.dask.hard

    asyncio.run(main())


# --------------------------------------------------------------------------- #
# The first caller does not own the execution -- not even through its loop
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(
    strict=False,
    reason=(
        "cancellation.md 'The pattern': execution is owned by the dedup site, "
        "never by the first caller. The cache-owned background task is created "
        "on the first caller's event loop; when that loop ends (asyncio.run "
        "returns, a Context closes) the task is cancelled and every surviving "
        "member receives ExecutionCanceledError."
    ),
)
def test_first_callers_loop_ending_does_not_kill_peer(inproc_cache, monkeypatch):
    cache = inproc_cache
    fake = FakeRunner("8")
    monkeypatch.setattr(transformation_cache, "run_transformation_dict", fake)
    tfc = cs("7")
    peer_registered = threading.Event()
    first_left = threading.Event()
    result = {}

    def first():
        async def main():
            t = _submit(cache, tfc)
            await asyncio.to_thread(peer_registered.wait, 5)
            t.cancel()  # the first caller softcancels; the peer remains
            with pytest.raises(asyncio.CancelledError):
                await t
            result["status_after_first_left"] = cache.transformation_status(tfc)
            first_left.set()

        asyncio.run(main())  # ... and then its loop ends

    def peer():
        async def main():
            await asyncio.to_thread(fake.started.wait, 5)
            t = _submit(cache, tfc)
            await _members(cache, tfc, 2)
            peer_registered.set()
            await asyncio.to_thread(first_left.wait, 5)
            await asyncio.sleep(0.2)  # the first caller's loop has closed
            fake.release.set()
            try:
                result["peer"] = await asyncio.wait_for(t, 5)
            except BaseException as exc:  # noqa: BLE001
                result["peer"] = exc

        asyncio.run(main())

    a = threading.Thread(target=first)
    b = threading.Thread(target=peer)
    a.start()
    b.start()
    a.join(20)
    b.join(20)
    fake.release.set()
    assert result.get("status_after_first_left") == "running"
    peer_result = result.get("peer")
    assert isinstance(peer_result, Checksum), repr(peer_result)
    assert peer_result == cs("8")
    assert fake.calls == 1
