"""Contract: the Transformation handle verb (contracts/cancellation.md, "The API").

``Transformation.cancel()`` / ``cancel_async()`` means "this handle gives up":
it makes *that* handle terminal and softcancels its participation, leaving the
shared run alive for every other member; ``recursive=True`` cascades the same
soft semantics upstream; and the tf_checksum is never invalidated.

Execution count is observed through marker files written by the transformer
body (one file per execution), so dedup and "no relaunch" are observable.
"""

import asyncio
import os
import time

import pytest

from seamless_transformer import delayed
from seamless_transformer.transformation_cache import get_transformation_cache


@delayed
def slow_marked(a, marker_dir):
    import os
    import time
    import uuid

    open(os.path.join(marker_dir, uuid.uuid4().hex), "w").close()
    time.sleep(1.5)
    return a * 2


slow_marked.local = True


@delayed
def plus_one(x):
    return x + 1


plus_one.local = True


def _executions(marker_dir):
    return len(os.listdir(marker_dir))


async def _set_has_members(n, timeout=10.0):
    cache = get_transformation_cache()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        sizes = [len(a.awaiters) for a in list(cache._active_submissions.values())]
        if sizes and max(sizes) >= n:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"no membership set reached {n} members")


async def _swallow(task):
    try:
        await task
    except BaseException:  # noqa: BLE001 - the cancelled handle's task
        pass


def test_handle_cancel_leaves_shared_run_alive_for_peer_handle(tmp_path):
    """Two handles, one tf_checksum, one execution. Cancelling the handle that
    submitted first is terminal for it only; the peer handle still gets the
    result from the same run, which is not relaunched."""
    marker = str(tmp_path)
    nonce = time.time()

    async def main():
        a = slow_marked(nonce, marker)
        b = slow_marked(nonce, marker)
        ta = asyncio.ensure_future(a.task())
        tb = asyncio.ensure_future(b.task())
        await _set_has_members(2)
        assert await a.cancel_async() is True
        assert a.status == "Status: canceled"
        await tb
        await _swallow(ta)
        return a, b

    a, b = asyncio.run(main())
    assert b.status == "Status: OK"
    assert b.value == nonce * 2
    assert _executions(marker) == 1
    assert a.status == "Status: canceled"


def test_handle_cancel_never_invalidates_the_tf_checksum(tmp_path):
    """After a handle was cancelled, a new handle of the same tf_checksum is a
    new submission that simply works (no cancelled state to clear)."""
    marker = str(tmp_path)
    nonce = time.time()

    async def main():
        a = slow_marked(nonce, marker)
        ta = asyncio.ensure_future(a.task())
        await _set_has_members(1)
        assert await a.cancel_async() is True
        await _swallow(ta)
        a.clear_exception()  # does not revive the handle
        assert a.status == "Status: canceled"
        again = slow_marked(nonce, marker)
        await again.task()
        return a, again

    a, again = asyncio.run(main())
    assert again.status == "Status: OK"
    assert again.value == nonce * 2
    assert again.transformation_checksum is not None
    assert a.status == "Status: canceled"


def test_handle_cancel_returns_false_when_nothing_active(tmp_path):
    marker = str(tmp_path)
    tf = slow_marked(time.time(), marker)
    tf.compute()
    assert tf.status == "Status: OK"
    assert tf.cancel() is False
    assert tf.status == "Status: OK"  # a completed handle is not made terminal


def test_recursive_cancel_is_soft_upstream(tmp_path):
    """recursive=True cascades the *same soft* semantics: the upstream handle is
    made terminal, but a peer handle sharing the upstream tf_checksum still gets
    its result from the one run."""
    marker = str(tmp_path)
    nonce = time.time()

    async def main():
        up = slow_marked(nonce, marker)
        peer = slow_marked(nonce, marker)
        down = plus_one(up)
        tp = asyncio.ensure_future(peer.task())
        td = asyncio.ensure_future(down.task())
        await _set_has_members(2)
        assert await down.cancel_async(recursive=True) is True
        assert down.status == "Status: canceled"
        assert up.status == "Status: canceled"
        await tp
        await _swallow(td)
        return peer

    peer = asyncio.run(main())
    assert peer.status == "Status: OK"
    assert peer.value == nonce * 2
    assert _executions(marker) == 1
