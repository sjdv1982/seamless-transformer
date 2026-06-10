"""Cancellation of a spawn-pool transformation reclaims the worker slot.

The cancel does not merely detach the promise and let the computation run to
completion: the worker subprocess executing the checksum is terminated and a fresh
one is respawned in its place, so the slot is freed immediately.
"""

import threading
import time
import uuid

import pytest

from seamless.transformer import delayed, direct, has_spawned, spawn
from seamless_transformer import worker


@pytest.fixture(scope="session")
def spawned_workers():
    if not has_spawned():
        spawn(2)
    yield True


def _worker_pids():
    manager = worker._worker_manager
    return {h.name: (h.process.pid if h.process else None) for h in manager._handles}


def test_cancel_reclaims_worker_slot(spawned_workers):
    manager = worker._worker_manager
    assert manager is not None

    nonce = str(uuid.uuid4())

    @delayed
    def slow(token):
        import os
        import time

        time.sleep(30)
        return os.getpid(), token

    tf = slow(nonce)
    tf.construct()
    checksum = tf.transformation_checksum
    tf_hex = checksum.hex()

    pids_before = _worker_pids()

    outcome = {}

    def _run():
        try:
            outcome["value"] = tf.compute()
        except BaseException as exc:  # noqa: BLE001 - we assert it was interrupted
            outcome["exc"] = exc

    thread = threading.Thread(target=_run, name="slow-compute")
    thread.start()

    # Wait until the job is actually executing on a worker subprocess.
    deadline = time.time() + 15
    while time.time() < deadline:
        with manager._active_dispatch_lock:
            running = bool(manager._active_handles.get(tf_hex))
        if running:
            break
        time.sleep(0.05)
    else:
        thread.join(timeout=1)
        pytest.fail("transformation was never dispatched to a worker")

    # Cancel: this must terminate + respawn the worker running it.
    canceled = worker.cancel_by_checksum(checksum)
    assert canceled is True

    # The blocked compute must unblock promptly (well before the 30s sleep).
    thread.join(timeout=15)
    assert not thread.is_alive(), "compute did not unblock after cancel"
    assert "value" not in outcome, "canceled computation should not return a result"

    # The worker that ran the job must have been respawned: at least one PID changes.
    deadline = time.time() + 10
    while time.time() < deadline:
        pids_after = _worker_pids()
        if all(pids_after.values()) and pids_after != pids_before:
            break
        time.sleep(0.1)
    else:
        pytest.fail("no worker was respawned; slot not reclaimed")

    # The cancellation guard is cleared once nothing is running the checksum.
    with manager._active_dispatch_lock:
        assert tf_hex not in manager._cancelled_checksums

    # The pool is healthy after the respawn: a fresh transformation still runs.
    @direct
    def quick(a, b):
        return a + b

    assert quick(2, 3) == 5
