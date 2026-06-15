"""Remote multi-tenant cancellation against a shared daskserver (Part III.2 Dask).

Dask uses a **first-runner -> latch-on-runner** set. The first-runner is a
*quasi-member*: counted for cancellation and result delivery, but **the cache/set
owns the future**, so softcancelling the first-runner must not release the future
out from under surviving latch-on-runners.

Tenant A is launched first (so it becomes the first-runner); tenant B latches on
once A's run is observably in flight.
"""

import time
import uuid

import pytest

from _harness import SLEEP_SECONDS, Cluster

pytestmark = pytest.mark.remote

TENANT_TIMEOUT = SLEEP_SECONDS + 90


def _nonce(tag: str) -> str:
    return f"{tag}-{uuid.uuid4().hex[:12]}"


def _communicate(proc):
    out, err = proc.communicate(timeout=TENANT_TIMEOUT)
    return out or "", err or ""


def test_dask_dedup_single_execution_across_tenants(dask_cluster):
    """Latch-on dedups across tenants: two tenants => one execution."""
    cl = dask_cluster
    nonce = _nonce("dask-dedup")
    markers = cl.new_marker_dir(nonce)

    a = cl.launch_tenant(action="complete", nonce=nonce, marker_dir=markers)
    assert Cluster.wait_for(markers, "started", timeout=TENANT_TIMEOUT), "first-runner never started"
    b = cl.launch_tenant(action="complete", nonce=nonce, marker_dir=markers)

    out_a, err_a = _communicate(a)
    out_b, err_b = _communicate(b)
    assert a.returncode == 0, (out_a, err_a)
    assert b.returncode == 0, (out_b, err_b)
    assert f"VALUE done-{nonce}" in out_a
    assert f"VALUE done-{nonce}" in out_b
    assert Cluster.count(markers, "started") == 1, "Dask latch-on dedup failed"
    assert Cluster.count(markers, "finished") == 1


def test_dask_latch_on_runner_softcancel_first_runner_survives(dask_cluster):
    """A latch-on-runner (B) losing interest must not free the future; the
    first-runner (A) still completes."""
    cl = dask_cluster
    nonce = _nonce("dask-latchoff")
    markers = cl.new_marker_dir(nonce)

    a = cl.launch_tenant(action="complete", nonce=nonce, marker_dir=markers)
    assert Cluster.wait_for(markers, "started", timeout=TENANT_TIMEOUT), "first-runner never started"
    b = cl.launch_tenant(action="soft_cancel", nonce=nonce, marker_dir=markers)

    out_b, _ = _communicate(b)
    out_a, err_a = _communicate(a)
    assert "CANCELLED" in out_b
    assert a.returncode == 0, (out_a, err_a)
    assert f"VALUE done-{nonce}" in out_a, "first-runner wrongly killed by a latcher's softcancel"
    assert Cluster.count(markers, "finished") == 1


def test_dask_first_runner_softcancel_latcher_survives(dask_cluster):
    """The subtle quasi-member case: the FIRST-RUNNER (A) softcancels, but a
    surviving latch-on-runner (B) must still receive the result — the cache/set
    owns the future, so it is not released when the first-runner leaves."""
    cl = dask_cluster
    nonce = _nonce("dask-firstoff")
    markers = cl.new_marker_dir(nonce)

    a = cl.launch_tenant(action="soft_cancel", nonce=nonce, marker_dir=markers)
    assert Cluster.wait_for(markers, "started", timeout=TENANT_TIMEOUT), "first-runner never started"
    b = cl.launch_tenant(action="complete", nonce=nonce, marker_dir=markers)

    out_a, _ = _communicate(a)
    out_b, err_b = _communicate(b)
    assert "CANCELLED" in out_a
    assert b.returncode == 0, (out_b, err_b)
    assert f"VALUE done-{nonce}" in out_b, "latcher lost the future when the first-runner left"
    assert Cluster.count(markers, "finished") == 1


def test_dask_hard_cancel_is_cross_tenant(dask_cluster):
    """Hard cancel kills the shared Dask submission for all latchers."""
    cl = dask_cluster
    nonce = _nonce("dask-hard")
    markers = cl.new_marker_dir(nonce)

    a = cl.launch_tenant(action="hard_cancel", nonce=nonce, marker_dir=markers)
    assert Cluster.wait_for(markers, "started", timeout=TENANT_TIMEOUT), "run never started"
    b = cl.launch_tenant(action="complete", nonce=nonce, marker_dir=markers)

    out_a, _ = _communicate(a)
    out_b, err_b = _communicate(b)
    assert "CANCELLED True" in out_a, out_a

    time.sleep(SLEEP_SECONDS)
    assert Cluster.count(markers, "finished") == 0, "hard cancel did not reach the shared Dask run"
    assert f"VALUE done-{nonce}" not in out_b, (out_b, err_b)

    replay_markers = cl.new_marker_dir(nonce + "-replay")
    duration = cl.replay_is_uncached(nonce=nonce, marker_dir=replay_markers)
    assert duration >= SLEEP_SECONDS - 1, ("replay was a stale cache hit", duration)
