"""Remote multi-tenant cancellation against a shared jobserver (Part III.2, III.5,
and constraints 1 & 2).

Two *independent processes* (tenants) submit the **same** transformation; the
jobserver dedups them onto one server-side run. We then make one tenant lose
interest / hard-cancel / crash, and assert the contractual effect on the peer via
filesystem markers written by the (server-side) transformation:

  started-*   : appears once the run begins  (=> execution count)
  finished-*  : appears only if the run completes (absent => it was killed)

Each test uses a unique ``nonce`` so the transformation identity is fresh and the
run actually executes (a cache hit would have nothing to cancel). Kill-tests then
replay and assert the replay re-executes (no poisoned-cache false pass).
"""

import time
import uuid

import pytest

from _harness import SLEEP_SECONDS, Cluster

pytestmark = pytest.mark.remote

# Generous: service round-trips + a real SLEEP_SECONDS run.
TENANT_TIMEOUT = SLEEP_SECONDS + 90


def _nonce(tag: str) -> str:
    return f"{tag}-{uuid.uuid4().hex[:12]}"


def _communicate(proc):
    out, err = proc.communicate(timeout=TENANT_TIMEOUT)
    return out or "", err or ""


def test_jobserver_dedup_single_execution_across_tenants(jobserver_cluster):
    """The membership set dedups across processes: two tenants => one execution."""
    cl = jobserver_cluster
    nonce = _nonce("dedup")
    markers = cl.new_marker_dir(nonce)

    a = cl.launch_tenant(action="complete", nonce=nonce, marker_dir=markers)
    assert Cluster.wait_for(markers, "started", timeout=TENANT_TIMEOUT), "tenant A never started"
    b = cl.launch_tenant(action="complete", nonce=nonce, marker_dir=markers)

    out_a, err_a = _communicate(a)
    out_b, err_b = _communicate(b)
    assert a.returncode == 0, (out_a, err_a)
    assert b.returncode == 0, (out_b, err_b)
    assert f"VALUE done-{nonce}" in out_a
    assert f"VALUE done-{nonce}" in out_b

    assert Cluster.count(markers, "started") == 1, "deduplication failed: ran more than once"
    assert Cluster.count(markers, "finished") == 1


def test_jobserver_softcancel_peer_survives(jobserver_cluster):
    """THE fixed footgun: tenant A softcancels (loses interest); tenant B's shared
    job must NOT be killed. B completes with the correct result."""
    cl = jobserver_cluster
    nonce = _nonce("softpeer")
    markers = cl.new_marker_dir(nonce)

    a = cl.launch_tenant(action="soft_cancel", nonce=nonce, marker_dir=markers)
    assert Cluster.wait_for(markers, "started", timeout=TENANT_TIMEOUT), "run never started"
    b = cl.launch_tenant(action="complete", nonce=nonce, marker_dir=markers)

    out_a, err_a = _communicate(a)
    out_b, err_b = _communicate(b)

    # A really cancelled its own participation...
    assert "CANCELLED True" in out_a, (out_a, err_a)
    # ...but B was untouched and got the result.
    assert b.returncode == 0, (out_b, err_b)
    assert f"VALUE done-{nonce}" in out_b, (out_b, err_b)
    assert Cluster.count(markers, "started") == 1
    assert Cluster.count(markers, "finished") == 1, "the shared job was wrongly killed by A's softcancel"


def test_jobserver_all_softcancel_is_benign(jobserver_cluster):
    """GUARANTEED contract: when every tenant softcancels (loses interest), no peer is
    harmed — there is no crash and no error is delivered to anyone. Whether the now-
    orphaned job is also *killed* is the separate, optional leaf-kill property tested
    (xfail) below."""
    cl = jobserver_cluster
    nonce = _nonce("allsoftbenign")
    markers = cl.new_marker_dir(nonce)

    a = cl.launch_tenant(action="soft_cancel", nonce=nonce, marker_dir=markers)
    assert Cluster.wait_for(markers, "started", timeout=TENANT_TIMEOUT), "run never started"
    b = cl.launch_tenant(action="soft_cancel", nonce=nonce, marker_dir=markers)

    out_a, err_a = _communicate(a)
    out_b, err_b = _communicate(b)
    # Both cleanly cancelled their own participation; neither crashed.
    assert a.returncode == 0, (out_a, err_a)
    assert b.returncode == 0, (out_b, err_b)
    assert "CANCELLED" in out_a and "CANCELLED" in out_b
    # No TransformationError ("...was canceled") was forced on either party: a peer is
    # never killed by another's softcancel.
    assert "was canceled" not in out_a, ("softcancel forced an error on a peer", out_a)
    assert "was canceled" not in out_b, ("softcancel forced an error on a peer", out_b)


@pytest.mark.xfail(
    strict=False,
    reason="OPTIONAL leaf-kill across the jobserver boundary is not yet realized. "
    "Evidence: tf.cancel() detaches the local awaiter but sends no server-side "
    "deregister, so the jobserver runs the orphaned job to completion (jobserver log "
    "shows Received/Attached/Completed and zero cancel messages). This is the "
    "soft-cascade-to-leaf item (design pass3 Part III constraint 1 / §10a). It is "
    "'benign under-cancellation' per constraint 2 (no peer is killed). Remove this "
    "xfail once the jobserver gains a server-side membership set that leaf-kills on "
    "empty.",
)
def test_jobserver_both_softcancel_leaf_kill(jobserver_cluster):
    """Both tenants leave => the membership set empties => leaf kill. The job does
    not finish, and a later replay re-executes (the cancelled run cached nothing)."""
    cl = jobserver_cluster
    nonce = _nonce("leafkill")
    markers = cl.new_marker_dir(nonce)

    a = cl.launch_tenant(action="soft_cancel", nonce=nonce, marker_dir=markers)
    assert Cluster.wait_for(markers, "started", timeout=TENANT_TIMEOUT), "run never started"
    b = cl.launch_tenant(action="soft_cancel", nonce=nonce, marker_dir=markers)

    out_a, _ = _communicate(a)
    out_b, _ = _communicate(b)
    assert "CANCELLED" in out_a
    assert "CANCELLED" in out_b

    # Give any in-flight completion a moment; it must NOT arrive.
    time.sleep(SLEEP_SECONDS)
    assert Cluster.count(markers, "started") >= 1
    assert Cluster.count(markers, "finished") == 0, "leaf kill failed: run completed after all left"

    # False-pass guard: replay must re-execute (cancelled run poisoned nothing).
    replay_markers = cl.new_marker_dir(nonce + "-replay")
    duration = cl.replay_is_uncached(nonce=nonce, marker_dir=replay_markers)
    assert duration >= SLEEP_SECONDS - 1, ("replay was a stale cache hit", duration)


def test_jobserver_hard_cancel_is_cross_tenant(jobserver_cluster):
    """Hard cancel from one tenant kills the shared run for all — correct for the
    wrong-envelope/hardware case. B does not complete; replay re-executes."""
    cl = jobserver_cluster
    nonce = _nonce("hardcross")
    markers = cl.new_marker_dir(nonce)

    a = cl.launch_tenant(action="hard_cancel", nonce=nonce, marker_dir=markers)
    assert Cluster.wait_for(markers, "started", timeout=TENANT_TIMEOUT), "run never started"
    b = cl.launch_tenant(action="complete", nonce=nonce, marker_dir=markers)

    out_a, _ = _communicate(a)
    out_b, err_b = _communicate(b)
    assert "CANCELLED True" in out_a, out_a

    time.sleep(SLEEP_SECONDS)
    # The shared run was killed for everyone: no completion marker, and B did not
    # obtain a real result.
    assert Cluster.count(markers, "finished") == 0, "hard cancel did not reach the shared run"
    assert f"VALUE done-{nonce}" not in out_b, (out_b, err_b)

    replay_markers = cl.new_marker_dir(nonce + "-replay")
    duration = cl.replay_is_uncached(nonce=nonce, marker_dir=replay_markers)
    assert duration >= SLEEP_SECONDS - 1, ("replay was a stale cache hit", duration)


def test_jobserver_crashed_member_is_benign(jobserver_cluster):
    """Constraint 2: a tenant that crashes (SIGKILL) mid-run never kills its peer.
    The shared job runs to completion for the surviving tenant."""
    cl = jobserver_cluster
    nonce = _nonce("crash")
    markers = cl.new_marker_dir(nonce)

    a = cl.launch_tenant(action="crash", nonce=nonce, marker_dir=markers)
    assert Cluster.wait_for(markers, "started", timeout=TENANT_TIMEOUT), "run never started"
    b = cl.launch_tenant(action="complete", nonce=nonce, marker_dir=markers)

    # Kill A hard, mid-run, without any cancel handshake (a leaked member).
    time.sleep(1.5)
    a.kill()
    a.communicate(timeout=15)

    out_b, err_b = _communicate(b)
    assert b.returncode == 0, (out_b, err_b)
    assert f"VALUE done-{nonce}" in out_b, "peer was wrongly affected by a crashed member"
    assert Cluster.count(markers, "finished") == 1
