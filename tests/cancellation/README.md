# Cancellation test suite — design & charter

These tests verify that **cancellation works properly** against the contract in
`seamless/context-internals-design-pass3.md` **Part III — Substrate B: Cancellation**.

They are written **black-box, from the contract** — deliberately *not* derived from the
cancellation implementation. The goal is an independent check that the implementation
satisfies the spec, not a mirror of what the code happens to do. The only things imported
from the implementation are its **public surface** (`TransformationCache.run`,
`.transformation_status`, `.cancel_by_checksum`, the high-level `tf.run()/tf.cancel()`
handle) and the `seamless-run` / cluster tooling — learned from the *existing tests*, not
from the cancellation logic.

## The contract under test (Part III, condensed)

A `tf_checksum` may be shared by many participants — sibling nodes, superseded vs current
runs, and **separate processes/tenants**. So cancellation is a **membership set per dedup
site**, with two operations:

- **`softcancel`** = *remove this member; if the set is now empty, cancel the underlying
  run.* It is **pure deregistration** — **no signal reaches the departing member**, and a
  real kill is delivered **only when the set empties**, **only to the executor**. On a
  completed/forgotten checksum it is a **no-op**. A **pure cache hit is never a member**.
- **`cancel` (hard)** = *kill the underlying run and signal every member.* Its cross-tenant
  reach is **correct**, not a footgun: it means "this run is **wrong** (wrong
  envelope/hardware)," which — the envelope being orthogonal to the `tf_checksum` — is
  wrong for every latcher at once.

Three constraints:

1. **`[LOAD-BEARING]` Soft cascades soft.** An empty in-process set must *deregister from*
   the next layer (jobserver/dask), never hard-kill it; a real kill happens **only at the
   leaf set** co-located with the executor.
2. **`[OPTIONAL]` Liveness.** A leaked member (crashed/disconnected tenant) is **benign**:
   it causes *under*-cancellation, never a peer kill — the job simply runs to completion.
   Connection-scoped membership auto-removes a member on disconnect *for free*.
3. **`[HYGIENE]` Per-set atomicity.** "remove last member → empty → cancel" races "a new
   submitter latches"; a lost race merely re-submits a fresh run and is never corrupting.

In-process & jobserver use an **awaiter set** (members deregister in `finally`); Dask uses a
**first-runner → latch-on-runner set** where the first-runner is a **quasi-member** (counted
for cancel + delivery, but the **cache/set owns the future**, so cancelling it cannot release
the future out from under surviving latchers).

## Mechanism mapping (how a test creates each contract action)

| contract action | in-process | remote tenant (separate process) |
|---|---|---|
| **softcancel** (lose interest) | cancel the awaiting `asyncio` task | `tf.cancel()` |
| **softcancel-at-zero** (last member leaves) | cancel the sole awaiter | last tenant calls `tf.cancel()` |
| **hard cancel** (run is wrong) | `cache.cancel_by_checksum(tfc)` | `get_transformation_cache().cancel_by_checksum(tfc)` |
| **leaked member** (crash) | n/a | `SIGKILL` the tenant mid-run |
| **observe "running"** | `cache.transformation_status(tfc)` | the `started-*` marker file appears |
| **observe "killed"** | status → `not-running`, popped from `_active_submissions` | the `finished-*` marker file **never** appears |
| **observe dedup/exec-count** | `FakeRunner.calls` | count of `started-*` marker files |

**Defeating the cache (false-pass guard).** A content-addressed cache hit means *no
execution happens*, so there is nothing to cancel and a test could pass vacuously. Every test
that needs a *live* run uses a **unique nonce** (in-process: a fresh `tf_checksum`; remote: a
unique input arg → fresh `tf_checksum` → fresh execution). Remote kill-tests additionally
**replay** the transformation afterwards and assert the replay re-executes (takes the full
sleep), proving the cancelled run did not poison the database with a partial/bogus result.

## Test inventory

### A. In-process awaiter set — deterministic, no services (`test_inprocess_membership_set.py`)

Uses a `FakeRunner` that blocks on an `Event`, so a "running" transformation is held still
while membership is manipulated.

| test | contract clause |
|---|---|
| `test_dedup_single_execution` | one membership set per `tf_checksum`; two awaiters ⇒ runner executes **once** |
| `test_softcancel_one_member_peer_survives` | softcancel = deregister; **above zero the run continues**; survivor still gets the result; **no signal to survivors** |
| `test_softcancel_middle_of_three_no_signal` | 3 members, cancel one ⇒ the other two are undisturbed (pure deregistration, no broadcast) |
| `test_softcancel_last_member_cancels_underlying` | **softcancel-at-zero**: sole awaiter leaves ⇒ set empties ⇒ underlying cancelled (`status==not-running`, popped) |
| `test_softcancel_at_zero_then_resubmit_runs_again` | after the set empties, the next submit is a **fresh** run (runner called again) |
| `test_cache_hit_is_never_a_member` | a pure cache hit returns immediately and creates **no** active submission (refcount-neutral) |
| `test_cancel_noops_on_unknown_and_completed` | soft/hard cancel of a forgotten/completed checksum is a **no-op** |

### B. In-process hard cancel & envelope orthogonality (`test_inprocess_hard_cancel.py`)

| test | contract clause |
|---|---|
| `test_hard_cancel_kills_all_members` | hard `cancel` signals **every** member ⇒ all awaiters raise `TransformationCancelledError` |
| `test_hard_cancel_returns_false_when_idle` | hard cancel is idempotent / no-op when nothing runs |
| `test_strict_dunder_rejects_during_active_run` | same `tf_checksum`, **different envelope** under `strict_dunder` is rejected while a run is active (envelope is orthogonal to identity) |
| `test_hard_cancel_enables_strict_resubmission_distinct_generation` | the **escape hatch**: hard cancel lets a wrong-envelope run be killed & resubmitted; the new run is a **distinct generation** (old-generation cancel cannot alias it) |

### C. In-process atomicity / races (`test_inprocess_atomicity.py`)

| test | contract clause |
|---|---|
| `test_concurrent_submit_and_softcancel_no_corruption` | constraint 3: interleaved submit/softcancel never corrupts; survivors get a consistent result; bounded executions |
| `test_softcancel_storm_keeps_one_holder_alive` | with ≥1 holder always present, the run is never dropped and executes exactly once |

### D. Remote multi-tenant — jobserver (`test_remote_multitenant_jobserver.py`)

**The headline section.** Two *independent processes* submit the **same** transformation to a
shared jobserver; the jobserver dedups to one run.

| test | contract clause | result |
|---|---|---|
| `test_jobserver_dedup_single_execution_across_tenants` | the membership set dedups **across processes** ⇒ one server-side execution for two tenants | PASS |
| `test_jobserver_softcancel_peer_survives` ★ | **the fixed footgun**: tenant A `softcancel`s ⇒ tenant B's job is **not** killed; B completes with the correct result, `finished` present, exec count 1 | PASS |
| `test_jobserver_all_softcancel_is_benign` | guaranteed: all tenants softcancel ⇒ no crash, no error forced on a peer | PASS |
| `test_jobserver_both_softcancel_leaf_kill` | constraint 1: both tenants leave ⇒ set empties ⇒ **leaf kill** | **XFAIL** (see Findings) |
| `test_jobserver_hard_cancel_is_cross_tenant` | hard `cancel` from one tenant kills the shared run for **all** (correct for the wrong-envelope case); `finished` absent; replay re-executes | PASS |
| `test_jobserver_crashed_member_is_benign` | constraint 2: tenant A is `SIGKILL`ed mid-run ⇒ tenant B is unaffected and completes (leaked member never kills a peer) | PASS |

### E. Remote multi-tenant — Dask (`test_remote_multitenant_dask.py`)

Dask's **first-runner → latch-on-runner** set; the first-runner is a quasi-member.

| test | contract clause | result |
|---|---|---|
| `test_dask_dedup_single_execution_across_tenants` | latch-on dedups across tenants ⇒ one execution | PASS |
| `test_dask_latch_on_runner_softcancel_first_runner_survives` | a **latch-on-runner** leaving does not free the future; the first-runner completes | PASS |
| `test_dask_first_runner_softcancel_latcher_survives` ★ | the subtle one: the **first-runner** leaving must **not** release the future out from under a surviving latcher (cache/set owns it); the latcher still gets the result | **XFAIL** (see Findings) |
| `test_dask_hard_cancel_is_cross_tenant` | hard cancel kills the shared Dask submission for all latchers | PASS |

★ = the load-bearing "no sibling/peer deletion on softcancel" assertions.

## Findings (run 2026-06-15, seamless1 env)

The in-process substrate and the *guaranteed* cross-process behaviours all verify
(13/13 in-process; jobserver dedup + peer-survival + hard-cancel + crash-benign;
dask dedup + latch-off + hard-cancel). Two **not-yet-realized cross-process soft-cancel**
properties were surfaced and are recorded as `xfail(strict=False)` (they flip to
`xpass` once implemented — they are not silenced):

1. **Jobserver soft-cascade leaf-kill (benign gap).** `tf.cancel()` detaches the local
   awaiter but does **not** send a server-side deregister, so when *all* tenants
   softcancel the jobserver runs the orphaned job to completion. The jobserver log shows
   `Received/Attached/Completed` and **zero** cancel messages. No peer is ever harmed, so
   this is "benign under-cancellation" (constraint 2); but the soft-cascade-to-leaf
   (constraint 1 / §10a) — the jobserver server-side membership set that kills on empty —
   is not in place. Only **hard** `cancel_by_checksum` actually stops a jobserver job.
2. **Dask first-runner departure cancels latchers (sibling-kill).** When the first-runner
   softcancels, the latcher receives `TransformationError("Transformation was canceled")`
   instead of the result — the shared Dask future is released out from under it. This is
   the §10b quasi-member "cache/set owns the future" fix, not realized across separate
   client processes. Unlike (1) this is **not** benign: a peer loses a result it wanted.

Both are exactly the `§10a`/`§10b` "net-new work" the design called out, and corroborate
the pass3 V.8 critique (declining cross-process liveness). They are implementation gaps,
not test defects: the in-process equivalents (`test_softcancel_last_member_cancels_underlying`,
`test_softcancel_one_member_peer_survives`) pass, isolating the gap to the server boundary.

## Running

Per the repo convention (and to avoid cross-file state bleed), run **one pytest process per
file**:

```bash
conda activate seamless1
cd seamless-transformer/tests/cancellation
bash run-tests.sh
```

- **A/B/C** need no services and are fast/deterministic.
- **D/E** start a local cluster (hashserver + database + jobserver/daskserver) under a temp
  `HOME`, and are slow (real ~6 s sleeps + service startup + replays). They clean up services
  in teardown. If a cluster cannot start in this environment they fail loudly rather than
  skip silently — these are the tests the exercise is about.
