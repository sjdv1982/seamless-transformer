"""Process-lock runtime behavior in isolated local backends."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def _isolated(function, *args, timeout=150):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent) + os.pathsep + env.get("PYTHONPATH", "")
    script = f"import test_contract_process_lock_runtime as m\nm.{function}{args!r}\nprint('LOCK_OK')"
    completed = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=timeout)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "LOCK_OK" in completed.stdout


def _maximum_overlap(intervals):
    events = [(start, 1) for _, start, end in intervals] + [(end, -1) for _, start, end in intervals]
    count = maximum = 0
    for _, delta in sorted(events):
        count += delta
        maximum = max(count, maximum)
    return maximum


def _sleeping_body(label: str):
    import os
    import time
    start = time.monotonic()
    time.sleep(0.15)
    return os.getpid(), start, time.monotonic()


def _concurrency(mode, language, claim):
    import seamless
    from seamless.transformer import delayed, Transformer, spawn
    import uuid
    if mode == "spawn":
        spawn(2)
    try:
        if language == "python":
            builder = delayed(_sleeping_body)
        elif language == "bash":
            builder = Transformer("bash")
            builder.celltypes.result = "plain"
            builder.celltypes.label = "text"
            builder.code = '''export owner=$PPID
python -c 'import json,os,time; start=time.monotonic(); time.sleep(.15); print(json.dumps([int(os.environ["owner"]),start,time.monotonic()]))' > RESULT'''
        else:
            builder = Transformer("c", compiled=True)
            builder.schema = '''inputs:
  - {name: label, dtype: int32}
outputs:
  - name: result
    dtype:
      fields:
        - {name: pid, dtype: int32}
        - {name: start, dtype: float64}
        - {name: end, dtype: float64}
'''
            builder.code = builder.header + '''
#include <unistd.h>
#include <time.h>
static double stamp(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int transform(int32_t label, ResultStruct *result) {
    result->pid = getpid(); result->start = stamp(); usleep(150000); result->end = stamp(); return 0;
}
'''
        builder.process_lock = claim
        nonce = uuid.uuid4().hex
        handles = [builder(label=i if language == "compiled" else nonce + str(i)).start() for i in range(10)]
        results = [handle.run() for handle in handles]
        if language == "compiled":
            intervals = [(int(r["pid"]), float(r["start"]), float(r["end"])) for r in results]
        else:
            intervals = [tuple(r) for r in results]
        pids = {row[0] for row in intervals}
        assert len(pids) == (2 if mode == "spawn" else 1), intervals
        maxima = [_maximum_overlap([row for row in intervals if row[0] == pid]) for pid in pids]
        if claim:
            assert maxima == [1] * len(pids), (maxima, intervals)
            assert _maximum_overlap(intervals) <= (2 if mode == "spawn" else 1), intervals
        else:
            assert maxima == [3] * len(pids), (maxima, intervals)
        assert all(handle.meta["process_lock"] is claim for handle in handles)
    finally:
        seamless.close()


@pytest.mark.parametrize("mode", ["process", "spawn"])
@pytest.mark.parametrize("language", ["python", "bash", "compiled"])
def test_default_body_exclusivity(mode, language):
    if language == "compiled" and not shutil.which("gcc"):
        pytest.skip("compiled lock coverage requires gcc")
    _isolated("_concurrency", mode, language, True)


@pytest.mark.parametrize("language", ["python", "bash"])
def test_optout_restores_three_active_bodies_per_worker(language):
    _isolated("_concurrency", "spawn", language, False)


def _nested_outer(label: str, route: str):
    import asyncio
    import os
    import time
    from seamless.transformer import delayed, direct
    from seamless_transformer.process_lock import current_holder

    own_start = time.monotonic()
    holder = current_holder()
    assert holder is not None and holder.held
    time.sleep(.03)
    own_end = time.monotonic()

    def leaf(label: str):
        import os
        import time
        start = time.monotonic()
        time.sleep(.03)
        return os.getpid(), start, time.monotonic()

    builder = delayed(leaf)
    left, right = builder(label + "left"), builder(label + "right")
    if route == "run":
        leaves = [left.run(), right.run()]
    elif route == "start":
        left.start()
        right.start()
        leaves = [left.run(), right.run()]
    elif route == "direct":
        inline = direct(leaf)
        leaves = [inline(label + "left"), inline(label + "right")]
    elif route == "async":
        async def launch():
            return await asyncio.gather(left.task(), right.task())
        leaves = asyncio.run(launch())
    elif route == "poll":
        left.start()
        right.start()
        deadline = time.monotonic() + 20
        while any(handle.status not in ("Status: OK", "Status: exception", "Status: unknown exception") for handle in (left, right)):
            if time.monotonic() > deadline:
                raise AssertionError((left.status, right.status))
            time.sleep(.002)
        leaves = [left.run(), right.run()]
    else:
        raise AssertionError(route)
    assert holder.held is False
    # Holding again after a nested result would serialize this unrelated scope.
    with __import__("seamless_transformer.process_lock", fromlist=["scope"]).scope(True):
        pass
    return (os.getpid(), own_start, own_end), leaves


def _nesting(mode, count, route):
    import seamless
    import uuid
    from seamless.transformer import delayed, spawn
    if mode == "spawn":
        spawn(2)
    try:
        builder = delayed(_nested_outer)
        handles = [builder(uuid.uuid4().hex, route).start() for _ in range(count)]
        results = [handle.run() for handle in handles]
        own = [tuple(row[0]) for row in results]
        leaves = [tuple(leaf) for row in results for leaf in row[1]]
        for pid in {row[0] for row in own + leaves}:
            # Both own work before submission and nested leaves must remain exclusive.
            assert _maximum_overlap([row for row in own + leaves if row[0] == pid]) == 1, (own, leaves)
        assert len(leaves) == count * 2
    finally:
        seamless.close()


@pytest.mark.parametrize("mode,count", [("process", 3), ("spawn", 8), ("spawn", 12)])
@pytest.mark.parametrize("route", ["run", "start", "direct", "async", "poll"])
def test_undeclared_nesting_over_capacity_drops_lock(mode, count, route):
    _isolated("_nesting", mode, count, route)


def _blocking_body(ready: str, release: str):
    import os
    import time
    from pathlib import Path
    start = time.monotonic()
    Path(ready).touch()
    deadline = time.monotonic() + 20
    while not Path(release).exists():
        assert time.monotonic() < deadline, "holder was never released"
        time.sleep(.005)
    return os.getpid(), start, time.monotonic()


def _signal_body(release: str):
    import os
    import time
    from pathlib import Path
    start = time.monotonic()
    Path(release).touch()
    return os.getpid(), start, time.monotonic()


def _wait_path(path, handle):
    import time
    deadline = time.monotonic() + 10
    while not path.exists():
        assert time.monotonic() < deadline, (str(path), handle.status, handle.exception)
        time.sleep(.005)


def _driver_while_busy():
    import seamless
    import tempfile
    from pathlib import Path
    from seamless.transformer import delayed
    try:
        with tempfile.TemporaryDirectory() as directory:
            ready, release = Path(directory) / "ready", Path(directory) / "release"
            holder = delayed(_blocking_body)(str(ready), str(release)).start()
            _wait_path(ready, holder)
            driver = delayed(_signal_body)
            driver.driver = True
            driver_result = driver(str(release)).run()
            holder_result = holder.run()
            assert driver_result[0] == holder_result[0]
            assert driver_result[1] < holder_result[2]
    finally:
        seamless.close()


def test_declared_driver_starts_while_process_lock_is_busy():
    _isolated("_driver_while_busy")


def _cancel_waiter():
    import seamless
    import tempfile
    import threading
    import time
    from pathlib import Path
    from contextlib import contextmanager
    from seamless.transformer import delayed
    import importlib
    run_module = importlib.import_module("seamless_transformer.run")
    entered = threading.Event()
    calls = []
    original_scope = run_module.process_lock_scope

    @contextmanager
    def observed_scope(*args, **kwargs):
        calls.append(threading.get_ident())
        if len(calls) == 2:
            entered.set()
        with original_scope(*args, **kwargs) as holder:
            yield holder

    run_module.process_lock_scope = observed_scope
    try:
        with tempfile.TemporaryDirectory() as directory:
            ready, release, side_effect = [Path(directory) / name for name in ("ready", "release", "side-effect")]
            holder = delayed(_blocking_body)(str(ready), str(release)).start()
            _wait_path(ready, holder)
            waiter = delayed(_signal_body)(str(side_effect)).start()
            assert entered.wait(10), "waiter did not enter lock scope"
            assert waiter.cancel()
            release.touch()
            holder.run()
            # A further default body must execute after the cancellation; canceled waiter cannot escape later.
            delayed(_signal_body)(str(Path(directory) / "after")).run()
            time.sleep(.1)
            assert not side_effect.exists(), "canceled waiter eventually ran its body"
    finally:
        run_module.process_lock_scope = original_scope
        seamless.close()


def test_cancelled_inprocess_lock_waiter_never_runs_body():
    _isolated("_cancel_waiter")


def _scratch_producer(nonce: str):
    return "recover this scratch value: " + nonce


def _holding_fingertip(checksum_hex: str, in_loop: bool):
    import asyncio
    from seamless import Checksum
    from seamless_transformer.process_lock import current_holder
    assert current_holder() is not None
    checksum = Checksum(checksum_hex)
    if in_loop:
        async def recover():
            # Synchronous fingertip inside an active loop takes the worker-thread bridge.
            return checksum.fingertip_sync("mixed")
        return asyncio.run(recover())
    return checksum.fingertip_sync("mixed")


def _fingertip(in_loop):
    import seamless
    import uuid
    from seamless.transformer import delayed
    from seamless import CacheMissError
    from seamless.caching.buffer_cache import get_buffer_cache
    try:
        producer = delayed(_scratch_producer)
        producer.scratch = True
        handle = producer(uuid.uuid4().hex)
        checksum = handle.compute()
        assert checksum is not None, handle.exception
        checksum.tempref()
        get_buffer_cache().purge_scratch(checksum)
        try:
            checksum.resolve()
        except CacheMissError:
            pass
        else:
            raise AssertionError("scratch buffer must be missing before holding-body fingertip")
        result = delayed(_holding_fingertip)(checksum.hex(), in_loop).run()
        assert result.startswith("recover this scratch value:")
    finally:
        seamless.close()


@pytest.mark.parametrize("in_loop", [False, True])
def test_fingertip_recompute_drops_holding_body_lock(in_loop):
    _isolated("_fingertip", in_loop)


def _compiled_optout_next_to_holder(mode):
    import json
    import seamless
    import tempfile
    from pathlib import Path
    from seamless.transformer import Transformer, delayed, spawn
    if mode == "spawn":
        spawn(1)
    try:
        with tempfile.TemporaryDirectory() as directory:
            ready, release = Path(directory) / "ready", Path(directory) / "release"
            compiled = Transformer("c", compiled=True)
            compiled.schema = '''inputs:
  - {name: label, dtype: int32}
outputs:
  - {name: result, dtype: int32}
'''
            compiled.code = compiled.header + '''
#include <unistd.h>
#include <stdio.h>
int transform(int32_t label, int32_t *result) {
    FILE *f = fopen(''' + json.dumps(str(release)) + ''', "w");
    if (!f) return 1; fclose(f); *result = getpid(); return 0;
}
'''
            compiled.process_lock = False
            # Warm compilation before the body owns process-global state.
            compiled(label=0).run()
            release.unlink()
            holder = delayed(_blocking_body)(str(ready), str(release)).start()
            _wait_path(ready, holder)
            compiled_pid = compiled(label=1).run()
            holder_result = holder.run()
            assert compiled_pid == holder_result[0]
    finally:
        seamless.close()


@pytest.mark.parametrize("mode", ["process", "spawn"])
def test_compiled_optout_can_run_alongside_python_holder(mode):
    if not shutil.which("gcc"):
        pytest.skip("compiled opt-out coverage requires gcc")
    _isolated("_compiled_optout_next_to_holder", mode)


def _cached_outer(label: str):
    from seamless.transformer import delayed
    from seamless_transformer.process_lock import current_holder
    from test_contract_process_lock_runtime import _sleeping_body
    holder = current_holder()
    assert holder is not None and holder.held
    result = delayed(_sleeping_body)(label).run()
    assert holder.held is False, "a cached nested submission must still drop its parent's lock"
    return result


def _cached_submission():
    import seamless
    import uuid
    from seamless.transformer import delayed
    try:
        label = uuid.uuid4().hex
        original = delayed(_sleeping_body)(label).run()
        assert delayed(_cached_outer)(label).run() == original
    finally:
        seamless.close()


def test_cached_nested_submission_also_drops_holder():
    _isolated("_cached_submission")
