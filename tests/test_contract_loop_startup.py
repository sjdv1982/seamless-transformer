"""Background-loop startup and task submission must not lose a wake-up."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


def _startup_interleaving():
    import asyncio
    import importlib
    import threading
    module = importlib.import_module("seamless_transformer.transformation_class")
    loop = asyncio.new_event_loop()
    runner_entered = threading.Event()
    permit_run = threading.Event()
    ensure_returned = threading.Event()
    body_ran = threading.Event()
    failures = []
    readiness = []
    original_run_forever = loop.run_forever

    def gated_run_forever():
        runner_entered.set()
        assert permit_run.wait(5), "test did not release loop startup"
        original_run_forever()

    loop.run_forever = gated_run_forever

    async def body():
        body_ran.set()

    def submit():
        try:
            module._ensure_loop_running(loop)
            readiness.append(loop.is_running())
            ensure_returned.set()
            module._create_task_on_loop(loop, body())
        except BaseException as exc:
            failures.append(repr(exc))
            ensure_returned.set()

    caller = threading.Thread(target=submit, daemon=True)
    caller.start()
    try:
        assert runner_entered.wait(3), "loop thread was not launched"
        # The loop thread exists, but it cannot process anything before this gate.
        # Returning now exposes the stale !is_running scheduling branch.
        assert not ensure_returned.wait(.2), "startup returned before the loop could accept work"
        permit_run.set()
        assert ensure_returned.wait(3), "startup readiness never became visible"
        assert readiness == [True], readiness
        assert body_ran.wait(3), "submitted computation never woke the loop"
        caller.join(3)
        assert not caller.is_alive()
        assert not failures, failures
    finally:
        permit_run.set()
        if loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        runner = module._LOOP_THREADS.get(loop)
        if runner is not None:
            runner.join(3)
        if not loop.is_running():
            loop.close()


def _same_loop_submission():
    import asyncio
    from seamless_transformer.transformation_class import _create_task_on_loop

    async def body():
        return "computed"

    async def caller():
        loop = asyncio.get_running_loop()
        task = _create_task_on_loop(loop, body())
        assert task.get_loop() is loop
        assert await task == "computed"

    asyncio.run(caller())


def _caller_holds_driver_lock():
    import asyncio
    import importlib
    import threading
    module = importlib.import_module("seamless_transformer.transformation_class")
    loop = asyncio.new_event_loop()
    original_driver_lock = module._DRIVER_LOOP_LOCK
    module._DRIVER_LOOP_LOCK = threading.Lock()
    body_ran = threading.Event()
    returned = threading.Event()
    errors = []

    async def body():
        body_ran.set()

    def caller():
        try:
            with module._DRIVER_LOOP_LOCK:
                module._ensure_loop_running(loop)
                module._create_task_on_loop(loop, body())
            returned.set()
        except BaseException as exc:
            errors.append(repr(exc))
            returned.set()

    thread = threading.Thread(target=caller, daemon=True)
    thread.start()
    try:
        assert returned.wait(3), "loop callback deadlocked on caller's driver lock"
        assert body_ran.wait(3), "body did not progress while caller held driver lock"
        assert not errors, errors
        thread.join(3)
    finally:
        module._DRIVER_LOOP_LOCK = original_driver_lock
        if loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        runner = module._LOOP_THREADS.get(loop)
        if runner is not None:
            runner.join(3)
        if not loop.is_running():
            loop.close()


@pytest.mark.parametrize("scenario", ["_startup_interleaving", "_same_loop_submission", "_caller_holds_driver_lock"])
def test_loop_startup_and_submission_are_live(scenario):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent) + os.pathsep + env.get("PYTHONPATH", "")
    script = f"import test_contract_loop_startup as m\nm.{scenario}()\nprint('LOOP_OK')"
    completed = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "LOOP_OK" in completed.stdout
