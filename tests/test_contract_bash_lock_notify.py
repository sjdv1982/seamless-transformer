"""Bash nested-work FIFO releases the holder for the script's whole lifetime."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


def _run_isolated(function, *args):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent) + os.pathsep + env.get("PYTHONPATH", "")
    script = f"import test_contract_bash_lock_notify as m\nm.{function}{args!r}\nprint('NOTIFY_OK')"
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=25)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NOTIFY_OK" in result.stdout


def _bash_notifies(close_pipes):
    import importlib
    import shlex
    import tempfile
    import threading
    import time
    module = importlib.import_module("seamless_transformer.execute_bash")
    from seamless_transformer.process_lock import scope, current_holder
    completed = threading.Event()
    failures = []
    with tempfile.TemporaryDirectory() as directory:
        active, release, fifo_path, job_path = [Path(directory) / name for name in ("active", "release", "fifo", "job")]
        def contender():
            deadline = time.monotonic() + 10
            try:
                while not active.exists():
                    assert time.monotonic() < deadline
                    time.sleep(.002)
                with scope(True, should_abort=lambda: time.monotonic() > deadline):
                    assert not completed.is_set(), "the lock was released only after the script ended"
                    release.touch()
            except BaseException as exc:
                failures.append(repr(exc))
                release.touch()

        thread = threading.Thread(target=contender, daemon=True)
        with scope(True):
            holder = current_holder()
            thread.start()
            script = '\n'.join([
                'printf "%s" "$SEAMLESS_PROCESS_LOCK_NOTIFY" > ' + shlex.quote(str(fifo_path)),
                'printf "%s" "$PWD" > ' + shlex.quote(str(job_path)),
                'touch ' + shlex.quote(str(active)),
                'exec 1>&- 2>&-' if close_pipes else ':',
                # Leave time for both output pipes to close before notifying.
                'sleep .05' if close_pipes else ':',
                'printf x > "$SEAMLESS_PROCESS_LOCK_NOTIFY"',
                'while ! test -f ' + shlex.quote(str(release)) + '; do sleep .01; done',
                'printf 42 > RESULT',
            ])
            try:
                assert module.execute_bash(script, [], None, {}, {}, "mixed") == 42
            finally:
                completed.set()
                thread.join(3)
            assert holder.held is False
        assert not failures, failures
        assert not thread.is_alive()
        path = Path(fifo_path.read_text())
        assert not path.exists(), "notification FIFO leaked after execution"
        assert path.parent == Path(job_path.read_text()).parent, "FIFO must live next to the job directory"


@pytest.mark.parametrize("close_pipes", [False, True])
def test_bash_notification_drops_lock_while_script_still_runs(close_pipes):
    _run_isolated("_bash_notifies", close_pipes)


def _bash_masks_notification(claim, drop):
    import importlib
    import tempfile
    from seamless_transformer.process_lock import scope, drop_for_nested_work, NOTIFY_ENV
    module = importlib.import_module("seamless_transformer.execute_bash")
    os.environ[NOTIFY_ENV] = "/inherited/notification/that/must/be/masked"
    with scope(claim):
        if drop:
            drop_for_nested_work()
        # Distinguish absent from present-but-empty in the shell environment.
        assert module.execute_bash('if test "${SEAMLESS_PROCESS_LOCK_NOTIFY+x}" = x; then printf 1; else printf 0; fi > RESULT', [], None, {}, {}, "mixed") == 0


@pytest.mark.parametrize("claim,drop", [(False, False), (True, True)])
def test_bash_masks_inherited_notification_without_a_held_lock(claim, drop):
    _run_isolated("_bash_masks_notification", claim, drop)


def _sender_fifo():
    import tempfile
    import stat
    from seamless_transformer.process_lock import drop_for_nested_work, NOTIFY_ENV
    with tempfile.TemporaryDirectory() as directory:
        fifo = Path(directory) / "notify"
        os.mkfifo(fifo)
        descriptor = os.open(fifo, os.O_RDWR | os.O_NONBLOCK)
        try:
            os.environ[NOTIFY_ENV] = str(fifo)
            drop_for_nested_work()
            assert len(os.read(descriptor, 1024)) == 1
            drop_for_nested_work()
            try:
                extra = os.read(descriptor, 1024)
            except BlockingIOError:
                extra = b""
            assert extra == b"", "sender notified more than once in one process"
            child = os.fork()
            if child == 0:
                try:
                    drop_for_nested_work()
                finally:
                    os._exit(0)
            _, status = os.waitpid(child, 0)
            assert os.waitstatus_to_exitcode(status) == 0
            assert len(os.read(descriptor, 1024)) == 1, "fork child inherited the parent's already-notified mark"
        finally:
            os.close(descriptor)


def test_sender_notifies_once_per_process_without_a_holder():
    _run_isolated("_sender_fifo")


def _sender_ignores_error(kind):
    import tempfile
    from seamless_transformer.process_lock import drop_for_nested_work, NOTIFY_ENV
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "notify"
        if kind == "no-reader":
            os.mkfifo(target)
        elif kind == "directory":
            target.mkdir()
        elif kind == "write-error":
            os.mkfifo(target)
        os.environ[NOTIFY_ENV] = str(target)
        if kind == "write-error":
            from unittest.mock import patch
            descriptor = os.open(target, os.O_RDWR | os.O_NONBLOCK)
            try:
                with patch("os.write", side_effect=OSError("injected write failure")):
                    drop_for_nested_work()
                    drop_for_nested_work()
            finally:
                os.close(descriptor)
        else:
            drop_for_nested_work()
            drop_for_nested_work()


@pytest.mark.parametrize("kind", ["missing", "no-reader", "directory", "write-error"])
def test_sender_ignores_missing_fifo_and_nonblocking_open_errors(kind):
    _run_isolated("_sender_ignores_error", kind)
