"""Bash execution preserves the Python process's working directory."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


def _check_nested_bash_cwd(nonce: str):
    # Keep the helper self-contained: it is serialized as a Python body.
    import os
    import tempfile
    import threading
    import time
    from seamless.transformer import Transformer

    original_cwd = os.getcwd()
    fd, sentinel = tempfile.mkstemp(prefix="bash-cwd-sentinel-", dir=original_cwd)
    os.close(fd)
    relative = os.path.basename(sentinel)
    with open(sentinel, "w") as stream:
        stream.write("sentinel")
    stop = threading.Event()
    ready = [threading.Event(), threading.Event()]
    failures = []
    observations = [0, 0]

    def watch_cwd():
        while not stop.is_set():
            try:
                assert os.getcwd() == original_cwd
                observations[0] += 1
            except Exception as exc:
                failures.append(repr(exc))
            ready[0].set()
            time.sleep(0.0005)

    def read_relative():
        while not stop.is_set():
            try:
                with open(relative) as stream:
                    assert stream.read() == "sentinel"
                observations[1] += 1
            except Exception as exc:
                failures.append(repr(exc))
            ready[1].set()
            time.sleep(0.0005)

    threads = [threading.Thread(target=watch_cwd), threading.Thread(target=read_relative)]
    for thread in threads:
        thread.start()
    try:
        assert all(event.wait(5) for event in ready)
        bash = Transformer("bash", direct=True)
        bash.celltypes.result = "int"
        bash.local = True
        for index in range(4):
            # Distinct code prevents a cache hit from bypassing the regression.
            bash.code = 'sleep 0.08; printf "%s" "$PPID" > RESULT # ' + str(index) + nonce
            assert bash() == os.getpid(), "nested bash must execute in the watched process"
    finally:
        stop.set()
        for thread in threads:
            thread.join(5)
        os.unlink(sentinel)
    assert all(not thread.is_alive() for thread in threads)
    assert all(count > 10 for count in observations), observations
    assert not failures, failures[:10]
    assert os.getcwd() == original_cwd
    return os.getpid()


@pytest.mark.parametrize("mode", ["process", "spawn"])
def test_bash_preserves_cwd_and_relative_reads(mode):
    # Isolate worker lifecycle and bound a possible nesting deadlock.
    script = '''
import os
import seamless
import uuid
from seamless.transformer import delayed, spawn
from test_contract_bash_cwd import _check_nested_bash_cwd
mode = os.environ["BASH_CWD_TEST_MODE"]
if mode == "spawn":
    spawn(1)
try:
    outer = delayed(_check_nested_bash_cwd)
    outer.local = True
    pid = outer(uuid.uuid4().hex).run()
    assert (pid != os.getpid()) == (mode == "spawn"), (pid, os.getpid(), mode)
finally:
    seamless.close()
print("CWD_OK")
'''
    env = os.environ.copy()
    env["BASH_CWD_TEST_MODE"] = mode
    env["PYTHONPATH"] = str(Path(__file__).parent) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=60
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "CWD_OK" in completed.stdout
