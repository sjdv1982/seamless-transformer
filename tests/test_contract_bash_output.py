"""Phase A: bash drains both streams and releases its process-wide lock."""

import importlib
import os
import subprocess
import sys
import textwrap
import threading

import pytest


@pytest.mark.parametrize("returncode", [0, 7])
def test_bash_drains_one_mib_stderr_before_stdout_eof(returncode):
    # Isolate the timeout so a stream-draining regression cannot hang pytest.
    script = textwrap.dedent(
        f"""
        import contextlib
        import importlib
        import io
        import os
        from seamless_transformer import SeamlessStreamTransformationError
        module = importlib.import_module('seamless_transformer.execute_bash')
        code = "head -c 1048576 /dev/zero | tr '\\\\0' x >&2; printf 'stderr-tail' >&2; printf 'stdout-tail'; printf 42 > RESULT; exit {returncode}"
        module.write_bash_job = lambda code, *args, **kwargs: (os.environ.copy(), code)
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                result = module.execute_bash(code, [], None, {{}}, {{}}, 'mixed')
            except SeamlessStreamTransformationError as exc:
                assert {returncode} != 0
                assert 'Return code {returncode}' in str(exc)
            else:
                assert {returncode} == 0
                assert result == 42
        assert stdout.getvalue() == 'stdout-tail', repr(stdout.getvalue())
        assert stderr.getvalue() == 'x' * 1048576 + 'stderr-tail', len(stderr.getvalue())
        print('OK')
        """
    )
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "OK"


def test_bash_cleanup_failure_releases_lock_and_uses_safe_session(monkeypatch, tmp_path):
    from seamless_transformer import process_lock

    module = importlib.import_module("seamless_transformer.execute_bash")
    monkeypatch.setattr(module, "write_bash_job", lambda code, *args, **kwargs: (os.environ.copy(), code))
    monkeypatch.setattr(module.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path))
    # Avoid installing the production SIGTERM handler in the pytest process.
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    real_popen = module.subprocess.Popen
    old_cwd = os.getcwd()
    options = []

    def chdir(path):
        pytest.fail("bash execution must never change the process working directory")

    def rmtree(path, **kwargs):
        raise OSError("injected cleanup failure")

    def popen(*args, **kwargs):
        options.append(kwargs)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(module.os, "chdir", chdir)
    monkeypatch.setattr(module.shutil, "rmtree", rmtree)
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    with pytest.raises(OSError, match="injected cleanup failure"):
        with process_lock.scope(True):
            module.execute_bash("printf 42 > RESULT", [], None, {}, {}, "mixed")
    assert os.getcwd() == old_cwd
    import time
    deadline = time.monotonic() + 1
    with process_lock.scope(True, should_abort=lambda: time.monotonic() > deadline):
        assert process_lock.current_holder() is not None
    assert options[0].get("start_new_session") is True
    assert "preexec_fn" not in options[0]
