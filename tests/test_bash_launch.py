"""Phase 1: materialized bash scripts and bash-only execution environments.

Environment metadata is orthogonal to identity; use fresh nonce pins for every
raw transformation, including failures and record probes.
"""

import importlib
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
import uuid

import pytest


def _isolated(function):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", f"import test_bash_launch as m; m.{function}(); print('LAUNCH_OK')"],
        env=env, capture_output=True, text=True, timeout=25,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "LAUNCH_OK" in result.stdout


def test_every_pin_is_exported_even_when_equal_to_host(monkeypatch, tmp_path):
    module = importlib.import_module("seamless_transformer.execute_bash")
    value = "same value 'with quotes'"
    monkeypatch.setenv("launch_pin", value)
    module.write_bash_job(
        'printf "%s" "$launch_pin" > RESULT', ["launch_pin"], None,
        {"launch_pin": value}, {}, directory=str(tmp_path),
    )
    # The on-disk job must work independently of the materializer's variables.
    env = os.environ.copy()
    env.pop("launch_pin", None)
    result = subprocess.run(["bash", "transform.sh"], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "RESULT").read_text() == value


def _path_pin():
    module = importlib.import_module("seamless_transformer.execute_bash")
    code = 'trap - EXIT\n[[ "$PATH" == /no-such-tools ]] || exit 42\nprintf 42 > RESULT'
    result = module.execute_bash(
        code, ["PATH"], None, {"PATH": "/no-such-tools"}, {}, "mixed"
    )
    assert result == 42


def test_path_pin_does_not_hide_host_bash():
    _isolated("_path_pin")


def _large_script():
    module = importlib.import_module("seamless_transformer.execute_bash")
    code = "#" + "x" * 140000 + '\nprintf "%s" "$0" > RESULT\n'
    assert module.execute_bash(code, [], None, {}, {}, "mixed") == "transform.sh"


def test_script_larger_than_one_argument_runs_as_file():
    _isolated("_large_script")


def _start_marker():
    module = importlib.import_module("seamless_transformer.execute_bash")
    assert module.execute_bash(
        "test -f .seamless-started; printf 42 > RESULT", [], None, {}, {}, "mixed"
    ) == 42


def test_start_marker_exists_before_user_code():
    _isolated("_start_marker")


def _launch_failure():
    module = importlib.import_module("seamless_transformer.execute_bash")
    from seamless_transformer import SeamlessStreamTransformationError
    original = module.write_bash_job

    def remove_script(*args, **kwargs):
        result = original(*args, **kwargs)
        (Path(kwargs["directory"]) / "transform.sh").unlink()
        return result

    module.write_bash_job = remove_script
    try:
        module.execute_bash("printf 42 > RESULT", [], None, {}, {}, "mixed")
    except SeamlessStreamTransformationError as exc:
        message = str(exc).lower()
        assert "not entered" in message or "could not be started" in message, message
        assert "127" in message, message
        assert "transform.sh" in message, message
    else:
        raise AssertionError("an absent entry point must fail before user code")


def test_absent_script_is_classified_as_launch_failure():
    _isolated("_launch_failure")


def _raw_setup(monkeypatch):
    from seamless import Buffer
    module = importlib.import_module("seamless_transformer.run")
    monkeypatch.setattr(module, "get_record_mode", lambda: False)
    monkeypatch.setattr(
        module, "build_transformation_namespace_sync",
        lambda transformation: (
            transformation["__code_text__"],
            {"PINS": {"nonce": transformation["nonce"][2]}, "nonce": transformation["nonce"][2],
             "FILESYSTEM": {}, "OUTPUTPIN": "mixed"},
            {},
        ),
    )
    return module, Buffer


def _raw(module, Buffer, language, code, environment=None, *, probe=False, compiled=False, process_lock=True):
    from seamless import Checksum
    nonce = uuid.uuid4().hex
    transformation = {
        "__language__": language, "__code_text__": code,
        "__output__": ("result", "mixed", None),
        "nonce": ("str", None, Buffer(nonce, "str").get_checksum().hex()),
        "__meta__": {"process_lock": process_lock},
    }
    dunder = {}
    if environment is not None:
        env_buffer = Buffer(environment, "plain")
        env_buffer.tempref()
        dunder["__env__"] = env_buffer.get_checksum().hex()
    if probe:
        dunder["__record_probe__"] = {"mode": "capture"}
    if compiled:
        dunder["__compiled__"] = True
    checksum = Buffer(transformation, "plain").get_checksum()
    return module.run_transformation_dict(transformation, Checksum(checksum), dunder,
                                          scratch=True, require_value=True)


def test_bash_docker_stopgap_refuses_host_execution(monkeypatch, tmp_path):
    module, Buffer = _raw_setup(monkeypatch)
    sentinel = tmp_path / "host-executed"
    code = f"touch {shlex.quote(str(sentinel))}; printf 42 > RESULT"
    with pytest.raises(RuntimeError, match="container execution is not implemented yet; refusing to run on the host"):
        _raw(module, Buffer, "bash", code, {"docker": {"name": "ubuntu:22.04"}})
    assert not sentinel.exists()


@pytest.mark.parametrize("environment", [
    {"docker": {"name": "ubuntu:22.04"}}, {"conda_environment": "seamless1"},
])
@pytest.mark.parametrize("compiled", [False, True])
def test_non_bash_environment_fails_before_body(monkeypatch, tmp_path, environment, compiled):
    module, Buffer = _raw_setup(monkeypatch)
    sentinel = tmp_path / "body-executed"
    code = f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\nresult = 42"
    monkeypatch.setattr(module, "_is_compiled_transformation", lambda *args: compiled)
    monkeypatch.setattr(module, "call_compiled_transform", lambda *args: pytest.fail("compiled body ran"))
    with pytest.raises(RuntimeError, match="bash transformers only"):
        _raw(module, Buffer, "python", code, environment, compiled=compiled)
    assert not sentinel.exists()


@pytest.mark.parametrize("environment", [
    {"docker": {"name": "ubuntu:22.04"}}, {"conda_environment": "seamless1"},
])
def test_record_probe_carrying_environment_runs_on_host(monkeypatch, environment):
    module, Buffer = _raw_setup(monkeypatch)
    checksum = _raw(module, Buffer, "python", "result = 42", environment, probe=True)
    assert checksum.resolve().get_value("mixed") == 42


def _overlap():
    import tempfile
    from pytest import MonkeyPatch
    with MonkeyPatch.context() as monkeypatch, tempfile.TemporaryDirectory() as directory:
        module, Buffer = _raw_setup(monkeypatch)
        results, failures = {}, []
        cwd = os.getcwd()

        def run(index):
            own, other = [Path(directory) / f"active-{i}" for i in (index, 1 - index)]
            code = "\n".join([
                f"touch {shlex.quote(str(own))}",
                "deadline=$((SECONDS + 8))",
                f"while ! test -f {shlex.quote(str(other))}; do",
                '  test "$SECONDS" -lt "$deadline" || exit 91',
                "  sleep .01", "done",
                f'printf "%s" "$PWD" > RESULT',
            ])
            try:
                checksum = _raw(module, Buffer, "bash", code, process_lock=False)
                results[index] = checksum.resolve().get_value("mixed")
            except BaseException as exc:
                failures.append(repr(exc))

        threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(12)
        assert all(not thread.is_alive() for thread in threads)
        assert not failures, failures
        assert len(results) == 2 and results[0] != results[1], results
        assert os.getcwd() == cwd


def test_process_lock_false_bash_jobs_overlap_without_disturbing_each_other():
    _isolated("_overlap")


def _notify():
    import tempfile
    from seamless_transformer.process_lock import scope, current_holder
    module = importlib.import_module("seamless_transformer.execute_bash")
    with tempfile.TemporaryDirectory() as directory:
        active, release = [Path(directory) / name for name in ("active", "release")]
        failures = []

        def contender():
            deadline = time.monotonic() + 8
            try:
                while not active.exists():
                    assert time.monotonic() < deadline
                    time.sleep(.005)
                with scope(True, should_abort=lambda: time.monotonic() > deadline):
                    release.touch()
            except BaseException as exc:
                failures.append(repr(exc))
                release.touch()

        thread = threading.Thread(target=contender)
        with scope(True):
            holder = current_holder()
            thread.start()
            code = "\n".join([
                f"touch {shlex.quote(str(active))}",
                'printf x > "$SEAMLESS_PROCESS_LOCK_NOTIFY"',
                "deadline=$((SECONDS + 8))",
                f"while ! test -f {shlex.quote(str(release))}; do",
                '  test "$SECONDS" -lt "$deadline" || exit 92',
                "  sleep .01", "done", "printf 42 > RESULT",
            ])
            try:
                assert module.execute_bash(code, [], None, {}, {}, "mixed") == 42
                assert not holder.held
            finally:
                thread.join(10)
        assert not thread.is_alive()
        assert not failures, failures


def test_script_notification_still_drops_parents_process_lock():
    _isolated("_notify")
