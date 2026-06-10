import os
import re
import subprocess
import textwrap
import time
import uuid

import pytest

from seamless import Checksum

from seamless_transformer.api import cancel

_CHECKSUM_RE = re.compile(r"\b[a-f0-9]{64}\b")
_SLEEP_SECONDS = 3.0
_MIN_UNCACHED_SECONDS = 2.6


def _run_command(args, *, cwd, env=None):
    run_env = os.environ.copy()
    run_env["PYTHONUNBUFFERED"] = "1"
    if env:
        run_env.update(env)
    return subprocess.run(
        args,
        cwd=cwd,
        env=run_env,
        check=False,
        capture_output=True,
        text=True,
    )


def _popen_command(args, *, cwd, env=None):
    run_env = os.environ.copy()
    run_env["PYTHONUNBUFFERED"] = "1"
    if env:
        run_env.update(env)
    return subprocess.Popen(
        args,
        cwd=cwd,
        env=run_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _assert_success(proc):
    assert proc.returncode == 0, "\n".join(
        line
        for line in (proc.stdout or "").splitlines() + (proc.stderr or "").splitlines()
        if line
    )


def _write_remote_config(workdir, *, backend, project):
    cluster = f"cluster-{uuid.uuid4().hex}"
    (workdir / "seamless.yaml").write_text(
        "\n".join(
            [
                "- clusters:",
                f"    {cluster}:",
                "      type: local",
                "      workers: 1",
                "      frontends:",
                "        - hashserver:",
                f"            bufferdir: {workdir / 'buffers'}",
                "            conda: seamless1",
                "            port_start: 10000",
                "            port_end: 19999",
                "          database:",
                f"            database_dir: {workdir / 'buffers'}",
                "            conda: seamless1",
                "            port_start: 20000",
                "            port_end: 29999",
                "          jobserver:",
                "            conda: seamless1",
                "            network_interface: 0.0.0.0",
                "            port_start: 20000",
                "            port_end: 29999",
                "          daskserver:",
                "            network_interface: 0.0.0.0",
                "            port_start: 20000",
                "            port_end: 29999",
                "      default_queue: default",
                "      queues:",
                "        default:",
                "          conda: seamless1",
                "          interactive: true",
                "          walltime: 10m",
                "          memory: 30000MB",
                f"- project: {project}",
                "- execution: remote",
                f"- remote: {backend}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (workdir / "seamless.profile.yaml").write_text(
        f"- cluster: {cluster}\n",
        encoding="utf-8",
    )


def _cancel_running_checksum(checksum_file, *, cwd, env):
    last_proc = None
    for _attempt in range(80):
        proc = _run_command(
            ["seamless-cancel", str(checksum_file)],
            cwd=cwd,
            env=env,
        )
        last_proc = proc
        if proc.returncode == 0:
            return proc
        time.sleep(0.25)
    assert last_proc is not None
    assert False, "\n".join(
        line
        for line in (
            (last_proc.stdout or "").splitlines()
            + (last_proc.stderr or "").splitlines()
        )
        if line
    )


def _assert_replay_is_uncached(checksum_file, *, cwd, env):
    time.sleep(_SLEEP_SECONDS + 0.2)
    start = time.perf_counter()
    proc = _run_command(
        ["seamless-run-transformation", str(checksum_file)],
        cwd=cwd,
        env=env,
    )
    elapsed = time.perf_counter() - start
    _assert_success(proc)
    assert elapsed >= _MIN_UNCACHED_SECONDS, (
        f"replay finished in {elapsed:.3f}s, expected an uncached "
        f"{_SLEEP_SECONDS:.1f}s transformation"
    )
    return elapsed, proc


def _wait_for_checksum_file(checksum_file):
    for _attempt in range(300):
        if checksum_file.exists():
            text = checksum_file.read_text(encoding="utf-8").strip()
            if _CHECKSUM_RE.fullmatch(text):
                return text
        time.sleep(0.1)
    raise AssertionError(f"{checksum_file} was not written")


def _prewarm_cancel_backend(*, cwd, env):
    proc = _run_command(
        ["seamless-cancel", "7" * 64],
        cwd=cwd,
        env=env,
    )
    assert proc.returncode in (0, 2), "\n".join(
        line
        for line in (proc.stdout or "").splitlines() + (proc.stderr or "").splitlines()
        if line
    )


def test_cancel_cli_returns_zero_when_backend_cancels(monkeypatch, capsys):
    checksum = Checksum("3" * 64)
    calls = []

    monkeypatch.setattr(cancel, "_parse_checksum", lambda _arg: checksum)
    monkeypatch.setattr(cancel.seamless_config, "init", lambda **_kwargs: None)

    def fake_cancel_by_checksum(tf_checksum):
        calls.append(tf_checksum)
        return True, ["dask: canceled"]

    monkeypatch.setattr(cancel, "cancel_by_checksum", fake_cancel_by_checksum)

    assert cancel._main([checksum.hex()]) == 0
    assert calls == [checksum]
    assert capsys.readouterr().out.strip() == "dask: canceled"


def test_cancel_cli_returns_two_when_nothing_active(monkeypatch, capsys):
    checksum = Checksum("4" * 64)

    monkeypatch.setattr(cancel, "_parse_checksum", lambda _arg: checksum)
    monkeypatch.setattr(cancel.seamless_config, "init", lambda **_kwargs: None)
    monkeypatch.setattr(
        cancel,
        "cancel_by_checksum",
        lambda _checksum: (False, ["dask: not-running"]),
    )

    assert cancel._main([checksum.hex()]) == 2
    assert capsys.readouterr().out.strip() == "dask: not-running"


def test_cancel_by_checksum_uses_process_registry(monkeypatch):
    checksum = Checksum("5" * 64)

    class FakeCache:
        def cancel_by_checksum(self, tf_checksum, *, remote=True):
            assert tf_checksum == checksum
            assert remote is False
            return True

    monkeypatch.setattr(
        "seamless_transformer.transformation_cache.get_transformation_cache",
        lambda: FakeCache(),
    )
    monkeypatch.setattr(
        "seamless_dask.transformer_client.get_seamless_dask_client",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(cancel, "jobserver_remote", None, raising=False)

    canceled, messages = cancel.cancel_by_checksum(checksum)

    assert canceled is True
    assert "process: canceled" in messages


def test_cancel_cli_reads_checksum_file(monkeypatch, tmp_path, capsys):
    checksum = Checksum("6" * 64)
    checksum_file = tmp_path / "transformation.json.CHECKSUM"
    checksum_file.write_text(checksum.hex() + "\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr(cancel.seamless_config, "init", lambda **_kwargs: None)

    def fake_cancel_by_checksum(tf_checksum):
        calls.append(tf_checksum)
        return True, ["jobserver: canceled"]

    monkeypatch.setattr(cancel, "cancel_by_checksum", fake_cancel_by_checksum)

    assert cancel._main([str(checksum_file)]) == 0
    assert calls == [checksum]
    assert capsys.readouterr().out.strip() == "jobserver: canceled"


@pytest.mark.parametrize("backend", ["jobserver", "daskserver"])
def test_cancel_run_transformation_checksum_file(tmp_path, backend):
    workdir = tmp_path / "work"
    job_dir = workdir / "job"
    workdir.mkdir()
    env = {}
    token = f"cancel-run-transformation-{backend}-{uuid.uuid4().hex}"
    _write_remote_config(
        workdir,
        backend=backend,
        project=f"cancel-run-transformation-{backend}-{uuid.uuid4().hex}",
    )

    proc = _run_command(
        [
            "seamless-run",
            "--dry-run",
            "--upload",
            "-q",
            "-j",
            str(job_dir),
            f"sleep {_SLEEP_SECONDS} && echo {token}",
        ],
        cwd=workdir,
        env=env,
    )
    _assert_success(proc)
    checksum_file = job_dir / "transformation.json.CHECKSUM"
    assert _CHECKSUM_RE.fullmatch(checksum_file.read_text(encoding="utf-8").strip())
    _prewarm_cancel_backend(cwd=workdir, env=env)

    runner = _popen_command(
        ["seamless-run-transformation", str(checksum_file)],
        cwd=workdir,
        env=env,
    )
    try:
        cancel_proc = _cancel_running_checksum(checksum_file, cwd=workdir, env=env)
        expected_backend = "dask" if backend == "daskserver" else "jobserver"
        assert f"{expected_backend}: canceled" in cancel_proc.stdout
        stdout, stderr = runner.communicate(timeout=_SLEEP_SECONDS + 20)
    finally:
        if runner.poll() is None:
            runner.kill()
            runner.communicate()

    assert runner.returncode != 0, stdout + stderr
    assert "Transformation was canceled" in stderr
    _assert_replay_is_uncached(checksum_file, cwd=workdir, env=env)


@pytest.mark.parametrize("backend", ["jobserver", "daskserver"])
def test_cancel_delayed_producer_checksum_file(tmp_path, backend):
    workdir = tmp_path / "work"
    workdir.mkdir()
    env = {}
    token = f"cancel-delayed-producer-{backend}-{uuid.uuid4().hex}"
    checksum_file = workdir / "transformation.CHECKSUM"
    _write_remote_config(
        workdir,
        backend=backend,
        project=f"cancel-delayed-producer-{backend}-{uuid.uuid4().hex}",
    )
    producer = workdir / "producer.py"
    producer.write_text(
        textwrap.dedent(
            f"""
            import pathlib
            import seamless
            import seamless.config as seamless_config
            from seamless.transformer import delayed

            seamless_config.init()

            @delayed
            def sleep_then_print(seconds, token):
                import time
                time.sleep(seconds)
                print(token)
                return token

            tf = sleep_then_print({_SLEEP_SECONDS!r}, {token!r})
            tf.construct()
            tf.transformation_checksum.resolve().incref()
            pathlib.Path({str(checksum_file)!r}).write_text(
                tf.transformation_checksum.hex() + "\\n",
                encoding="utf-8",
            )
            print(tf.run())
            seamless.close()
            """
        ),
        encoding="utf-8",
    )
    _prewarm_cancel_backend(cwd=workdir, env=env)

    producer_proc = _popen_command(["python", str(producer)], cwd=workdir, env=env)
    try:
        _wait_for_checksum_file(checksum_file)
        cancel_proc = _cancel_running_checksum(checksum_file, cwd=workdir, env=env)
        expected_backend = "dask" if backend == "daskserver" else "jobserver"
        assert f"{expected_backend}: canceled" in cancel_proc.stdout
        stdout, stderr = producer_proc.communicate(timeout=_SLEEP_SECONDS + 20)
    finally:
        if producer_proc.poll() is None:
            producer_proc.kill()
            producer_proc.communicate()

    assert producer_proc.returncode != 0, stdout + stderr
    assert "Transformation was canceled" in stderr
    _assert_replay_is_uncached(checksum_file, cwd=workdir, env=env)
