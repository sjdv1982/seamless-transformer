import os
import json
import signal
import subprocess
import sys
import time
from pathlib import Path


def _current_conda_env() -> str:
    env = os.environ.get("CONDA_DEFAULT_ENV")
    if env:
        return env
    prefix = Path(sys.prefix)
    if prefix.parent.name == "envs":
        return prefix.name
    return "base"


def _run_command(args, *, cwd: Path, env: dict[str, str], timeout: float = 30):
    run_env = os.environ.copy()
    run_env.update(env)
    run_env["PYTHONUNBUFFERED"] = "1"
    return subprocess.run(
        args,
        cwd=cwd,
        env=run_env,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _write_cluster_config(home: Path, *, conda_env: str) -> None:
    seamless_dir = home / ".seamless"
    seamless_dir.mkdir(parents=True, exist_ok=True)
    cluster_config = f"""
local-cancel:
  type: local
  workers: 1
  frontends:
    - hashserver:
        bufferdir: {home / "buffers"}
        conda: {conda_env}
        network_interface: localhost
        port_start: 61200
        port_end: 61249
      database:
        database_dir: {home / "database"}
        conda: {conda_env}
        network_interface: localhost
        port_start: 61250
        port_end: 61299
      daskserver:
        network_interface: localhost
        port_start: 61300
        port_end: 61349
  default_queue: default
  queues:
    default:
      conda: {conda_env}
      interactive: true
      walltime: 10m
      memory: 2000MB
      maximum_jobs: 1
local_cluster: local-cancel
""".strip()
    (seamless_dir / "clusters.yaml").write_text(cluster_config + "\n")

    conda_setup = home / ".remote-http-launcher" / "conda-setup.json"
    conda_setup.parent.mkdir(parents=True, exist_ok=True)
    conda_base = Path(sys.prefix).parents[1]
    conda_setup.write_text(
        json.dumps(
            {
                "conda_source": str(conda_base / "etc" / "profile.d" / "conda.sh"),
                "conda_base": str(conda_base),
                "envs": [str(Path(sys.prefix))],
            }
        )
        + "\n"
    )


def _write_test_config(workdir: Path) -> None:
    (workdir / "seamless.yaml").write_text(
        "- project: cmd-cancel-test\n"
        "- execution: remote\n"
        "- remote: daskserver\n"
    )
    (workdir / "seamless.profile.yaml").write_text("- cluster: local-cancel\n")


def test_seamless_run_sigint_does_not_persist_late_dask_result(tmp_path):
    sleep_seconds = 5
    conda_env = _current_conda_env()
    home = tmp_path / "home"
    workdir = tmp_path / "cmd"
    home.mkdir()
    workdir.mkdir()
    _write_cluster_config(home, conda_env=conda_env)
    _write_test_config(workdir)

    env = {
        "HOME": str(home),
        "RHL_FALLBACK_CONDA_SOURCE": str(
            Path(sys.prefix).parents[1] / "etc" / "profile.d" / "conda.sh"
        ),
    }
    warmup = _run_command(
        ["seamless-run", "-q", "-c", "echo warmup"],
        cwd=workdir,
        env=env,
        timeout=60,
    )
    assert warmup.returncode == 0, warmup.stderr or warmup.stdout

    command = f"sleep {sleep_seconds} && echo cancel-regression"
    args = ["seamless-run", "-q", "-c", command]

    first_env = os.environ.copy()
    first_env.update(env)
    first_env["PYTHONUNBUFFERED"] = "1"
    first = subprocess.Popen(
        args,
        cwd=workdir,
        env=first_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        time.sleep(1.5)
        first.send_signal(signal.SIGINT)
        stdout, stderr = first.communicate(timeout=10)
    finally:
        if first.poll() is None:
            first.kill()
            first.communicate(timeout=5)

    assert first.returncode == 130, (first.returncode, stdout, stderr)
    assert "Canceled transformation" in stderr

    time.sleep(sleep_seconds + 1)

    started = time.perf_counter()
    second = _run_command(args, cwd=workdir, env=env, timeout=30)
    duration = time.perf_counter() - started

    try:
        assert second.returncode == 0, second.stderr or second.stdout
        assert "cancel-regression" in (second.stdout + second.stderr)
        assert duration >= sleep_seconds - 1, (
            "re-submitted command returned too quickly; likely a stale DB hit",
            duration,
            second.stdout,
            second.stderr,
        )
    finally:
        _run_command(
            [
                "seamless-service-stop",
                "--cluster",
                "local-cancel",
                "--project",
                "cmd-cancel-test",
            ],
            cwd=workdir,
            env=env,
            timeout=20,
        )
        _run_command(
            [
                "seamless-service-rm",
                "--cluster",
                "local-cancel",
                "--project",
                "cmd-cancel-test",
            ],
            cwd=workdir,
            env=env,
            timeout=20,
        )
