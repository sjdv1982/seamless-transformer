import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


SLEEP_SECONDS = 5


def _current_conda_env() -> str:
    env = os.environ.get("CONDA_DEFAULT_ENV")
    if env:
        return env
    prefix = Path(sys.prefix)
    if prefix.parent.name == "envs":
        return prefix.name
    return "base"


def _run_command(args, *, cwd: Path, env: dict[str, str], timeout: float = 40):
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
local-cancel-jobserver:
  type: local
  workers: 1
  frontends:
    - hashserver:
        bufferdir: {home / "buffers"}
        conda: {conda_env}
        network_interface: localhost
        port_start: 61400
        port_end: 61449
      database:
        database_dir: {home / "database"}
        conda: {conda_env}
        network_interface: localhost
        port_start: 61450
        port_end: 61499
      jobserver:
        conda: {conda_env}
        network_interface: localhost
        port_start: 61500
        port_end: 61549
  default_queue: default
  queues:
    default:
      conda: {conda_env}
      interactive: true
      walltime: 10m
      memory: 2000MB
      maximum_jobs: 1
local_cluster: local-cancel-jobserver
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


def _write_test_config(workdir: Path, *, project: str) -> None:
    (workdir / "seamless.yaml").write_text(
        f"- project: {project}\n"
        "- execution: remote\n"
        "- remote: jobserver\n"
    )
    (workdir / "seamless.profile.yaml").write_text(
        "- cluster: local-cancel-jobserver\n"
    )


def _env_for(home: Path) -> dict[str, str]:
    return {
        "HOME": str(home),
        "RHL_FALLBACK_CONDA_SOURCE": str(
            Path(sys.prefix).parents[1] / "etc" / "profile.d" / "conda.sh"
        ),
    }


def _stop_services(workdir: Path, env: dict[str, str], *, project: str) -> None:
    for command in ("seamless-service-stop", "seamless-service-rm"):
        _run_command(
            [
                command,
                "--cluster",
                "local-cancel-jobserver",
                "--project",
                project,
            ],
            cwd=workdir,
            env=env,
            timeout=20,
        )


def _assert_replay_is_uncached(args, *, cwd: Path, env: dict[str, str]) -> None:
    time.sleep(SLEEP_SECONDS + 1)
    started = time.perf_counter()
    second = _run_command(args, cwd=cwd, env=env, timeout=40)
    duration = time.perf_counter() - started

    assert second.returncode == 0, second.stderr or second.stdout
    assert duration >= SLEEP_SECONDS - 1, (
        "re-submitted transformation returned too quickly; likely a stale DB hit",
        duration,
        second.stdout,
        second.stderr,
    )


def test_seamless_run_sigint_does_not_persist_late_jobserver_result(tmp_path):
    project = "cmd-cancel-jobserver-sigint"
    conda_env = _current_conda_env()
    home = tmp_path / "home"
    workdir = tmp_path / "cmd"
    home.mkdir()
    workdir.mkdir()
    _write_cluster_config(home, conda_env=conda_env)
    _write_test_config(workdir, project=project)
    env = _env_for(home)

    warmup = _run_command(
        ["seamless-run", "-q", "-c", "echo warmup"],
        cwd=workdir,
        env=env,
        timeout=60,
    )
    assert warmup.returncode == 0, warmup.stderr or warmup.stdout

    command = f"sleep {SLEEP_SECONDS} && echo cancel-jobserver-sigint"
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
        stdout, stderr = first.communicate(timeout=15)
        assert first.returncode == 130, (first.returncode, stdout, stderr)
        assert "Canceled transformation" in stderr
        _assert_replay_is_uncached(args, cwd=workdir, env=env)
    finally:
        if first.poll() is None:
            first.kill()
            first.communicate(timeout=5)
        _stop_services(workdir, env, project=project)


def test_python_transformation_cancel_does_not_persist_late_jobserver_result(tmp_path):
    project = "cmd-cancel-jobserver-python"
    conda_env = _current_conda_env()
    home = tmp_path / "home"
    workdir = tmp_path / "cmd"
    home.mkdir()
    workdir.mkdir()
    _write_cluster_config(home, conda_env=conda_env)
    _write_test_config(workdir, project=project)
    env = _env_for(home)
    checksum_file = workdir / "transformation.CHECKSUM"
    script = workdir / "cancel_delayed.py"
    script.write_text(
        "\n".join(
            [
                "import pathlib",
                "import threading",
                "import time",
                "import seamless",
                "import seamless.config as seamless_config",
                "from seamless.transformer import delayed",
                "",
                "seamless_config.init()",
                "",
                "@delayed",
                "def sleep_then_print(seconds):",
                "    import time",
                "    time.sleep(seconds)",
                "    print('cancel-jobserver-python')",
                "    return 'cancel-jobserver-python'",
                "",
                f"tf = sleep_then_print({float(SLEEP_SECONDS)!r})",
                "tf.construct()",
                "tf.transformation_checksum.resolve().incref()",
                f"pathlib.Path({str(checksum_file)!r}).write_text(",
                "    tf.transformation_checksum.hex() + '\\n', encoding='utf-8'",
                ")",
                "state = {}",
                "",
                "def run():",
                "    try:",
                "        state['value'] = tf.run()",
                "    except Exception as exc:",
                "        state['error'] = str(exc)",
                "",
                "thread = threading.Thread(target=run)",
                "thread.start()",
                "time.sleep(1.5)",
                "state['cancelled'] = tf.cancel()",
                "thread.join(20)",
                "seamless.close()",
                "print('CANCELLED', state.get('cancelled'))",
                "print('VALUE', state.get('value', ''))",
                "print('ERROR', state.get('error', ''))",
                "if not state.get('cancelled'):",
                "    raise SystemExit(1)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        warmup = _run_command(
            ["seamless-run", "-q", "-c", "echo warmup"],
            cwd=workdir,
            env=env,
            timeout=60,
        )
        assert warmup.returncode == 0, warmup.stderr or warmup.stdout

        first = _run_command(["python", str(script)], cwd=workdir, env=env, timeout=40)
        assert first.returncode == 0, first.stderr or first.stdout
        assert checksum_file.exists()

        _assert_replay_is_uncached(
            ["seamless-run-transformation", str(checksum_file)],
            cwd=workdir,
            env=env,
        )
    finally:
        _stop_services(workdir, env, project=project)
