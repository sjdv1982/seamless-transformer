"""Directory pin contracts for seamless-run, locally and on a jobserver."""

import json
import os
from pathlib import Path
import subprocess
import sys
import uuid


def _isolated_env(root: Path, *, conda_setup: bool = False) -> dict[str, str]:
    home = root / "home"
    home.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "SEAMLESS_CACHE": str(root / "cache"),
            "XDG_CONFIG_HOME": str(root / "xdg-config"),
            "XDG_CACHE_HOME": str(root / "xdg-cache"),
            "XDG_DATA_HOME": str(root / "xdg-data"),
        }
    )
    if conda_setup:
        conda_base = Path(sys.prefix).parents[1]
        env["RHL_FALLBACK_CONDA_SOURCE"] = str(
            conda_base / "etc" / "profile.d" / "conda.sh"
        )
    return env


def _stop_local_services(env: dict[str, str]) -> None:
    home = Path(env["HOME"])
    keys = [
        path.stem
        for path in (home / ".remote-http-launcher" / "server").glob("*.json")
    ]
    if not keys:
        return
    for command in ("rhl-stop", "rhl-rm"):
        result = subprocess.run(
            [command, *keys],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def _dry_transformation(root: Path, options: list[str], command: str) -> dict:
    workdir = root / "work"
    jobdir = root / "job"
    workdir.mkdir(parents=True, exist_ok=True)
    env = _isolated_env(root)
    try:
        result = subprocess.run(
            [
                "seamless-run",
                "--local",
                "--dry",
                "-j",
                str(jobdir),
                *options,
                command,
            ],
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return json.loads((jobdir / "transformation.json").read_text())
    finally:
        _stop_local_services(env)


def test_directory_argument_declares_folder(tmp_path):
    source = tmp_path / "work" / "source"
    source.mkdir(parents=True)
    (source / "member.txt").write_text("directory input\n")

    transformation = _dry_transformation(tmp_path, ["-i", "source"], "echo ready")

    input_format = transformation["__format__"]["source"]
    assert input_format["celltype"] == "folder"
    assert input_format["filesystem"]["mode"] == "directory"
    input_pin = transformation["source"]
    assert input_pin[0] == "folder"


def test_result_targets_declare_folder(tmp_path):
    transformation = _dry_transformation(
        tmp_path, ["--capture", "result"], "mkdir -p result && echo ready > result/file.txt"
    )

    assert transformation["__output__"][0] == "result"
    assert transformation["__output__"][1] == "folder"


def _write_jobserver_config(home: Path, *, conda_env: str) -> None:
    buffer_dir = home / "jobserver-only-buffers"
    database_dir = home / "database"
    cluster_config = f"""
local-directory-pins:
  type: local
  workers: 1
  frontends:
    - hashserver:
        bufferdir: {buffer_dir}
        conda: {conda_env}
        network_interface: localhost
        port_start: 61600
        port_end: 61649
      database:
        database_dir: {database_dir}
        conda: {conda_env}
        network_interface: localhost
        port_start: 61650
        port_end: 61699
      jobserver:
        conda: {conda_env}
        network_interface: localhost
        port_start: 61700
        port_end: 61749
  default_queue: default
  queues:
    default:
      conda: {conda_env}
      interactive: true
      walltime: 10m
      memory: 2000MB
      maximum_jobs: 1
local_cluster: local-directory-pins
""".strip()
    seamless_dir = home / ".seamless"
    seamless_dir.mkdir(parents=True, exist_ok=True)
    (seamless_dir / "clusters.yaml").write_text(cluster_config + "\n")

    conda_base = Path(sys.prefix).parents[1]
    conda_setup = home / ".remote-http-launcher" / "conda-setup.json"
    conda_setup.parent.mkdir(parents=True, exist_ok=True)
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


def _stop_jobserver(workdir: Path, env: dict[str, str], *, project: str) -> None:
    for command in ("seamless-service-stop", "seamless-service-rm"):
        subprocess.run(
            [command, "--cluster", "local-directory-pins", "--project", project],
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=40,
            check=False,
        )


def test_directory_inputs_and_targets_on_jobserver_without_buffer_directory(tmp_path):
    project = "cmd-directory-pins-" + uuid.uuid4().hex[:12]
    env = _isolated_env(tmp_path, conda_setup=True)
    home = Path(env["HOME"])
    conda_env = os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name
    _write_jobserver_config(home, conda_env=conda_env)

    workdir = tmp_path / "work"
    source = workdir / "source"
    workdir.mkdir()
    source.mkdir()
    (source / "member.txt").write_text("from the folder pin\n")
    (workdir / "consume.sh").write_text(
        "#!/bin/bash\nset -e\nmkdir -p result\n"
        "cat source/member.txt > result/copied.txt\n"
    )
    (workdir / "seamless.yaml").write_text(
        f"- project: {project}\n- execution: remote\n- remote: jobserver\n"
    )
    (workdir / "seamless.profile.yaml").write_text(
        "- cluster: local-directory-pins\n"
    )

    # The worker hashserver owns this buffer directory. The isolated client
    # HOME has no configured local read-buffer directory pointing at it.
    worker_buffers = home / "jobserver-only-buffers"
    assert not worker_buffers.exists()
    assert not any(
        key in env
        for key in (
            "SEAMLESS_READ_BUFFER_DIRECTORY",
            "SEAMLESS_READ_BUFFER_DIRECTORIES",
        )
    )

    try:
        result = subprocess.run(
            [
                "seamless-run",
                "-q",
                "-y",
                "-i",
                "source",
                "--capture",
                "result",
                "-c",
                "bash consume.sh",
            ],
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert (workdir / "result" / "copied.txt").read_text() == "from the folder pin\n"
    finally:
        _stop_jobserver(workdir, env, project=project)
