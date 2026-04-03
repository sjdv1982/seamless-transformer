import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

CMD_DIR = Path(__file__).resolve().parent
_CHECKSUM_RE = re.compile(r"\b[a-f0-9]{64}\b")


def _run_command(
    args, *, cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    run_env = os.environ.copy()
    run_env["PYTHONUNBUFFERED"] = "1"
    if env:
        run_env.update(env)
    result = subprocess.run(
        args,
        cwd=cwd,
        env=run_env,
        check=False,
        capture_output=True,
        text=True,
    )
    return result


def _current_conda_env() -> str:
    env = os.environ.get("CONDA_DEFAULT_ENV")
    if env:
        return env
    prefix = Path(sys.prefix)
    if prefix.parent.name == "envs":
        return prefix.name
    return "base"


def _write_cluster_config(home: Path, *, conda_env: str) -> None:
    seamless_dir = home / ".seamless"
    seamless_dir.mkdir(parents=True, exist_ok=True)
    cluster_config = f"""
ssh-localhost:
  tunnel: false
  type: local
  workers: 1
  frontends:
    - hostname: localhost
      ssh_hostname: localhost
      hashserver:
        bufferdir: {home / "buffers"}
        conda: {conda_env}
        network_interface: localhost
        port_start: 61001
        port_end: 61049
      database:
        database_dir: {home / "database"}
        conda: {conda_env}
        network_interface: localhost
        port_start: 61050
        port_end: 61099
      jobserver:
        conda: {conda_env}
        network_interface: localhost
        port_start: 61100
        port_end: 61149
""".strip()
    (seamless_dir / "clusters.yaml").write_text(cluster_config + "\n")


def _write_test_config(workdir: Path) -> None:
    (workdir / "seamless.yaml").write_text(
        "- project: cmd-test\n- execution: remote\n- remote: jobserver\n"
    )
    (workdir / "seamless.profile.yaml").write_text("- cluster: ssh-localhost\n")
    (workdir / "data").symlink_to(CMD_DIR / "data", target_is_directory=True)


@pytest.mark.skipif(
    subprocess.run(
        ["ssh", "localhost", "true"], check=False, capture_output=True
    ).returncode
    != 0,
    reason="ssh localhost is required for the SSH-backed remote job test",
)
def test_write_remote_job_roundtrip_via_ssh(tmp_path):
    cluster_hostname = "localhost"
    conda_env = _current_conda_env()
    home = tmp_path / "home"
    workdir = tmp_path / "cmd"
    launcher_dir = tmp_path / "launcher-state"
    remote_job_dir = tmp_path / "remote-job-1"
    remote_job_dir_2 = tmp_path / "remote-job-2"

    home.mkdir()
    workdir.mkdir()
    launcher_dir.mkdir()
    _write_cluster_config(home, conda_env=conda_env)
    _write_test_config(workdir)

    env = {
        "HOME": str(home),
        "REMOTE_HTTP_LAUNCHER_DIR": str(launcher_dir),
    }
    command = "paste data/b.txt data/a.txt | awk 'NF == 2 && $1 > 10{print $2}'"
    prepare_args = [
        "seamless-run",
        "-q",
        "--dry",
        "--write-remote-job",
        str(remote_job_dir),
        "-c",
        command,
    ]
    prepare_args_2 = [
        "seamless-run",
        "-q",
        "--dry",
        "--write-remote-job",
        str(remote_job_dir_2),
        "-c",
        command,
    ]

    for path in (remote_job_dir, remote_job_dir_2):
        absent = _run_command(
            ["ssh", cluster_hostname, "test", "!", "-e", str(path)],
            cwd=workdir,
            env=env,
        )
        assert absent.returncode == 0, f"Remote job dir already exists: {path}"

    prepare = _run_command(prepare_args, cwd=workdir, env=env)
    assert prepare.returncode == 0, prepare.stderr or prepare.stdout
    output_text = "\n".join([prepare.stdout, prepare.stderr])
    assert "Transformation submitted to remote server" in output_text
    assert "Launch jobserver..." in output_text
    assert not _CHECKSUM_RE.search(output_text), output_text

    try:
        assert remote_job_dir.is_dir()
        transform_script = remote_job_dir / "transform.sh"
        assert transform_script.is_file()
        assert (remote_job_dir / "data" / "a.txt").is_file()
        assert (remote_job_dir / "data" / "b.txt").is_file()
        assert not (remote_job_dir / "RESULT").exists()

        jobserver_state = launcher_dir / "jobserver-ssh-localhost-rw-cmd-test.json"
        remote_state = _run_command(
            ["ssh", cluster_hostname, "test", "-f", str(jobserver_state)],
            cwd=workdir,
            env=env,
        )
        assert remote_state.returncode == 0, remote_state.stderr or remote_state.stdout

        remote_job = _run_command(
            ["ssh", cluster_hostname, "test", "-f", str(transform_script)],
            cwd=workdir,
            env=env,
        )
        assert remote_job.returncode == 0, remote_job.stderr or remote_job.stdout

        manual = _run_command(["bash", "transform.sh"], cwd=remote_job_dir, env=env)
        assert manual.returncode == 0, manual.stderr or manual.stdout
        assert (remote_job_dir / "RESULT").read_text() == "pears\npineapples\n"

        execute = _run_command(prepare_args_2, cwd=workdir, env=env)
        assert execute.returncode == 0, execute.stderr or execute.stdout
        assert remote_job_dir_2.is_dir()
    finally:
        pass

    assert remote_job_dir.exists()
    assert remote_job_dir_2.exists()
