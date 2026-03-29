import os
import re
import shutil
import subprocess
from pathlib import Path

CMD_DIR = Path(__file__).resolve().parent
_CHECKSUM_RE = re.compile(r"\b[a-f0-9]{64}\b")


def _run_command(args, *, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    return result


def _extract_checksum(result: subprocess.CompletedProcess[str]) -> str:
    text = "\n".join([result.stdout, result.stderr])
    matches = _CHECKSUM_RE.findall(text)
    if not matches:
        raise AssertionError(text)
    return matches[-1]


def test_write_remote_job_roundtrip(tmp_path):
    remote_job_dir = tmp_path / "remote-job"
    command = "paste data/b.txt data/a.txt | awk 'NF == 2 && $1 > 10{print $2}'"

    prepare = _run_command(
        [
            "seamless-run",
            "-q",
            "--stage",
            "remote-write-job",
            "--dry",
            "--upload",
            "--write-remote-job",
            str(remote_job_dir),
            "-c",
            command,
        ],
        cwd=CMD_DIR,
    )
    assert prepare.returncode == 0, prepare.stderr or prepare.stdout
    tf_checksum = _extract_checksum(prepare)

    execute = _run_command(
        [
            "seamless-run-transformation",
            "--stage",
            "remote-write-job",
            tf_checksum,
        ],
        cwd=CMD_DIR,
    )
    assert execute.returncode == 0, execute.stderr or execute.stdout

    try:
        assert remote_job_dir.is_dir()
        transform_script = remote_job_dir / "transform.sh"
        assert transform_script.is_file()
        assert (remote_job_dir / "data" / "a.txt").is_file()
        assert (remote_job_dir / "data" / "b.txt").is_file()
        assert not (remote_job_dir / "RESULT").exists()

        manual = _run_command(["bash", "transform.sh"], cwd=remote_job_dir)
        assert manual.returncode == 0, manual.stderr or manual.stdout
        assert (remote_job_dir / "RESULT").read_text() == "pears\npineapples\n"

        shutil.rmtree(remote_job_dir)
        execute_again = _run_command(
            [
                "seamless-run-transformation",
                "--stage",
                "remote-write-job",
                tf_checksum,
            ],
            cwd=CMD_DIR,
        )
        assert execute_again.returncode == 0, execute_again.stderr or execute_again.stdout
        assert remote_job_dir.is_dir()
    finally:
        shutil.rmtree(remote_job_dir, ignore_errors=True)

    assert not remote_job_dir.exists()
