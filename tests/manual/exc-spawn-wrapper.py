import subprocess
import sys
from pathlib import Path


def run_expected_failure(script_name):
    script = Path(__file__).with_name(script_name)
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = proc.communicate()

    if stdout:
        print(f"=== {script_name} stdout ===")
        print(stdout, end="")
    if stderr:
        print(f"=== {script_name} stderr ===", file=sys.stderr)
        print(stderr, end="", file=sys.stderr)

    if proc.returncode == 0:
        raise SystemExit(f"{script_name} unexpectedly exited with status 0")

    print(f"{script_name} exited with expected status {proc.returncode}")


if __name__ == "__main__":
    run_expected_failure("exc-spawn.py")
    run_expected_failure("exc.py")
