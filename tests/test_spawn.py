import os
import subprocess
import sys
from pathlib import Path
from subprocess import TimeoutExpired

import pytest


ROOT = Path(__file__).resolve().parent


def _run_script(script: str, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("SEAMLESS_DEBUG_TRANSFORMATION", None)
    script_path = ROOT / script
    try:
        return subprocess.run(
            [sys.executable, str(script_path)],
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except TimeoutExpired as exc:
        pytest.fail(f"{script} hung for {timeout} seconds: {exc}")


def test_minimal_spawn():
    result = _run_script("minimal-spawn.py", timeout=10)
    expected = "STOP\n"
    assert result.returncode == 0, result.stderr
    assert result.stdout == expected


def test_minimal_spawn_without_guard():
    result = _run_script("minimal-spawn2.py", timeout=10)
    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    warning = (
        "seamless_transformer.worker.spawn() called outside MainProcess; ignoring"
    )
    assert warning in combined

    stop_lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert stop_lines and all(line == "STOP" for line in stop_lines)


def test_spawn_script_output():
    result = _run_script("spawn_test.py")
    expected = "\n".join(
        [
            "FUNC spawned process: False",
            "From spawned toplevel False",
            "SPAWN",
            "FUNC spawned process: True",
            "From spawned toplevel True",
            "MAIN spawned process: True",
            "FUNC spawned process: True",
            "From spawned toplevel False",
            "",
        ]
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == expected


def test_spawn_import_output():
    result = _run_script("spawn_import_test.py")
    expected = "\n".join(
        [
            "FUNC spawned process: True",
            "From spawned toplevel False",
            "",
        ]
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == expected
