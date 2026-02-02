from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_SKIP_EXIT = 77


def _run_script(path: Path, env: dict[str, str]) -> None:
    proc = subprocess.run(
        [sys.executable, str(path)],
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode == _SKIP_EXIT:
        reason = (proc.stderr or proc.stdout or "").strip()
        pytest.skip(reason or "Persistent scratch prerequisites not satisfied")
    if proc.returncode != 0:
        output = "\n".join(
            line
            for line in (proc.stdout or "").splitlines() + (proc.stderr or "").splitlines()
            if line
        )
        raise AssertionError(
            f"Persistent scratch script failed ({path.name}):\n{output}"
        )


def test_scratch_persistent_roundtrip(tmp_path: Path) -> None:
    state_path = tmp_path / "scratch_persistent_state.json"
    base_dir = Path(__file__).parent
    write_script = base_dir / "scratch_persistent_write.py"
    verify_script = base_dir / "scratch_persistent_verify.py"

    env = os.environ.copy()
    env["SEAMLESS_SCRATCH_PERSISTENT_PATH"] = str(state_path)

    _run_script(write_script, env)
    _run_script(verify_script, env)
