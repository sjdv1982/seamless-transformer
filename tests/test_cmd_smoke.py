import os
import shutil
import subprocess
from pathlib import Path

import pytest


def _ensure_cmd_available(cmd: str) -> None:
    if shutil.which(cmd) is None:
        pytest.skip(f"Command not available: {cmd}")


def _require_cmd_tests_enabled() -> None:
    if os.environ.get("SEAMLESS_RUN_CMD_TESTS", "").lower() not in ("1", "true", "yes"):
        pytest.skip("Set SEAMLESS_RUN_CMD_TESTS=1 to run cmd/CLI integration tests")


def _skip_if_docker_unusable() -> None:
    """The Seamless cmd stack may try to use Docker in some configurations."""
    sock = Path("/var/run/docker.sock")
    if sock.exists() and not os.access(sock, os.R_OK | os.W_OK):
        pytest.skip("Docker socket exists but is not accessible")


def _run(cmd, *, cwd: Path, timeout: int = 60, env: dict | None = None) -> subprocess.CompletedProcess:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=merged_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=True,
    )


def _copy_cmd_fixture(tmp_path: Path) -> Path:
    src = Path(__file__).resolve().parent / "cmd"
    dest = tmp_path / "cmd"
    shutil.copytree(src, dest)
    # The checked-in fixture is configured for remote execution (daskserver),
    # which requires user-specific cluster definitions. For CI/smoke tests,
    # force in-process execution so we can validate the cmd transformer quickly.
    cfg = "- project: cmd-test\n- execution: process\n- cluster: null\n"
    # Newer seamless-run looks for `.SEAMLESS.yaml`; keep the legacy lowercase files too.
    (dest / ".SEAMLESS.yaml").write_text(cfg, encoding="utf-8")
    (dest / "SEAMLESS.yaml").write_text(cfg, encoding="utf-8")
    (dest / "seamless.profile.yaml").write_text("- cluster: null\n", encoding="utf-8")
    (dest / "seamless.yaml").write_text("- project: cmd-test\n- execution: process\n", encoding="utf-8")
    return dest


def _isolated_env(tmp_path: Path) -> dict:
    # Ensure CLI tests don't write into the real home directory (often read-only in CI/sandboxes).
    base = tmp_path / "home"
    base.mkdir(parents=True, exist_ok=True)
    return {
        "HOME": str(base),
        "XDG_CACHE_HOME": str(base / ".cache"),
        "XDG_CONFIG_HOME": str(base / ".config"),
        "XDG_DATA_HOME": str(base / ".local" / "share"),
    }


def test_cmd_simple_pipeline(tmp_path: Path) -> None:
    _require_cmd_tests_enabled()
    _ensure_cmd_available("seamless-run")
    _skip_if_docker_unusable()
    cmd_dir = _copy_cmd_fixture(tmp_path)

    proc = _run(["bash", "simple.sh"], cwd=cmd_dir, timeout=60, env=_isolated_env(tmp_path))

    expected = (cmd_dir / "test-output" / "simple.out").read_text(encoding="utf-8").strip()
    got = proc.stdout.strip()
    assert got == expected


def test_cmd_calc_pi_nested_smoke(tmp_path: Path) -> None:
    _require_cmd_tests_enabled()
    _ensure_cmd_available("seamless-run")
    _ensure_cmd_available("seamless-init")
    _skip_if_docker_unusable()
    cmd_dir = _copy_cmd_fixture(tmp_path)

    # Keep this tiny; the original defaults are performance-focused.
    ntrials = "2"
    ndots = "20000"

    proc = _run(
        ["bash", "calc-pi-nesting.sh", ntrials, ndots],
        cwd=cmd_dir,
        timeout=60,
        env=_isolated_env(tmp_path),
    )

    # The script writes calc_pi-nested.job-* files containing floats.
    job_files = sorted(cmd_dir.glob("calc_pi-nested.job-*"))
    assert len(job_files) == int(ntrials)
    for f in job_files:
        value = float(f.read_text(encoding="utf-8").strip())
        assert 2.5 < value < 3.8

    # Ensure the wrapper produced some output and did not hang.
    assert proc.stdout.strip()


def test_cmd_input_file(tmp_path: Path) -> None:
    _require_cmd_tests_enabled()
    _ensure_cmd_available("seamless-run")
    _skip_if_docker_unusable()
    cmd_dir = _copy_cmd_fixture(tmp_path)

    proc = _run(["bash", "input-file.sh"], cwd=cmd_dir, timeout=120, env=_isolated_env(tmp_path))
    got = proc.stdout

    assert got
    assert "data/a.txt" in got
    assert "data/b.txt" in got
    # Input paths from file should appear in argtypes dictionary preparation logs.
    assert '"data/a.txt": {' in got
    assert '"data/b.txt": {' in got


def test_cmd_var_injection(tmp_path: Path) -> None:
    _require_cmd_tests_enabled()
    _ensure_cmd_available("seamless-run")
    _skip_if_docker_unusable()
    cmd_dir = _copy_cmd_fixture(tmp_path)

    value = "seamless-var-integration-ok"
    script = cmd_dir / "print_env.py"
    script.write_text(
        "import os\nprint(os.environ['SEAMLESS_VAR_TEST'])\n", encoding="utf-8"
    )
    proc = _run(
        ["seamless-run", "--var", f"SEAMLESS_VAR_TEST={value}", "python", "print_env.py"],
        cwd=cmd_dir,
        timeout=60,
        env=_isolated_env(tmp_path),
    )
    got = proc.stdout
    assert value in got
