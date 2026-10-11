"""Undeclared bash nesting completes on a jobserver with one worker."""

import importlib.util
from pathlib import Path
import re
import shlex
import sys
import uuid


def test_undeclared_bash_seamless_run_nesting_on_single_worker(tmp_path):
    helper_path = Path(__file__).parent / "cancellation" / "_harness.py"
    spec = importlib.util.spec_from_file_location("bash_notify_cluster_harness", helper_path)
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    home, workdir = tmp_path / "home", tmp_path / "work"
    home.mkdir()
    workdir.mkdir()
    cluster_name = "bash-notify-" + uuid.uuid4().hex
    project = cluster_name
    harness._write_cluster_config(home, cluster_name=cluster_name, conda_env=harness.current_conda_env(), remote_kind="jobserver", port_base=62000)
    harness._write_test_config(workdir, project=project, cluster_name=cluster_name, remote_kind="jobserver")
    assert "workers: 1" in (home / ".seamless" / "clusters.yaml").read_text()
    env = {
        "HOME": str(home),
        "RHL_FALLBACK_CONDA_SOURCE": str(Path(sys.prefix).parents[1] / "etc" / "profile.d" / "conda.sh"),
    }
    cluster = harness.Cluster(workdir=workdir, env=env, project=project, cluster_name=cluster_name)
    try:
        warmup = cluster.run_command(["seamless-run", "-q", "-c", "echo warmup"], timeout=120)
        assert warmup.returncode == 0, warmup.stdout + warmup.stderr
        nonce = uuid.uuid4().hex
        leaf = 'printf "leaf:%s:' + nonce + '\\n" "$PPID"'
        # The nested independent CLI discovers configuration in its cwd.
        # Keep the outer job directory while giving that CLI the original project.
        outer = ('printf "outer:%s\\n" "$PPID"; (cd ' + shlex.quote(str(workdir))
                 + ' && seamless-run -q -c ' + shlex.quote(leaf) + ')')
        result = cluster.run_command(["seamless-run", "-q", "-c", outer], timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        outer_pid = re.search(r"outer:(\d+)", result.stdout)
        leaf_pid = re.search(r"leaf:(\d+):" + nonce, result.stdout)
        assert outer_pid and leaf_pid, result.stdout + result.stderr
        assert outer_pid.group(1) == leaf_pid.group(1), "nested leaf must run in the holder's sole worker process"
    finally:
        # Unlike the legacy build_cluster helper, also clean up if warmup fails.
        for command in ("seamless-service-stop", "seamless-service-rm"):
            try:
                cluster.run_command([command, "--cluster", cluster_name, "--project", project], timeout=30)
            except Exception:
                # Still attempt removal if stopping a half-started service failed.
                continue
