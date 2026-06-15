"""Shared helpers for the cancellation test suite.

Contract under test: ``seamless/context-internals-design-pass3.md`` Part III.

These helpers carry **no knowledge of the cancellation implementation**. They only
use the public surface (``TransformationCache.run`` / ``transformation_status`` /
``cancel_by_checksum``, the high-level ``tf.run()/tf.cancel()`` handle) and the
``seamless-run`` cluster tooling, exactly as the existing tests do.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from seamless import Checksum

# Long enough that a cancel at ~1.5 s lands solidly mid-run, short enough to keep
# the remote tests bearable.
SLEEP_SECONDS = 6


def cs(char: str) -> Checksum:
    """A 64-hex checksum built from a single repeated character (test sentinel)."""
    return Checksum(char * 64)


# --------------------------------------------------------------------------- #
# In-process: a controllable stand-in for ``run_transformation_dict``.
# --------------------------------------------------------------------------- #
class FakeRunner:
    """Blocks until ``release`` is set, so a 'running' transformation can be held
    still while the test manipulates membership. ``calls`` counts executions, which
    is how dedup is observed (one membership set => one execution)."""

    def __init__(self, result_char: str = "b"):
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self._result = cs(result_char)

    def __call__(self, *args, **kwargs):
        self.calls += 1
        self.started.set()
        assert self.release.wait(15), "FakeRunner was never released"
        return self._result

    def wait_started(self, timeout: float = 5) -> None:
        assert self.started.wait(timeout), "FakeRunner never started"


# --------------------------------------------------------------------------- #
# Remote: local-cluster lifecycle (modelled on tests/cmd/test_cancel_*).
# --------------------------------------------------------------------------- #
def current_conda_env() -> str:
    env = os.environ.get("CONDA_DEFAULT_ENV")
    if env:
        return env
    prefix = Path(sys.prefix)
    if prefix.parent.name == "envs":
        return prefix.name
    return "base"


def _frontends_block(remote_kind: str, *, home: Path, conda_env: str, port_base: int) -> str:
    hs = f"""      hashserver:
        bufferdir: {home / "buffers"}
        conda: {conda_env}
        network_interface: localhost
        port_start: {port_base}
        port_end: {port_base + 49}"""
    db = f"""      database:
        database_dir: {home / "database"}
        conda: {conda_env}
        network_interface: localhost
        port_start: {port_base + 50}
        port_end: {port_base + 99}"""
    if remote_kind == "jobserver":
        rk = f"""      jobserver:
        conda: {conda_env}
        network_interface: localhost
        port_start: {port_base + 100}
        port_end: {port_base + 149}"""
    elif remote_kind == "daskserver":
        rk = f"""      daskserver:
        network_interface: localhost
        port_start: {port_base + 100}
        port_end: {port_base + 149}"""
    else:  # pragma: no cover - defensive
        raise ValueError(remote_kind)
    return "\n".join([hs, db, rk])


def _write_cluster_config(
    home: Path, *, cluster_name: str, conda_env: str, remote_kind: str, port_base: int
) -> None:
    seamless_dir = home / ".seamless"
    seamless_dir.mkdir(parents=True, exist_ok=True)
    cluster_config = f"""
{cluster_name}:
  type: local
  workers: 1
  frontends:
    - {_frontends_block(remote_kind, home=home, conda_env=conda_env, port_base=port_base).lstrip()}
  default_queue: default
  queues:
    default:
      conda: {conda_env}
      interactive: true
      walltime: 10m
      memory: 2000MB
      maximum_jobs: 2
local_cluster: {cluster_name}
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


def _write_test_config(workdir: Path, *, project: str, cluster_name: str, remote_kind: str) -> None:
    (workdir / "seamless.yaml").write_text(
        f"- project: {project}\n- execution: remote\n- remote: {remote_kind}\n"
    )
    (workdir / "seamless.profile.yaml").write_text(f"- cluster: {cluster_name}\n")


# A self-contained tenant: builds one deterministic transformation (identity fixed
# by ``nonce``) that records markers and sleeps, then performs one of:
#   complete | soft_cancel | hard_cancel | crash
_TENANT_SCRIPT = r'''
import argparse, os, pathlib, threading, time, uuid

import seamless
import seamless.config as seamless_config
from seamless.transformer import delayed

seamless_config.init()


def make_tf(nonce, marker_dir, sleep_seconds):
    @delayed
    def run_marker(nonce, marker_dir, sleep_seconds):
        import os, time, uuid
        os.makedirs(marker_dir, exist_ok=True)
        token = uuid.uuid4().hex
        with open(os.path.join(marker_dir, "started-" + token), "w") as fh:
            fh.write(str(os.getpid()))
        time.sleep(sleep_seconds)
        with open(os.path.join(marker_dir, "finished-" + token), "w") as fh:
            fh.write("done")
        return "done-" + nonce

    return run_marker(nonce, marker_dir, sleep_seconds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nonce", required=True)
    ap.add_argument("--marker-dir", required=True)
    ap.add_argument("--sleep", type=float, required=True)
    ap.add_argument("--action", required=True)
    ap.add_argument("--cancel-after", type=float, default=1.5)
    args = ap.parse_args()

    tf = make_tf(args.nonce, args.marker_dir, args.sleep)
    tf.construct()
    checksum = tf.transformation_checksum
    print("CHECKSUM", checksum.hex(), flush=True)

    state = {}

    def runner():
        try:
            state["value"] = tf.run()
        except BaseException as exc:  # noqa: BLE001 - report whatever happened
            state["error"] = repr(exc)

    if args.action == "complete":
        runner()
    elif args.action == "crash":
        # The orchestrator SIGKILLs us mid-run; run to completion otherwise.
        runner()
    elif args.action in ("soft_cancel", "hard_cancel"):
        th = threading.Thread(target=runner)
        th.start()
        time.sleep(args.cancel_after)
        if args.action == "soft_cancel":
            state["cancelled"] = bool(tf.cancel())
        else:
            from seamless_transformer.transformation_cache import (
                get_transformation_cache,
            )
            state["cancelled"] = bool(
                get_transformation_cache().cancel_by_checksum(checksum)
            )
        th.join(args.sleep + 20)
    else:
        raise SystemExit("unknown action: " + args.action)

    print("VALUE", state.get("value", ""), flush=True)
    print("ERROR", state.get("error", ""), flush=True)
    print("CANCELLED", state.get("cancelled", ""), flush=True)
    try:
        seamless.close()
    except BaseException:
        pass


if __name__ == "__main__":
    main()
'''


class Cluster:
    """A live local cluster plus helpers to launch tenant processes against it."""

    def __init__(self, *, workdir: Path, env: dict, project: str, cluster_name: str):
        self.workdir = workdir
        self.env = env
        self.project = project
        self.cluster_name = cluster_name
        self.tenant_script = workdir / "tenant.py"
        self.tenant_script.write_text(_TENANT_SCRIPT, encoding="utf-8")
        self._marker_root = workdir / "markers"
        self._marker_root.mkdir(exist_ok=True)

    # -- markers ----------------------------------------------------------- #
    def new_marker_dir(self, name: str) -> Path:
        d = self._marker_root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def count(marker_dir: Path, prefix: str) -> int:
        return len(list(marker_dir.glob(prefix + "-*")))

    @staticmethod
    def wait_for(marker_dir: Path, prefix: str, *, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if Cluster.count(marker_dir, prefix) >= 1:
                return True
            time.sleep(0.1)
        return False

    # -- tenants ----------------------------------------------------------- #
    def launch_tenant(
        self, *, action: str, nonce: str, marker_dir: Path, sleep: float = SLEEP_SECONDS,
        cancel_after: float = 1.5,
    ) -> subprocess.Popen:
        run_env = os.environ.copy()
        run_env.update(self.env)
        run_env["PYTHONUNBUFFERED"] = "1"
        args = [
            sys.executable,
            str(self.tenant_script),
            "--nonce", nonce,
            "--marker-dir", str(marker_dir),
            "--sleep", str(sleep),
            "--action", action,
            "--cancel-after", str(cancel_after),
        ]
        return subprocess.Popen(
            args, cwd=self.workdir, env=run_env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

    def run_command(self, args, *, timeout: float = 60) -> subprocess.CompletedProcess:
        run_env = os.environ.copy()
        run_env.update(self.env)
        run_env["PYTHONUNBUFFERED"] = "1"
        return subprocess.run(
            args, cwd=self.workdir, env=run_env,
            check=False, capture_output=True, text=True, timeout=timeout,
        )

    def replay_is_uncached(self, *, nonce: str, marker_dir: Path) -> float:
        """Re-run the same transformation to completion; return its duration.

        Used after kill-tests: a fresh execution (~SLEEP_SECONDS) proves the
        cancelled run did not poison the database with a partial/bogus result.
        """
        started = time.perf_counter()
        proc = self.launch_tenant(action="complete", nonce=nonce, marker_dir=marker_dir)
        out, err = proc.communicate(timeout=SLEEP_SECONDS + 60)
        duration = time.perf_counter() - started
        assert proc.returncode == 0, ("replay tenant failed", out, err)
        assert ("VALUE done-" + nonce) in out, ("replay produced no result", out, err)
        return duration


@contextlib.contextmanager
def build_cluster(home_root: Path, *, remote_kind: str, cluster_name: str, port_base: int):
    """Write config, start services (warmup), yield a :class:`Cluster`, then stop+rm."""
    conda_env = current_conda_env()
    home = home_root / "home"
    workdir = home_root / "work"
    home.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    project = f"cancel-mt-{remote_kind}"

    _write_cluster_config(
        home, cluster_name=cluster_name, conda_env=conda_env,
        remote_kind=remote_kind, port_base=port_base,
    )
    _write_test_config(workdir, project=project, cluster_name=cluster_name, remote_kind=remote_kind)

    env = {
        "HOME": str(home),
        "RHL_FALLBACK_CONDA_SOURCE": str(
            Path(sys.prefix).parents[1] / "etc" / "profile.d" / "conda.sh"
        ),
    }
    cluster = Cluster(workdir=workdir, env=env, project=project, cluster_name=cluster_name)

    # Warm up: first remote call boots the services.
    warmup = cluster.run_command(["seamless-run", "-q", "-c", "echo warmup"], timeout=120)
    assert warmup.returncode == 0, ("cluster warmup failed", warmup.stdout, warmup.stderr)

    try:
        yield cluster
    finally:
        for command in ("seamless-service-stop", "seamless-service-rm"):
            try:
                cluster.run_command(
                    [command, "--cluster", cluster_name, "--project", project], timeout=30
                )
            except Exception:
                pass
