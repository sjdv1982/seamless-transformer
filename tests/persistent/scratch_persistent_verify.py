from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import seamless
import seamless.config
import seamless_config as _config
from seamless import Buffer, CacheMissError, Checksum
from seamless.caching.buffer_cache import get_buffer_cache
from seamless.transformer import delayed
from seamless_transformer.transformation_cache import get_transformation_cache

_SKIP_EXIT = 77


def _skip(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(_SKIP_EXIT)


async def _assert_rev_present(result_checksum: Checksum, tf_checksum: Checksum) -> None:
    from seamless_remote import database_remote

    rev = await database_remote.get_rev_transformations(result_checksum)
    if not rev or tf_checksum not in rev:
        raise RuntimeError(
            "Persistent reverse transformation cache missing for scratch checksum"
        )


async def _undo_transformation(tf_checksum: Checksum, result_checksum: Checksum) -> None:
    from seamless_remote import database_remote

    ok = await database_remote.undo_transformation_result(tf_checksum, result_checksum)
    if ok is False:
        raise RuntimeError("Failed to contest transformation result")


def _run_cli(
    tf_checksum: Checksum, *, fingertip: bool, workdir: Path
) -> subprocess.CompletedProcess[str]:
    exe = "seamless-run-transformation"
    cmd = [exe, tf_checksum.hex(), "--scratch", "--stage", "persistent"]
    if fingertip:
        cmd.append("--fingertip")
    if shutil.which(exe) is None:
        cmd = [
            sys.executable,
            "-m",
            "seamless_transformer.api.run_transformation",
            tf_checksum.hex(),
            "--scratch",
            "--stage",
            "persistent",
        ]
        if fingertip:
            cmd.append("--fingertip")
    return subprocess.run(
        cmd,
        cwd=workdir,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=10,
    )


def main() -> None:
    try:
        try:
            workdir = Path(__file__).resolve().parent.parent
            seamless.config.set_workdir(workdir)
            seamless.config.set_stage("persistent")
        except _config.ConfigurationError as exc:
            _skip(f"Persistent scratch requires configured remote: {exc}")

        try:
            from seamless_remote import buffer_remote, database_remote
        except Exception as exc:
            _skip(f"seamless_remote is required: {exc}")

        if not buffer_remote.has_write_server() or not buffer_remote.has_read_server():
            _skip("Buffer read/write servers must be configured")
        if not database_remote.has_write_server() or not database_remote.has_read_server():
            _skip("Database read/write servers must be configured")

        state_path = os.environ.get("SEAMLESS_SCRATCH_PERSISTENT_PATH")
        if not state_path:
            raise RuntimeError("SEAMLESS_SCRATCH_PERSISTENT_PATH is not set")
        payload = json.loads(Path(state_path).read_text(encoding="utf-8"))
        result_checksum = Checksum(payload["result_checksum"])
        tf_checksum = Checksum(payload["tf_checksum"])
        expected_value = float(payload["result_value"])

        cache = get_transformation_cache()
        if cache.get_reverse_transformations(result_checksum):
            raise RuntimeError(
                "In-process reverse cache must be empty for persistent test"
            )

        get_buffer_cache().purge_scratch(result_checksum)
        try:
            result_checksum.resolve()
        except CacheMissError:
            pass
        else:
            raise RuntimeError("Scratch result unexpectedly resolvable after purge")

        asyncio.run(_assert_rev_present(result_checksum, tf_checksum))

        value = asyncio.run(result_checksum.fingertip("mixed"))
        if abs(value - expected_value) > 1e-6:
            raise RuntimeError(f"Unexpected scratch recompute value: {value!r}")

        get_buffer_cache().purge_scratch(result_checksum)
        try:
            result_checksum.resolve()
        except CacheMissError:
            pass
        else:
            raise RuntimeError("Scratch result unexpectedly resolvable before input test")

        @delayed
        def add_one(x) -> float:
            return x + 1

        add_one.scratch = True
        add_one.allow_input_fingertip = False
        tf_fail = add_one(result_checksum)
        failed_checksum = tf_fail.compute()
        if failed_checksum is not None:
            raise RuntimeError(
                "Input checksum unexpectedly resolved without allow_input_fingertip"
            )
        if not tf_fail.exception or "CacheMiss" not in tf_fail.exception:
            raise RuntimeError(
                "Expected CacheMissError when allow_input_fingertip is False"
            )

        @delayed
        def add_one_finger(x) -> float:
            return x + 1

        add_one_finger.scratch = True
        add_one_finger.allow_input_fingertip = True
        tf_ok = add_one_finger(result_checksum)
        value2 = tf_ok.run()
        if abs(value2 - (expected_value + 1.0)) > 1e-6:
            raise RuntimeError(f"Unexpected fingertip input value: {value2!r}")
        asyncio.run(
            _undo_transformation(tf_ok.transformation_checksum, tf_ok.result_checksum)
        )

        get_buffer_cache().purge_scratch(result_checksum)
        try:
            result_checksum.resolve()
        except CacheMissError:
            pass
        else:
            raise RuntimeError("Scratch result unexpectedly resolvable before CLI test")

        @delayed
        def add_one_cli(x) -> float:
            return x + 1

        add_one_cli.scratch = True
        tf_cli = add_one_cli(result_checksum)
        tf_cli_checksum = tf_cli.construct()
        if tf_cli_checksum is None:
            raise RuntimeError(
                tf_cli.exception or "CLI transformation checksum unavailable"
            )

        proc = _run_cli(tf_cli_checksum, fingertip=False, workdir=workdir)
        if proc.returncode == 0:
            raise RuntimeError("CLI succeeded without --fingertip")
        err_output = "\n".join(
            line
            for line in (proc.stdout or "").splitlines()
            + (proc.stderr or "").splitlines()
            if line
        )
        if "CacheMiss" not in err_output:
            raise RuntimeError(
                "Expected CacheMissError when CLI runs without --fingertip"
            )

        proc = _run_cli(tf_cli_checksum, fingertip=True, workdir=workdir)
        if proc.returncode != 0:
            output = "\n".join(
                line
                for line in (proc.stdout or "").splitlines()
                + (proc.stderr or "").splitlines()
                if line
            )
            raise RuntimeError(f"CLI failed with --fingertip:\n{output}")
        output_lines = [
            line.strip()
            for line in (proc.stdout or "").splitlines()
            if line.strip()
        ]
        if not output_lines:
            raise RuntimeError("CLI produced no checksum output")
        cli_result_checksum = Checksum(output_lines[-1])
        expected_cli_checksum = Buffer(expected_value + 1.0, "mixed").get_checksum()
        if cli_result_checksum != expected_cli_checksum:
            raise RuntimeError(
                f"CLI checksum mismatch: {cli_result_checksum.hex()} != {expected_cli_checksum.hex()}"
            )
        asyncio.run(_undo_transformation(tf_cli_checksum, cli_result_checksum))

        get_buffer_cache().purge_scratch(result_checksum)
        try:
            result_checksum.resolve()
        except CacheMissError:
            pass
        else:
            raise RuntimeError(
                "Scratch result unexpectedly resolvable after recompute purge"
            )
    finally:
        try:
            seamless.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
