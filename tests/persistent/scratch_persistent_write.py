from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

import seamless
import seamless.config
import seamless_config as _config
from seamless import Checksum
from seamless.transformer import delayed

_SKIP_EXIT = 77


def _skip(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(_SKIP_EXIT)


async def _wait_for_rev(result_checksum: Checksum, tf_checksum: Checksum) -> bool:
    try:
        from seamless_remote import database_remote
    except Exception:
        return False
    for _ in range(30):
        try:
            rev = await database_remote.get_rev_transformations(result_checksum)
        except Exception:
            rev = None
        if rev and tf_checksum in rev:
            return True
        await asyncio.sleep(0.1)
    return False


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

        @delayed
        def func(a, b) -> float:
            return 2.13 * a + 2.86 * b

        func.scratch = True
        func.local = True

        seed = uuid.uuid4().int
        a = (seed % 10_000) / 37.0
        b = ((seed // 10_000) % 10_000) / 53.0
        expected_value = 2.13 * a + 2.86 * b

        tf = func(a, b)
        result_checksum = tf.compute()
        if not isinstance(result_checksum, Checksum):
            raise RuntimeError(tf.exception or "Scratch compute returned no checksum")

        tf_checksum = tf.transformation_checksum

        ok = asyncio.run(_wait_for_rev(result_checksum, tf_checksum))
        if not ok:
            raise RuntimeError(
                "Persistent reverse transformation cache was not written"
            )

        state_path = os.environ.get("SEAMLESS_SCRATCH_PERSISTENT_PATH")
        if not state_path:
            raise RuntimeError("SEAMLESS_SCRATCH_PERSISTENT_PATH is not set")
        payload = {
            "result_checksum": result_checksum.hex(),
            "tf_checksum": tf_checksum.hex(),
            "result_value": expected_value,
        }
        Path(state_path).write_text(json.dumps(payload), encoding="utf-8")
    finally:
        try:
            seamless.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
