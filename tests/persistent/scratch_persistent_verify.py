from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import seamless.config
import seamless_config as _config
from seamless import CacheMissError, Checksum
from seamless.caching.buffer_cache import get_buffer_cache
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


def main() -> None:
    try:
        workdir = Path(__file__).resolve().parent.parent
        seamless.config.set_workdir(workdir)
        seamless.config.set_stage("simple-remote")
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

    cache = get_transformation_cache()
    if cache.get_reverse_transformations(result_checksum):
        raise RuntimeError("In-process reverse cache must be empty for persistent test")

    get_buffer_cache().purge_scratch(result_checksum)
    try:
        result_checksum.resolve()
    except CacheMissError:
        pass
    else:
        raise RuntimeError("Scratch result unexpectedly resolvable after purge")

    asyncio.run(_assert_rev_present(result_checksum, tf_checksum))

    value = asyncio.run(result_checksum.fingertip("mixed"))
    if abs(value - 12.84) > 1e-6:
        raise RuntimeError(f"Unexpected scratch recompute value: {value!r}")

    get_buffer_cache().purge_scratch(result_checksum)
    try:
        result_checksum.resolve()
    except CacheMissError:
        pass
    else:
        raise RuntimeError("Scratch result unexpectedly resolvable after recompute purge")


if __name__ == "__main__":
    main()
