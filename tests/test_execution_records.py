import asyncio

import pytest

from seamless import Buffer, Checksum

import seamless_transformer.transformation_cache as transformation_cache
from seamless_transformer.transformation_cache import TransformationCache


class _FakeDatabaseRemote:
    def __init__(self, *, write_server=True):
        self.write_server = write_server
        self.transformation_results = []
        self.execution_records = []

    async def get_transformation_result(self, tf_checksum):
        del tf_checksum
        return None

    async def set_transformation_result(self, tf_checksum, result_checksum):
        self.transformation_results.append((tf_checksum.hex(), result_checksum.hex()))
        return True

    async def set_execution_record(self, tf_checksum, result_checksum, record):
        self.execution_records.append(
            (tf_checksum.hex(), result_checksum.hex(), record)
        )
        return True

    def has_write_server(self):
        return self.write_server


class _ImmediateLoop:
    async def run_in_executor(self, executor, func, *args):
        del executor
        return func(*args)


def _make_checksum(value, celltype="plain") -> Checksum:
    buf = Buffer(value, celltype)
    checksum = buf.get_checksum()
    if len(buf):
        buf.tempref()
    return checksum


def test_record_mode_writes_execution_record(monkeypatch):
    cache = TransformationCache()
    fake_database_remote = _FakeDatabaseRemote()
    env_checksum = _make_checksum({"conda_environment": "seamless1"})
    result_checksum = _make_checksum(3, "mixed")
    tf_checksum = _make_checksum({"kind": "execution-record-test"})

    def _fake_run_transformation_dict(
        transformation_dict,
        tf_checksum_arg,
        tf_dunder,
        scratch,
        require_value,
    ):
        del transformation_dict, tf_checksum_arg, tf_dunder, scratch, require_value
        return result_checksum

    monkeypatch.setattr(transformation_cache, "database_remote", fake_database_remote)
    monkeypatch.setattr(transformation_cache, "buffer_remote", None)
    monkeypatch.setattr(transformation_cache, "_buffer_writer", None)
    monkeypatch.setattr(transformation_cache, "jobserver_remote", None)
    monkeypatch.setattr(
        transformation_cache.asyncio, "get_running_loop", lambda: _ImmediateLoop()
    )
    monkeypatch.setattr(transformation_cache, "get_execution", lambda: "process")
    monkeypatch.setattr(transformation_cache, "get_record", lambda: True)
    monkeypatch.setattr(transformation_cache, "get_remote", lambda: None)
    monkeypatch.setattr(transformation_cache, "get_selected_cluster", lambda: None)
    monkeypatch.setattr(transformation_cache, "get_queue", lambda cluster=None: None)
    monkeypatch.setattr(transformation_cache, "get_node", lambda: None)
    monkeypatch.setattr(
        transformation_cache, "run_transformation_dict", _fake_run_transformation_dict
    )

    result = asyncio.run(
        cache.run(
            {
                "__language__": "python",
                "__output__": ("result", "mixed", None),
            },
            tf_checksum=tf_checksum,
            tf_dunder={"__env__": env_checksum.hex()},
            scratch=False,
            require_value=True,
            force_local=True,
        )
    )

    assert result == result_checksum
    assert fake_database_remote.transformation_results == [
        (tf_checksum.hex(), result_checksum.hex())
    ]
    assert len(fake_database_remote.execution_records) == 1
    record = fake_database_remote.execution_records[0][2]
    assert record["schema_version"] == 1
    assert record["tf_checksum"] == tf_checksum.hex()
    assert record["result_checksum"] == result_checksum.hex()
    assert record["execution_mode"] == "process"
    assert record["environment"] == env_checksum.hex()
    assert record["execution_envelope"]["language_kind"] == "python"
    assert record["execution_envelope"]["scratch"] is False
    assert record["execution_envelope"]["resolved_env_checksum"] == env_checksum.hex()
    assert record["execution_envelope"]["active_tf_dunder_keys"] == ["__env__"]
    assert record["hostname"]
    assert isinstance(record["worker_execution_index"], int)


def test_record_mode_requires_database_write_server(monkeypatch):
    cache = TransformationCache()
    tf_checksum = _make_checksum({"kind": "execution-record-test"})

    monkeypatch.setattr(transformation_cache, "database_remote", None)
    monkeypatch.setattr(transformation_cache, "buffer_remote", None)
    monkeypatch.setattr(transformation_cache, "_buffer_writer", None)
    monkeypatch.setattr(transformation_cache, "jobserver_remote", None)
    monkeypatch.setattr(
        transformation_cache.asyncio, "get_running_loop", lambda: _ImmediateLoop()
    )
    monkeypatch.setattr(transformation_cache, "get_execution", lambda: "process")
    monkeypatch.setattr(transformation_cache, "get_record", lambda: True)

    with pytest.raises(
        RuntimeError, match="Record mode requires an active database write server"
    ):
        asyncio.run(
            cache.run(
                {
                    "__language__": "python",
                    "__output__": ("result", "mixed", None),
                },
                tf_checksum=tf_checksum,
                tf_dunder={},
                scratch=False,
                require_value=False,
                force_local=True,
            )
        )
