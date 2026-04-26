import pytest

from seamless import Buffer, Checksum

import seamless_transformer.probe_index as probe_index
import seamless_transformer.run as run_module


class _FakeDatabaseRemote:
    def __init__(self, probes):
        self.probes = probes

    async def get_bucket_probe(self, bucket_kind, label):
        return self.probes.get((bucket_kind, label))

    def has_read_server(self):
        return True


def _env_checksum(value):
    buf = Buffer(value, "plain")
    checksum = buf.get_checksum()
    if len(buf):
        buf.tempref()
    return checksum.hex()


def _make_process_probes(plan):
    node_label = plan["labels"]["node"]
    env_label = plan["labels"]["environment"]
    node_probe = {
        "bucket_kind": "node",
        "label": node_label,
        "bucket_checksum": "1" * 64,
        "captured_at": "2026-04-26T12:00:00Z",
        "freshness_tokens": plan["live_tokens"]["node"],
    }
    env_probe = {
        "bucket_kind": "environment",
        "label": env_label,
        "bucket_checksum": "2" * 64,
        "captured_at": "2026-04-26T12:00:00Z",
        "freshness_tokens": plan["live_tokens"]["environment"],
    }
    node_env_label = f"{node_probe['bucket_checksum']}:{env_probe['bucket_checksum']}"
    node_env_probe = {
        "bucket_kind": "node_env",
        "label": node_env_label,
        "bucket_checksum": "3" * 64,
        "captured_at": "2026-04-26T12:00:00Z",
        "freshness_tokens": {
            "node": plan["live_tokens"]["node"],
            "environment": plan["live_tokens"]["environment"],
        },
    }
    return {
        ("node", node_label): node_probe,
        ("environment", env_label): env_probe,
        ("node_env", node_env_label): node_env_probe,
    }


def test_ensure_record_bucket_preconditions_process_success(monkeypatch):
    env_checksum = _env_checksum({"conda_environment": "seamless1"})
    transformation_dict = {
        "__language__": "python",
        "__output__": ("result", "mixed", None),
        "__env__": env_checksum,
    }
    monkeypatch.setattr(probe_index, "get_record", lambda: True)
    monkeypatch.setattr(probe_index, "get_execution", lambda: "process")
    plan = probe_index.resolve_probe_plan(transformation_dict, {})
    monkeypatch.setattr(
        probe_index, "database_remote", _FakeDatabaseRemote(_make_process_probes(plan))
    )

    result = probe_index.ensure_record_bucket_preconditions_sync(transformation_dict, {})

    assert result["required_bucket_checksums"] == {
        "node": "1" * 64,
        "environment": "2" * 64,
        "node_env": "3" * 64,
    }
    assert result["required_bucket_labels"]["node"] == plan["labels"]["node"]
    assert result["required_bucket_labels"]["environment"] == plan["labels"][
        "environment"
    ]


def test_ensure_record_bucket_preconditions_process_missing_bucket(monkeypatch):
    transformation_dict = {
        "__language__": "python",
        "__output__": ("result", "mixed", None),
    }
    monkeypatch.setattr(probe_index, "get_record", lambda: True)
    monkeypatch.setattr(probe_index, "get_execution", lambda: "process")
    plan = probe_index.resolve_probe_plan(transformation_dict, {})
    probes = _make_process_probes(plan)
    probes.pop(("node_env", "1" * 64 + ":" + "2" * 64))
    monkeypatch.setattr(probe_index, "database_remote", _FakeDatabaseRemote(probes))

    with pytest.raises(
        probe_index.RecordBucketError, match="Record mode requires bucket probes"
    ):
        probe_index.ensure_record_bucket_preconditions_sync(transformation_dict, {})


def test_ensure_record_bucket_preconditions_process_stale_bucket(monkeypatch):
    transformation_dict = {
        "__language__": "python",
        "__output__": ("result", "mixed", None),
    }
    monkeypatch.setattr(probe_index, "get_record", lambda: True)
    monkeypatch.setattr(probe_index, "get_execution", lambda: "process")
    plan = probe_index.resolve_probe_plan(transformation_dict, {})
    probes = _make_process_probes(plan)
    node_key = ("node", plan["labels"]["node"])
    probes[node_key] = {
        **probes[node_key],
        "freshness_tokens": {"hostname": "stale-host"},
    }
    monkeypatch.setattr(probe_index, "database_remote", _FakeDatabaseRemote(probes))

    with pytest.raises(
        probe_index.RecordBucketError, match="Record mode detected stale bucket probes"
    ):
        probe_index.ensure_record_bucket_preconditions_sync(transformation_dict, {})


def test_run_transformation_dict_checks_bucket_preconditions(monkeypatch):
    monkeypatch.setattr(
        run_module,
        "ensure_record_bucket_preconditions_sync",
        lambda transformation, tf_dunder: (_ for _ in ()).throw(
            probe_index.RecordBucketError("record bucket failure")
        ),
    )

    with pytest.raises(probe_index.RecordBucketError, match="record bucket failure"):
        run_module.run_transformation_dict(
            {"__language__": "python", "__output__": ("result", "mixed", None)},
            Checksum("1" * 64),
            {},
        )
