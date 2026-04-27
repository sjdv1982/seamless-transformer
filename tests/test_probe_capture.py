import asyncio

from seamless import Buffer

import seamless_transformer.probe_capture as probe_capture


class _FakeBufferRemote:
    def has_write_server(self):
        return True


class _FakeDatabaseRemote:
    def __init__(self, probes):
        self.probes = dict(probes)
        self.set_calls = []

    async def get_bucket_probe(self, bucket_kind, label):
        return self.probes.get((bucket_kind, label))

    async def set_bucket_probe(
        self, bucket_kind, label, bucket_checksum, freshness_tokens, captured_at
    ):
        probe = {
            "bucket_kind": bucket_kind,
            "label": label,
            "bucket_checksum": str(bucket_checksum),
            "freshness_tokens": freshness_tokens,
            "captured_at": captured_at,
        }
        self.probes[(bucket_kind, label)] = probe
        self.set_calls.append(probe)
        return True

    def has_write_server(self):
        return True

    def has_read_server(self):
        return True


def _checksum(value):
    buf = Buffer(value, "plain")
    checksum = buf.get_checksum()
    if len(buf):
        buf.tempref()
    return checksum.hex()


async def _fake_buffer_write(self):
    del self
    return True


def test_refresh_required_buckets_updates_stale_base_and_composite(monkeypatch):
    plan = {
        "required_kinds": ["node", "environment", "node_env"],
        "labels": {
            "node": "worker-1",
            "environment": "conda:/envs/seamless1",
        },
        "live_tokens": {
            "node": {"hostname": "worker-1", "boot_id": "boot-1"},
            "environment": {"sys_prefix": "/envs/seamless1"},
        },
        "hostname": "worker-1",
        "cluster": None,
        "remote_target": None,
    }
    node_probe = {
        "bucket_kind": "node",
        "label": "worker-1",
        "bucket_checksum": "a" * 64,
        "freshness_tokens": plan["live_tokens"]["node"],
        "captured_at": "2026-04-26T12:00:00Z",
    }
    stale_env_probe = {
        "bucket_kind": "environment",
        "label": "conda:/envs/seamless1",
        "bucket_checksum": "b" * 64,
        "freshness_tokens": {"sys_prefix": "/envs/old"},
        "captured_at": "2026-04-26T12:00:00Z",
    }
    fake_db = _FakeDatabaseRemote(
        {
            ("node", node_probe["label"]): node_probe,
            ("environment", stale_env_probe["label"]): stale_env_probe,
        }
    )

    def _fake_capture(_target_transformation, _target_tf_dunder, *, request):
        return {
            "bucket_kind": request["bucket_kind"],
            "label": request["label"],
            "node_checksum": request.get("node_checksum"),
            "environment_checksum": request.get("environment_checksum"),
        }

    monkeypatch.setattr(probe_capture, "buffer_remote", _FakeBufferRemote())
    monkeypatch.setattr(probe_capture, "database_remote", fake_db)
    monkeypatch.setattr(
        probe_capture, "discover_probe_plan_sync", lambda *args, **kwargs: plan
    )
    monkeypatch.setattr(
        probe_capture, "capture_probe_payload_sync", _fake_capture
    )
    monkeypatch.setattr(probe_capture.Buffer, "write", _fake_buffer_write)

    result = probe_capture.refresh_required_buckets_sync(
        {"__language__": "python", "__output__": ("result", "plain", None)},
        {},
    )

    env_payload = {
        "bucket_kind": "environment",
        "label": "conda:/envs/seamless1",
        "node_checksum": None,
        "environment_checksum": None,
    }
    env_checksum = _checksum(env_payload)
    node_env_payload = {
        "bucket_kind": "node_env",
        "label": f"{'a' * 64}:{env_checksum}",
        "node_checksum": "a" * 64,
        "environment_checksum": env_checksum,
    }
    node_env_checksum = _checksum(node_env_payload)

    assert result["required_bucket_checksums"] == {
        "node": "a" * 64,
        "environment": env_checksum,
        "node_env": node_env_checksum,
    }
    assert result["required_bucket_labels"]["node_env"] == f"{'a' * 64}:{env_checksum}"
    assert [item["bucket_kind"] for item in result["refreshed"]] == [
        "environment",
        "node_env",
    ]
    assert [item["bucket_kind"] for item in result["reused"]] == ["node"]

