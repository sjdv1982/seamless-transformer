import asyncio
import json
import sys
from types import ModuleType

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


def test_numpy_show_config_prefers_structured_mode(monkeypatch):
    fake_numpy = ModuleType("numpy")

    def _show_config(*, mode=None):
        assert mode == "dicts"
        return {"blas": {"name": "openblas"}}

    fake_numpy.show_config = _show_config
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)

    result = probe_capture._numpy_show_config()

    assert result == {"blas": {"name": "openblas"}}


def test_python_packages_includes_direct_url(monkeypatch, tmp_path):
    dist_info = tmp_path / "demo.dist-info"
    dist_info.mkdir()
    direct_url = {
        "url": "file:///workspace/demo",
        "dir_info": {"editable": True},
    }
    (dist_info / "direct_url.json").write_text(json.dumps(direct_url), encoding="utf-8")

    class _FakeMetadata(dict):
        def get(self, key, default=None):
            return super().get(key, default)

    class _FakeDistribution:
        metadata = _FakeMetadata({"Name": "demo"})
        version = "1.2.3"

        def locate_file(self, path):
            return dist_info / path

    monkeypatch.setattr(
        probe_capture.importlib.metadata,
        "distributions",
        lambda: [_FakeDistribution()],
    )

    result = probe_capture._python_packages()

    assert result == [
        {
            "name": "demo",
            "version": "1.2.3",
            "direct_url": direct_url,
        }
    ]


def test_cuda_toolkit_version_prefers_torch(monkeypatch):
    fake_torch = ModuleType("torch")
    fake_torch.version = ModuleType("torch.version")
    fake_torch.version.cuda = "12.4"
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    result = probe_capture._cuda_toolkit_version()

    assert result == "12.4"


def test_visible_gpu_mapping_uses_pynvml(monkeypatch):
    fake_pynvml = ModuleType("pynvml")
    state = {"init": 0, "shutdown": 0}

    def _init():
        state["init"] += 1

    def _shutdown():
        state["shutdown"] += 1

    fake_pynvml.nvmlInit = _init
    fake_pynvml.nvmlShutdown = _shutdown
    fake_pynvml.nvmlDeviceGetCount = lambda: 3
    fake_pynvml.nvmlDeviceGetHandleByIndex = lambda index: index
    fake_pynvml.nvmlDeviceGetUUID = lambda handle: f"GPU-{handle}"
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,0")

    result = probe_capture._visible_gpu_mapping()

    assert result == [
        {"visible_token": "2", "device_index": 2, "gpu_uuid": "GPU-2"},
        {"visible_token": "0", "device_index": 0, "gpu_uuid": "GPU-0"},
    ]
    assert state == {"init": 1, "shutdown": 1}


def test_node_gpu_inventory_uses_pynvml(monkeypatch):
    fake_pynvml = ModuleType("pynvml")
    state = {"init": 0, "shutdown": 0}

    def _init():
        state["init"] += 1

    def _shutdown():
        state["shutdown"] += 1

    fake_pynvml.nvmlInit = _init
    fake_pynvml.nvmlShutdown = _shutdown
    fake_pynvml.nvmlSystemGetDriverVersion = lambda: "550.54"
    fake_pynvml.nvmlDeviceGetCount = lambda: 1
    fake_pynvml.nvmlDeviceGetHandleByIndex = lambda index: index
    fake_pynvml.nvmlDeviceGetName = lambda handle: "NVIDIA A100"
    fake_pynvml.nvmlDeviceGetUUID = lambda handle: "GPU-ABC"
    fake_pynvml.nvmlDeviceGetMemoryInfo = lambda handle: type(
        "MemInfo", (), {"total": 80 * 1024 * 1024 * 1024}
    )()
    fake_pynvml.nvmlDeviceGetCudaComputeCapability = lambda handle: (8, 0)
    fake_pynvml.nvmlDeviceGetEccMode = lambda handle: (1, 1)
    fake_pynvml.nvmlDeviceGetPersistenceMode = lambda handle: 1
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)

    result = probe_capture._node_gpu_inventory()

    assert result == {
        "driver_version": "550.54",
        "gpus": [
            {
                "index": 0,
                "name": "NVIDIA A100",
                "uuid": "GPU-ABC",
                "memory_total_bytes": 80 * 1024 * 1024 * 1024,
                "compute_capability": "8.0",
                "ecc_mode": 1,
                "persistence_mode": 1,
            }
        ],
    }
    assert state == {"init": 1, "shutdown": 1}


def test_build_node_payload_includes_proc_and_sys_facts(monkeypatch):
    monkeypatch.setattr(
        probe_capture,
        "_cpuinfo_summary",
        lambda: {
            "model_name": "AMD EPYC",
            "microcode": "0x123",
            "flags": ["avx2", "fma"],
        },
    )
    monkeypatch.setattr(
        probe_capture,
        "_numa_topology",
        lambda: [{"node": "node0", "cpulist": "0-7"}],
    )
    monkeypatch.setattr(
        probe_capture,
        "_node_gpu_inventory",
        lambda: {"driver_version": "550.54", "gpus": []},
    )
    monkeypatch.setattr(
        probe_capture,
        "_os_release",
        lambda: {"ID": "ubuntu", "VERSION_ID": "24.04"},
    )
    monkeypatch.setattr(
        probe_capture,
        "_kernel_setting",
        lambda path: {
            "/sys/kernel/mm/transparent_hugepage/enabled": "always [madvise] never",
            "/proc/sys/kernel/randomize_va_space": "2",
            "/proc/sys/vm/overcommit_memory": "0",
        }.get(path),
    )

    payload = probe_capture._build_node_payload({"label": "worker-1"})

    assert payload["label"] == "worker-1"
    assert payload["cpu"]["model_name"] == "AMD EPYC"
    assert payload["cpu"]["microcode"] == "0x123"
    assert payload["cpu"]["flags"] == ["avx2", "fma"]
    assert payload["numa_topology"] == [{"node": "node0", "cpulist": "0-7"}]
    assert payload["gpu_inventory"] == {"driver_version": "550.54", "gpus": []}
    assert payload["distribution"] == {"ID": "ubuntu", "VERSION_ID": "24.04"}
    assert payload["transparent_hugepages"] == "always [madvise] never"
    assert payload["aslr"] == "2"
    assert payload["overcommit_memory"] == "0"
