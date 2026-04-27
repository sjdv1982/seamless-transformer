"""Probe capture helpers for seamless-probe and execution-record buckets."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import ctypes
import ctypes.util
import importlib.metadata
import io
import json
import locale
import mmap
import os
import platform
import re
import resource
import shutil
import socket
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from seamless import Buffer, Checksum

from seamless_config.cluster import get_cluster
from seamless_transformer.pretransformation import PreTransformation
from seamless_transformer.probe_index import RECORD_PROBE_DUNDER, resolve_probe_plan
from seamless_transformer.transformation_class import transformation_from_pretransformation
from seamless_transformer.transformation_utils import resolve_env_checksum

try:
    from seamless_remote import buffer_remote, database_remote
except ImportError:  # pragma: no cover - optional dependency
    buffer_remote = None
    database_remote = None


PROBE_CODE = """
from seamless_transformer.probe_capture import execute_probe_request
result = execute_probe_request(probe_payload)
""".strip()


def _utcnow_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _stable_digest(value: Any) -> str:
    try:
        payload = json.dumps(value, sort_keys=True, default=repr).encode("utf-8")
    except TypeError:
        payload = repr(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _selected_env_vars(*prefixes: str, names: tuple[str, ...] = ()) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in os.environ.items():
        if key in names or any(key.startswith(prefix) for prefix in prefixes):
            result[key] = value
    return dict(sorted(result.items()))


def _split_path_like(value: str) -> list[str]:
    return [item for item in value.split(":") if item]


def _split_preload(value: str) -> list[str]:
    try:
        import shlex

        tokens = shlex.split(value)
    except Exception:
        tokens = value.split()
    items: list[str] = []
    for token in tokens:
        items.extend(part for part in token.split(":") if part)
    return items


def _normalize_path(path: str) -> str:
    return os.path.realpath(os.path.expanduser(os.path.expandvars(path)))


def _memory_total_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
    except Exception:
        return None
    try:
        return int(page_size) * int(phys_pages)
    except Exception:
        return None


def _physical_core_count() -> int | None:
    try:
        import psutil
    except Exception:
        return None
    try:
        count = psutil.cpu_count(logical=False)
    except Exception:
        return None
    if count is None or count <= 0:
        return None
    return int(count)


def _affinity_count() -> int | None:
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return None


def _read_text_file(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            value = f.read().strip()
    except Exception:
        return None
    return value or None


def _cpuinfo_summary() -> dict[str, Any] | None:
    text = _read_text_file("/proc/cpuinfo")
    if not text:
        return None
    blocks = [block for block in text.split("\n\n") if block.strip()]
    if not blocks:
        return None
    first: dict[str, str] = {}
    for line in blocks[0].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        first[key.strip()] = value.strip()
    flags = first.get("flags") or first.get("Features")
    return {
        "model_name": first.get("model name") or first.get("Processor"),
        "microcode": first.get("microcode"),
        "flags": flags.split() if isinstance(flags, str) and flags else None,
    }


def _numa_topology() -> list[dict[str, Any]] | None:
    sys_node = Path("/sys/devices/system/node")
    if not sys_node.exists():
        return None
    result = []
    for node_dir in sorted(sys_node.glob("node[0-9]*")):
        cpulist = _read_text_file(str(node_dir / "cpulist"))
        if cpulist is None:
            continue
        result.append({"node": node_dir.name, "cpulist": cpulist})
    return result or None


def _os_release() -> dict[str, str] | None:
    text = _read_text_file("/etc/os-release")
    if not text:
        return None
    result: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key] = value.strip().strip('"')
    return result or None


def _system_library_roots() -> tuple[str, ...]:
    roots = {
        "/lib",
        "/lib64",
        "/usr/lib",
        "/usr/lib64",
        "/lib/x86_64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu",
    }
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        roots.add(os.path.join(conda_prefix, "lib"))
    return tuple(sorted(roots))


def _path_allowed_for_contract(path: str, *, conda_prefix: str | None) -> bool:
    if not path:
        return True
    if conda_prefix and path.startswith(conda_prefix + os.sep):
        return True
    return False


def _allowlisted_library_name(name: str) -> bool:
    return name.startswith(
        (
            "ld-linux",
            "libc.so",
            "libm.so",
            "libdl.so",
            "libpthread.so",
            "librt.so",
            "libutil.so",
            "libcuda.so",
            "libnvidia-ml.so",
        )
    )


def _resolve_library_path(name: str) -> str | None:
    if not name:
        return None
    if os.path.isabs(name):
        path = os.path.realpath(name)
        return path if os.path.exists(path) else None
    for root in _system_library_roots():
        candidate = os.path.realpath(os.path.join(root, name))
        if os.path.exists(candidate):
            return candidate
    return None


def _library_identity(short_name: str) -> dict[str, Any] | None:
    library_name = ctypes.util.find_library(short_name)
    if not library_name:
        return None
    path = _resolve_library_path(library_name)
    return {
        "name": library_name,
        "path": path,
        "basename": os.path.basename(path) if path else library_name,
    }


def _kernel_setting(path: str) -> str | None:
    return _read_text_file(path)


def _cgroup_memory_limit_bytes() -> int | None:
    candidates = (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    )
    for path in candidates:
        try:
            value = open(path, "r", encoding="utf-8").read().strip()
        except Exception:
            continue
        if not value or value == "max":
            return None
        try:
            limit = int(value)
        except Exception:
            continue
        if limit <= 0 or limit >= 2**60:
            return None
        return limit
    return None


def _resource_limits() -> dict[str, list[int | str]]:
    limits: dict[str, list[int | str]] = {}
    for name in (
        "RLIMIT_CPU",
        "RLIMIT_DATA",
        "RLIMIT_STACK",
        "RLIMIT_RSS",
        "RLIMIT_NOFILE",
        "RLIMIT_AS",
    ):
        if not hasattr(resource, name):
            continue
        try:
            soft, hard = resource.getrlimit(getattr(resource, name))
        except Exception:
            continue
        limits[name] = [
            "unlimited" if value == resource.RLIM_INFINITY else int(value)
            for value in (soft, hard)
        ]
    return limits


def _compiler_version(command: str) -> str | None:
    try:
        completed = subprocess.run(
            [command, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    output = (completed.stdout or completed.stderr or "").strip().splitlines()
    if not output:
        return None
    return output[0].strip() or None


def _compiler_details(command_text: str | None) -> dict[str, Any] | None:
    if not isinstance(command_text, str) or not command_text.strip():
        return None
    try:
        import shlex

        argv = shlex.split(command_text)
    except Exception:
        argv = command_text.split()
    if not argv:
        return None
    binary = argv[0]
    path = shutil.which(binary)
    return {
        "command": command_text,
        "binary": binary,
        "path": path,
        "version": _compiler_version(path or binary),
    }


def _compiler_inventory() -> dict[str, Any]:
    selected = {
        name: _compiler_details(os.environ.get(name))
        for name in ("CC", "CXX", "FC")
        if os.environ.get(name)
    }
    defaults: dict[str, Any] = {}
    for name, candidates in (
        ("c", ("cc", "gcc", "clang")),
        ("cxx", ("c++", "g++", "clang++")),
        ("fortran", ("gfortran", "flang", "ifort", "ifx")),
    ):
        for candidate in candidates:
            if shutil.which(candidate):
                defaults[name] = _compiler_details(candidate)
                break
    return {
        "selected": selected,
        "defaults": defaults,
    }


def _conda_env_export() -> str | None:
    conda_exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda_exe or not os.environ.get("CONDA_PREFIX"):
        return None
    try:
        completed = subprocess.run(
            [conda_exe, "env", "export", "--no-builds"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    result = completed.stdout.strip()
    return result or None


def _python_packages() -> list[dict[str, Any]]:
    packages: list[dict[str, Any]] = []
    try:
        distributions = importlib.metadata.distributions()
    except Exception:
        return packages
    for dist in distributions:
        name = dist.metadata.get("Name") or dist.metadata.get("Summary")
        if not name:
            continue
        payload: dict[str, Any] = {"name": str(name), "version": str(dist.version)}
        try:
            direct_url_path = dist.locate_file("direct_url.json")
        except Exception:
            direct_url_path = None
        if direct_url_path is not None:
            try:
                if os.path.exists(direct_url_path):
                    with open(direct_url_path, "r", encoding="utf-8") as f:
                        payload["direct_url"] = _json_safe(json.load(f))
            except Exception:
                pass
        packages.append(payload)
    packages.sort(key=lambda item: item["name"].lower())
    return packages


def _numpy_show_config() -> Any | None:
    try:
        import numpy as np
    except Exception:
        return None
    try:
        result = np.show_config(mode="dicts")
    except TypeError:
        result = None
    except Exception:
        return None
    if result is not None:
        return _json_safe(result)
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        try:
            np.show_config()
        except Exception:
            return None
    result = stream.getvalue().strip()
    return result or None


def _threadpool_info() -> list[dict[str, Any]] | None:
    try:
        from threadpoolctl import threadpool_info
    except Exception:
        return None
    try:
        return _json_safe(threadpool_info())
    except Exception:
        return None


def _cudnn_version() -> int | None:
    try:
        import torch
    except Exception:
        return None
    try:
        version = torch.backends.cudnn.version()
    except Exception:
        return None
    if version is None:
        return None
    try:
        return int(version)
    except Exception:
        return None


def _cuda_toolkit_version() -> str | None:
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None:
        try:
            version = getattr(getattr(torch, "version", None), "cuda", None)
        except Exception:
            version = None
        if isinstance(version, str) and version:
            return version

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        version_txt = os.path.join(cuda_home, "version.txt")
        try:
            with open(version_txt, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except Exception:
            content = None
        if content:
            match = re.search(r"(\d+\.\d+(?:\.\d+)?)", content)
            if match is not None:
                return match.group(1)
            return content

    try:
        completed = subprocess.run(
            ["nvcc", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    output = "\n".join(
        part for part in (completed.stdout, completed.stderr) if part
    ).strip()
    if not output:
        return None
    match = re.search(r"release (\d+\.\d+(?:\.\d+)?)", output)
    if match is not None:
        return match.group(1)
    match = re.search(r"V(\d+\.\d+(?:\.\d+)?)", output)
    if match is not None:
        return match.group(1)
    return None


def _visible_gpu_mapping() -> list[dict[str, Any]] | None:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    tokens = [item.strip() for item in raw.split(",") if item.strip()]
    if not tokens:
        return []
    try:
        import pynvml
    except Exception:
        return None
    try:
        pynvml.nvmlInit()
        device_count = int(pynvml.nvmlDeviceGetCount())
        uuids = []
        for index in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode("utf-8", errors="replace")
            uuids.append(str(uuid))
    except Exception:
        return None
    finally:
        shutdown = getattr(pynvml, "nvmlShutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass

    mapping: list[dict[str, Any]] = []
    for token in tokens:
        entry: dict[str, Any] = {"visible_token": token}
        try:
            index = int(token)
        except Exception:
            index = None
        if index is not None:
            entry["device_index"] = index
            if 0 <= index < len(uuids):
                entry["gpu_uuid"] = uuids[index]
        else:
            for uuid in uuids:
                if token == uuid or uuid.endswith(token):
                    entry["gpu_uuid"] = uuid
                    break
        mapping.append(entry)
    return mapping


def _node_gpu_inventory() -> dict[str, Any] | None:
    try:
        import pynvml
    except Exception:
        return None
    try:
        pynvml.nvmlInit()
        driver_version = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver_version, bytes):
            driver_version = driver_version.decode("utf-8", errors="replace")
        gpus = []
        device_count = int(pynvml.nvmlDeviceGetCount())
        for index in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            if isinstance(uuid, bytes):
                uuid = uuid.decode("utf-8", errors="replace")
            memory_info = None
            try:
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            except Exception:
                memory_info = None
            compute_capability = None
            get_compute_capability = getattr(
                pynvml, "nvmlDeviceGetCudaComputeCapability", None
            )
            if callable(get_compute_capability):
                try:
                    major, minor = get_compute_capability(handle)
                    compute_capability = f"{major}.{minor}"
                except Exception:
                    compute_capability = None
            ecc_mode = None
            get_ecc_mode = getattr(pynvml, "nvmlDeviceGetEccMode", None)
            if callable(get_ecc_mode):
                try:
                    current, _pending = get_ecc_mode(handle)
                    ecc_mode = current
                except Exception:
                    ecc_mode = None
            persistence_mode = None
            get_persistence_mode = getattr(
                pynvml, "nvmlDeviceGetPersistenceMode", None
            )
            if callable(get_persistence_mode):
                try:
                    persistence_mode = get_persistence_mode(handle)
                except Exception:
                    persistence_mode = None
            gpus.append(
                {
                    "index": index,
                    "name": str(name),
                    "uuid": str(uuid),
                    "memory_total_bytes": (
                        int(memory_info.total) if memory_info is not None else None
                    ),
                    "compute_capability": compute_capability,
                    "ecc_mode": ecc_mode,
                    "persistence_mode": persistence_mode,
                }
            )
    except Exception:
        return None
    finally:
        shutdown = getattr(pynvml, "nvmlShutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass
    return {
        "driver_version": str(driver_version),
        "gpus": gpus,
    }


def _parse_mountinfo() -> list[dict[str, Any]]:
    text = _read_text_file("/proc/self/mountinfo")
    if not text:
        return []
    result = []
    for line in text.splitlines():
        if " - " not in line:
            continue
        left, right = line.split(" - ", 1)
        left_fields = left.split()
        right_fields = right.split()
        if len(left_fields) < 6 or len(right_fields) < 3:
            continue
        result.append(
            {
                "mount_point": left_fields[4],
                "root": left_fields[3],
                "mount_options": left_fields[5],
                "optional_fields": left_fields[6:],
                "filesystem_type": right_fields[0],
                "mount_source": right_fields[1],
                "super_options": right_fields[2:],
            }
        )
    return result


def _mount_for_path(path: str, mountinfo: list[dict[str, Any]]) -> dict[str, Any] | None:
    best = None
    best_len = -1
    for entry in mountinfo:
        mount_point = entry.get("mount_point")
        if not isinstance(mount_point, str):
            continue
        if path == mount_point or path.startswith(mount_point.rstrip("/") + "/"):
            if len(mount_point) > best_len:
                best = entry
                best_len = len(mount_point)
    return best


def _filesystem_facts() -> dict[str, Any] | None:
    mountinfo = _parse_mountinfo()
    if not mountinfo:
        return None
    targets = {
        "cwd": os.getcwd(),
        "tempdir": tempfile.gettempdir(),
        "sys_prefix": sys.prefix,
    }
    purelib = sysconfig.get_path("purelib")
    if purelib:
        targets["purelib"] = purelib
    result: dict[str, Any] = {}
    for name, path in targets.items():
        resolved = os.path.realpath(path)
        entry = _mount_for_path(resolved, mountinfo)
        if entry is None:
            continue
        result[name] = {
            "path": resolved,
            "mount_point": entry["mount_point"],
            "filesystem_type": entry["filesystem_type"],
            "mount_source": entry["mount_source"],
        }
    return result or None


def _container_identity() -> dict[str, Any] | None:
    markers = []
    for marker in ("/.dockerenv", "/run/.containerenv"):
        if os.path.exists(marker):
            markers.append(marker)
    container_env = {
        name: os.environ[name]
        for name in ("SINGULARITY_CONTAINER", "APPTAINER_CONTAINER")
        if os.environ.get(name)
    }
    mountinfo = _parse_mountinfo()
    root_mount = _mount_for_path("/", mountinfo)
    root_fstype = None if root_mount is None else root_mount.get("filesystem_type")
    root_source = "" if root_mount is None else str(root_mount.get("mount_source") or "")
    root_looks_containerized = root_fstype in ("overlay", "fuse-overlayfs", "squashfs") or (
        "containers" in root_source or "overlay" in root_source
    )
    if not markers and not container_env and not root_looks_containerized:
        return None
    identity: dict[str, Any] = {
        "markers": markers,
        "environment": container_env or None,
    }
    if root_mount is not None:
        mount_source = root_mount.get("mount_source")
        super_options = list(root_mount.get("super_options") or [])
        root_payload = {
            "filesystem_type": root_mount.get("filesystem_type"),
            "mount_source": mount_source,
            "mount_point": root_mount.get("mount_point"),
        }
        layer_ids = sorted(set(re.findall(r"[0-9a-f]{64}", " ".join(super_options))))
        if layer_ids:
            root_payload["overlay_layer_ids"] = layer_ids
        identity["root_mount"] = root_payload
    return identity


def _docker_image_digest(env_spec: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(env_spec, dict):
        return None
    docker = env_spec.get("docker")
    if not isinstance(docker, dict):
        return None
    image_name = docker.get("name")
    if not isinstance(image_name, str) or not image_name:
        return None
    payload: dict[str, Any] = {"name": image_name}
    for client in ("docker", "podman"):
        binary = shutil.which(client)
        if not binary:
            continue
        try:
            completed = subprocess.run(
                [binary, "inspect", "--format={{.Id}}", image_name],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            continue
        digest = completed.stdout.strip()
        if completed.returncode == 0 and digest:
            payload["digest"] = digest
            payload["client"] = client
            break
    return payload


def _resolve_env_spec(env_checksum: str | None) -> dict[str, Any]:
    if not env_checksum:
        return {}
    try:
        env_buffer = Checksum(env_checksum).resolve()
    except Exception:
        return {}
    if env_buffer is None:
        return {}
    try:
        env_spec = env_buffer.get_value("plain")
    except Exception:
        return {}
    if not isinstance(env_spec, dict):
        return {}
    return env_spec


def _mxcsr_state() -> dict[str, Any] | None:
    if platform.machine().lower() not in ("x86_64", "amd64"):
        return None
    code = b"\x0f\xae\x1f\xc3"  # stmxcsr [rdi]; ret
    try:
        buf = mmap.mmap(
            -1,
            len(code),
            prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
        )
    except Exception:
        return None
    try:
        buf.write(code)
        storage = ctypes.c_uint32()
        func_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_uint32))
        func = func_type(ctypes.addressof(ctypes.c_char.from_buffer(buf)))
        func(ctypes.byref(storage))
        raw = int(storage.value)
    except Exception:
        return None
    finally:
        buf.close()
    return {
        "raw": raw,
        "ftz": bool(raw & (1 << 15)),
        "daz": bool(raw & (1 << 6)),
    }


def _gpu_determinism_env() -> dict[str, str]:
    return _selected_env_vars(
        names=(
            "CUBLAS_WORKSPACE_CONFIG",
            "TF_DETERMINISTIC_OPS",
            "TF_CUDNN_DETERMINISTIC",
            "PYTORCH_CUDA_ALLOC_CONF",
            "CUDA_DEVICE_ORDER",
            "CUDA_LAUNCH_BLOCKING",
            "NVIDIA_TF32_OVERRIDE",
        )
    )


def _determinant_environment() -> dict[str, str]:
    return _selected_env_vars(
        "OMP_",
        "GOMP_",
        "KMP_",
        "MKL_",
        "OPENBLAS_",
        names=(
            "PATH",
            "PYTHONPATH",
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "PYTHONHASHSEED",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "TZ",
        ),
    ) | _gpu_determinism_env()


def _environment_validation_views() -> dict[str, Any]:
    determinant_env = _determinant_environment()
    path_entries = [_normalize_path(item) for item in _split_path_like(os.environ.get("PATH", ""))]
    sys_path_entries = [
        _normalize_path(item) for item in sys.path if isinstance(item, str) and item
    ]
    ld_library_path = [
        _normalize_path(item)
        for item in _split_path_like(os.environ.get("LD_LIBRARY_PATH", ""))
    ]
    ld_preload = [
        _normalize_path(item)
        for item in _split_preload(os.environ.get("LD_PRELOAD", ""))
    ]
    return {
        "determinant_env": determinant_env,
        "determinant_env_hash": _stable_digest(determinant_env),
        "path_entries": path_entries,
        "path_hash": _stable_digest(path_entries),
        "sys_path_entries": sys_path_entries,
        "sys_path_hash": _stable_digest(sys_path_entries),
        "ld_library_path_entries": ld_library_path,
        "ld_preload_entries": ld_preload,
    }


def _queue_runtime_env() -> dict[str, str]:
    return _selected_env_vars(
        "GOMP_",
        "KMP_",
        names=(
            "OMP_NUM_THREADS",
            "OMP_SCHEDULE",
            "OMP_PROC_BIND",
            "OMP_PLACES",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "CUBLAS_WORKSPACE_CONFIG",
            "TF_DETERMINISTIC_OPS",
            "TF_CUDNN_DETERMINISTIC",
            "PYTORCH_CUDA_ALLOC_CONF",
            "CUDA_DEVICE_ORDER",
            "TMPDIR",
            "TEMP",
            "TMP",
        ),
    )


def _allocation_counts() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "logical_cores": os.cpu_count(),
        "affinity_cores": _affinity_count(),
    }
    for name in (
        "SLURM_CPUS_PER_TASK",
        "SLURM_CPUS_ON_NODE",
        "SLURM_JOB_NUM_NODES",
        "SLURM_NTASKS",
        "PBS_NP",
        "NCPUS",
        "NSLOTS",
    ):
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            payload[name.lower()] = int(value)
        except Exception:
            payload[name.lower()] = value
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        payload["visible_gpu_count"] = len(
            [item for item in cuda_visible_devices.split(",") if item]
        )
    return payload


def _write_snapshot_checksum_sync(value: Any) -> str | None:
    buffer = Buffer(value, "plain")

    async def _write() -> bool:
        return await buffer.write()

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        try:
            written = asyncio.run(_write())
        except Exception:
            return None
    else:
        result: dict[str, bool] = {}
        error: dict[str, BaseException] = {}

        def _runner():
            try:
                result["written"] = asyncio.run(_write())
            except BaseException as exc:
                error["exc"] = exc

        thread = threading.Thread(target=_runner, daemon=True, name="probe-snapshot")
        thread.start()
        thread.join()
        if error:
            return None
        written = result.get("written", False)
    if not written:
        return None
    return buffer.get_checksum().hex()


def _apply_bucket_contract_summary(
    payload: dict[str, Any],
    *,
    contract_violations: list[str],
    validation_details: dict[str, Any] | None,
) -> dict[str, Any]:
    payload["contract_violations"] = sorted(
        {code for code in contract_violations if isinstance(code, str) and code}
    )
    payload["contract_ok"] = not payload["contract_violations"]
    payload["validation_snapshot"] = None
    if validation_details:
        payload["validation_snapshot"] = _write_snapshot_checksum_sync(validation_details)
    return payload


def _environment_contract_summary(
    validation_views: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_prefix = _normalize_path(conda_prefix)
    violations: set[str] = set()
    for entry in validation_views.get("ld_library_path_entries", []):
        if not _path_allowed_for_contract(entry, conda_prefix=conda_prefix):
            violations.add("ld_library_path_outside_conda_prefix")
            break
    for entry in validation_views.get("ld_preload_entries", []):
        if _allowlisted_library_name(os.path.basename(entry)):
            continue
        if not _path_allowed_for_contract(entry, conda_prefix=conda_prefix):
            violations.add("ld_preload_outside_conda_prefix")
            break
    details = {
        "schema_version": 1,
        "bucket_kind": "environment",
        "conda_prefix": conda_prefix,
        "validation_views": validation_views,
        "violations": sorted(violations),
    }
    return sorted(violations), details


def _queue_node_contract_summary() -> tuple[list[str], dict[str, Any]]:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_prefix = _normalize_path(conda_prefix)
    ld_library_path_entries = [
        _normalize_path(item)
        for item in _split_path_like(os.environ.get("LD_LIBRARY_PATH", ""))
    ]
    ld_preload_entries = [
        _normalize_path(item)
        for item in _split_preload(os.environ.get("LD_PRELOAD", ""))
    ]
    violations: set[str] = set()
    for entry in ld_library_path_entries:
        if not _path_allowed_for_contract(entry, conda_prefix=conda_prefix):
            violations.add("ld_library_path_outside_conda_prefix")
            break
    for entry in ld_preload_entries:
        if _allowlisted_library_name(os.path.basename(entry)):
            continue
        if not _path_allowed_for_contract(entry, conda_prefix=conda_prefix):
            violations.add("ld_preload_outside_conda_prefix")
            break
    details = {
        "schema_version": 1,
        "bucket_kind": "queue_node",
        "conda_prefix": conda_prefix,
        "runtime_environment": _queue_runtime_env(),
        "ld_library_path_entries": ld_library_path_entries,
        "ld_preload_entries": ld_preload_entries,
        "violations": sorted(violations),
    }
    return sorted(violations), details


def _common_payload(bucket_kind: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "bucket_kind": bucket_kind,
        "contract_ok": True,
        "contract_violations": [],
        "validation_snapshot": None,
    }


def _build_node_payload(request: dict[str, Any]) -> dict[str, Any]:
    hostname = socket.gethostname()
    uname = platform.uname()
    cpuinfo = _cpuinfo_summary()
    payload = _common_payload("node")
    payload.update(
        {
            "label": request.get("label"),
            "hostname": hostname,
            "platform": {
                "system": uname.system,
                "node": uname.node,
                "release": uname.release,
                "version": uname.version,
                "machine": uname.machine,
                "processor": uname.processor,
            },
            "os": {
                "python_build": list(platform.python_build()),
                "python_compiler": platform.python_compiler(),
                "libc": list(platform.libc_ver()),
                "byteorder": sys.byteorder,
            },
            "cpu": {
                "model_name": None if cpuinfo is None else cpuinfo.get("model_name"),
                "microcode": None if cpuinfo is None else cpuinfo.get("microcode"),
                "flags": None if cpuinfo is None else cpuinfo.get("flags"),
                "physical_cores": _physical_core_count(),
                "logical_cores": os.cpu_count(),
                "affinity_cores": _affinity_count(),
            },
            "memory_total_bytes": _memory_total_bytes(),
            "numa_topology": _numa_topology(),
            "gpu_inventory": _node_gpu_inventory(),
            "distribution": _os_release(),
            "container": _container_identity(),
            "filesystems": _filesystem_facts(),
            "transparent_hugepages": _kernel_setting(
                "/sys/kernel/mm/transparent_hugepage/enabled"
            ),
            "aslr": _kernel_setting("/proc/sys/kernel/randomize_va_space"),
            "overcommit_memory": _kernel_setting("/proc/sys/vm/overcommit_memory"),
            "libraries": {
                "glibc": _library_identity("c"),
                "libm": _library_identity("m"),
            },
        }
    )
    return _apply_bucket_contract_summary(
        payload,
        contract_violations=[],
        validation_details=None,
    )


def _build_environment_payload(request: dict[str, Any]) -> dict[str, Any]:
    env_spec = _resolve_env_spec(request.get("env_checksum"))
    validation_views = _environment_validation_views()
    payload = _common_payload("environment")
    payload.update(
        {
            "label": request.get("label"),
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "executable": sys.executable,
                "prefix": sys.prefix,
                "path": list(sys.path),
            },
            "compiler_environment": {
                name: os.environ[name]
                for name in ("CC", "CXX", "FC")
                if os.environ.get(name) is not None
            },
            "compiler_inventory": _compiler_inventory(),
            "conda_env_export": _conda_env_export(),
            "locale": {
                "preferred_encoding": locale.getpreferredencoding(False),
                "default_locale": list(locale.getlocale()),
            },
            "timezone": list(time.tzname),
            "environment_variables": _selected_env_vars(
                "OMP_",
                "GOMP_",
                "KMP_",
                "MKL_",
                "OPENBLAS_",
                names=(
                    "PATH",
                    "PYTHONPATH",
                    "LD_LIBRARY_PATH",
                    "LD_PRELOAD",
                    "PYTHONHASHSEED",
                    "CONDA_DEFAULT_ENV",
                    "CONDA_PREFIX",
                    "LANG",
                    "LC_ALL",
                    "LC_CTYPE",
                    "TZ",
                ),
            ),
            "gpu_determinism_environment_variables": _gpu_determinism_env(),
            "docker_image": _docker_image_digest(env_spec),
            "python_packages": _python_packages(),
            "validation_views": validation_views,
        }
    )
    contract_violations, validation_details = _environment_contract_summary(
        validation_views
    )
    return _apply_bucket_contract_summary(
        payload,
        contract_violations=contract_violations,
        validation_details=validation_details,
    )


def _build_node_env_payload(request: dict[str, Any]) -> dict[str, Any]:
    payload = _common_payload("node_env")
    payload.update(
        {
            "label": request.get("label"),
            "node_checksum": request.get("node_checksum"),
            "environment_checksum": request.get("environment_checksum"),
            "numpy_show_config": _numpy_show_config(),
            "threadpoolctl": _threadpool_info(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "visible_gpu_mapping": _visible_gpu_mapping(),
            "cuda_toolkit_version": _cuda_toolkit_version(),
            "cudnn_version": _cudnn_version(),
            "mxcsr_state": _mxcsr_state(),
        }
    )
    return _apply_bucket_contract_summary(
        payload,
        contract_violations=[],
        validation_details={
            "schema_version": 1,
            "bucket_kind": "node_env",
            "visible_gpu_mapping": payload.get("visible_gpu_mapping"),
            "cuda_toolkit_version": payload.get("cuda_toolkit_version"),
            "cudnn_version": payload.get("cudnn_version"),
            "mxcsr_state": payload.get("mxcsr_state"),
        },
    )


def _build_queue_payload(request: dict[str, Any]) -> dict[str, Any]:
    cluster_name = request.get("cluster")
    queue_name = request.get("queue")
    queue_def = None
    cluster_type = None
    if isinstance(cluster_name, str) and cluster_name:
        cluster = get_cluster(cluster_name)
        cluster_type = cluster.type
        queues = cluster.queues or {}
        queue_def = queues.get(queue_name)
    payload = _common_payload("queue")
    payload.update(
        {
            "label": request.get("label"),
            "cluster": cluster_name,
            "cluster_type": cluster_type,
            "queue": queue_name,
            "remote_target": request.get("remote_target"),
            "requested_node": request.get("requested_node"),
            "queue_parameters": (
                _json_safe(dataclasses.asdict(queue_def))
                if queue_def is not None
                else None
            ),
        }
    )
    return _apply_bucket_contract_summary(
        payload,
        contract_violations=[],
        validation_details={
            "schema_version": 1,
            "bucket_kind": "queue",
            "queue_parameters": payload.get("queue_parameters"),
        },
    )


def _build_queue_node_payload(request: dict[str, Any]) -> dict[str, Any]:
    payload = _common_payload("queue_node")
    payload.update(
        {
            "label": request.get("label"),
            "hostname": socket.gethostname(),
            "queue_checksum": request.get("queue_checksum"),
            "requested_node": request.get("requested_node"),
            "environment_variables": _queue_runtime_env(),
            "allocation_counts": _allocation_counts(),
            "resource_limits": _resource_limits(),
            "cgroup_memory_limit_bytes": _cgroup_memory_limit_bytes(),
            "affinity_cores": _affinity_count(),
        }
    )
    contract_violations, validation_details = _queue_node_contract_summary()
    return _apply_bucket_contract_summary(
        payload,
        contract_violations=contract_violations,
        validation_details=validation_details,
    )


def _probe_request_for_bucket(
    bucket_kind: str,
    plan: dict[str, Any],
    *,
    label: str,
    current_checksums: dict[str, str],
) -> dict[str, Any]:
    request = {
        "bucket_kind": bucket_kind,
        "label": label,
        "cluster": plan.get("cluster"),
        "remote_target": plan.get("remote_target"),
        "requested_node": plan["labels"].get("node"),
        "hostname": plan.get("hostname"),
        "live_tokens": plan.get("live_tokens", {}).get(bucket_kind),
        "env_checksum": plan.get("env_checksum"),
    }
    if bucket_kind == "node_env":
        request["node_checksum"] = current_checksums.get("node")
        request["environment_checksum"] = current_checksums.get("environment")
    elif bucket_kind == "queue":
        queue_label = label.split("/", 2)
        if len(queue_label) >= 2:
            request["cluster"] = queue_label[0]
            request["queue"] = queue_label[1]
    elif bucket_kind == "queue_node":
        request["queue_checksum"] = current_checksums.get("queue")
    return request


def execute_probe_request(probe_payload: dict[str, Any]) -> dict[str, Any]:
    mode = probe_payload.get("mode")
    if mode == "discover":
        transformation_dict = dict(probe_payload.get("target_transformation_dict") or {})
        tf_dunder = dict(probe_payload.get("target_tf_dunder") or {})
        return resolve_probe_plan(transformation_dict, tf_dunder)

    if mode != "capture":
        raise ValueError(f"Unknown probe mode '{mode}'")

    request = dict(probe_payload.get("request") or {})
    bucket_kind = request.get("bucket_kind")
    if bucket_kind == "node":
        return _build_node_payload(request)
    if bucket_kind == "environment":
        return _build_environment_payload(request)
    if bucket_kind == "node_env":
        return _build_node_env_payload(request)
    if bucket_kind == "queue":
        return _build_queue_payload(request)
    if bucket_kind == "queue_node":
        return _build_queue_node_payload(request)
    raise ValueError(f"Unknown bucket kind '{bucket_kind}'")


def _run_probe_transformation_sync(
    target_transformation_dict: dict[str, Any],
    target_tf_dunder: dict[str, Any] | None,
    *,
    probe_payload: dict[str, Any],
) -> dict[str, Any]:
    env_checksum = resolve_env_checksum(target_transformation_dict, target_tf_dunder)
    pretransformation_dict: dict[str, Any] = {
        "__language__": "python",
        "__output__": ("result", "plain", None),
        RECORD_PROBE_DUNDER: {"mode": probe_payload["mode"]},
        "code": ("python", None, PROBE_CODE),
        "probe_payload": ("plain", None, probe_payload),
        "nonce": ("str", None, _utcnow_iso()),
    }
    if env_checksum is not None:
        pretransformation_dict["__env__"] = Checksum(env_checksum).hex()

    pre = PreTransformation(pretransformation_dict)
    tf = transformation_from_pretransformation(
        pre,
        upstream_dependencies={},
        meta={},
        scratch=False,
    )
    result = tf.run()
    if not isinstance(result, dict):
        raise RuntimeError(
            f"Probe transformation returned {type(result).__name__}, expected dict"
        )
    return result


def discover_probe_plan_sync(
    target_transformation_dict: dict[str, Any],
    target_tf_dunder: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _run_probe_transformation_sync(
        target_transformation_dict,
        target_tf_dunder,
        probe_payload={
            "mode": "discover",
            "target_transformation_dict": target_transformation_dict,
            "target_tf_dunder": target_tf_dunder or {},
        },
    )


def capture_probe_payload_sync(
    target_transformation_dict: dict[str, Any],
    target_tf_dunder: dict[str, Any] | None,
    *,
    request: dict[str, Any],
) -> dict[str, Any]:
    return _run_probe_transformation_sync(
        target_transformation_dict,
        target_tf_dunder,
        probe_payload={"mode": "capture", "request": request},
    )


def _needs_refresh(
    existing_probe: dict[str, Any] | None,
    *,
    live_tokens: dict[str, Any],
    force: bool,
) -> tuple[bool, str]:
    if force:
        return True, "force"
    if existing_probe is None:
        return True, "missing"
    if existing_probe.get("freshness_tokens") != live_tokens:
        return True, "stale"
    return False, "fresh"


async def refresh_required_buckets(
    target_transformation_dict: dict[str, Any],
    target_tf_dunder: dict[str, Any] | None = None,
    *,
    force: bool = False,
    msg_func=None,
) -> dict[str, Any]:
    if buffer_remote is None or database_remote is None:
        raise RuntimeError("Bucket probing requires hashserver and database clients")
    if not buffer_remote.has_write_server():
        raise RuntimeError("Bucket probing requires an active hashserver")
    if not database_remote.has_write_server():
        raise RuntimeError("Bucket probing requires an active database server")
    if not database_remote.has_read_server():
        raise RuntimeError("Bucket probing requires an active database read server")

    plan = discover_probe_plan_sync(target_transformation_dict, target_tf_dunder)
    required_kinds = list(plan["required_kinds"])
    labels = dict(plan["labels"])
    live_tokens = dict(plan["live_tokens"])
    current_probes: dict[str, dict[str, Any]] = {}
    refreshed: list[dict[str, Any]] = []
    reused: list[dict[str, Any]] = []
    refreshed_kinds: set[str] = set()

    async def _store_probe(bucket_kind: str, label: str, request: dict[str, Any]) -> None:
        payload = capture_probe_payload_sync(
            target_transformation_dict,
            target_tf_dunder,
            request=request,
        )
        checksum_hex = Buffer(payload, "plain").get_checksum().hex()
        written = await Buffer(payload, "plain").write()
        if not written:
            raise RuntimeError(
                f"Failed to write {bucket_kind} probe payload to the hashserver"
            )
        captured_at = _utcnow_iso()
        await database_remote.set_bucket_probe(
            bucket_kind,
            label,
            checksum_hex,
            live_tokens[bucket_kind],
            captured_at,
        )
        current_probes[bucket_kind] = {
            "bucket_kind": bucket_kind,
            "label": label,
            "bucket_checksum": checksum_hex,
            "captured_at": captured_at,
            "freshness_tokens": live_tokens[bucket_kind],
        }
        refreshed_kinds.add(bucket_kind)
        refreshed.append({"bucket_kind": bucket_kind, "label": label})
        if msg_func is not None:
            msg_func(1, f"Refreshed {bucket_kind} bucket for {label!r}")

    for bucket_kind in ("node", "environment", "queue"):
        if bucket_kind not in required_kinds:
            continue
        label = labels[bucket_kind]
        existing = await database_remote.get_bucket_probe(bucket_kind, label)
        need_refresh, reason = _needs_refresh(
            existing,
            live_tokens=live_tokens[bucket_kind],
            force=force,
        )
        if need_refresh:
            request = _probe_request_for_bucket(
                bucket_kind,
                plan,
                label=label,
                current_checksums={
                    kind: probe["bucket_checksum"]
                    for kind, probe in current_probes.items()
                },
            )
            await _store_probe(bucket_kind, label, request)
        else:
            current_probes[bucket_kind] = existing
            reused.append({"bucket_kind": bucket_kind, "label": label, "reason": reason})
            if msg_func is not None:
                msg_func(2, f"Reused fresh {bucket_kind} bucket for {label!r}")

    if "node_env" in required_kinds:
        labels["node_env"] = (
            f"{current_probes['node']['bucket_checksum']}:"
            f"{current_probes['environment']['bucket_checksum']}"
        )
        live_tokens["node_env"] = {
            "node": live_tokens["node"],
            "environment": live_tokens["environment"],
        }
        existing = await database_remote.get_bucket_probe("node_env", labels["node_env"])
        need_refresh, reason = _needs_refresh(
            existing,
            live_tokens=live_tokens["node_env"],
            force=force or bool({"node", "environment"} & refreshed_kinds),
        )
        if need_refresh:
            request = _probe_request_for_bucket(
                "node_env",
                plan,
                label=labels["node_env"],
                current_checksums={
                    "node": current_probes["node"]["bucket_checksum"],
                    "environment": current_probes["environment"]["bucket_checksum"],
                },
            )
            await _store_probe("node_env", labels["node_env"], request)
        else:
            current_probes["node_env"] = existing
            reused.append(
                {
                    "bucket_kind": "node_env",
                    "label": labels["node_env"],
                    "reason": reason,
                }
            )

    if "queue_node" in required_kinds:
        labels["queue_node"] = (
            f"{current_probes['queue']['bucket_checksum']}:{plan['hostname']}"
        )
        live_tokens["queue_node"] = {
            "queue": live_tokens["queue"],
            "node": live_tokens["node"],
        }
        existing = await database_remote.get_bucket_probe(
            "queue_node", labels["queue_node"]
        )
        need_refresh, reason = _needs_refresh(
            existing,
            live_tokens=live_tokens["queue_node"],
            force=force or bool({"queue", "node"} & refreshed_kinds),
        )
        if need_refresh:
            request = _probe_request_for_bucket(
                "queue_node",
                plan,
                label=labels["queue_node"],
                current_checksums={
                    "queue": current_probes["queue"]["bucket_checksum"],
                },
            )
            await _store_probe("queue_node", labels["queue_node"], request)
        else:
            current_probes["queue_node"] = existing
            reused.append(
                {
                    "bucket_kind": "queue_node",
                    "label": labels["queue_node"],
                    "reason": reason,
                }
            )

    return {
        "required_bucket_labels": {kind: labels[kind] for kind in required_kinds},
        "required_bucket_checksums": {
            kind: current_probes[kind]["bucket_checksum"] for kind in required_kinds
        },
        "live_tokens": {kind: live_tokens[kind] for kind in required_kinds},
        "bucket_tokens": {
            kind: current_probes[kind]["freshness_tokens"] for kind in required_kinds
        },
        "refreshed": refreshed,
        "reused": reused,
    }


def refresh_required_buckets_sync(
    target_transformation_dict: dict[str, Any],
    target_tf_dunder: dict[str, Any] | None = None,
    *,
    force: bool = False,
    msg_func=None,
) -> dict[str, Any]:
    async def _run():
        return await refresh_required_buckets(
            target_transformation_dict,
            target_tf_dunder,
            force=force,
            msg_func=msg_func,
        )

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run())

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def _runner():
        try:
            result["value"] = asyncio.run(_run())
        except BaseException as exc:  # pragma: no cover - rare sync bridge
            error["exc"] = exc

    import threading

    thread = threading.Thread(target=_runner, daemon=True, name="probe-refresh")
    thread.start()
    thread.join()
    if error:
        raise error["exc"]
    return result["value"]


__all__ = [
    "RECORD_PROBE_DUNDER",
    "discover_probe_plan_sync",
    "capture_probe_payload_sync",
    "execute_probe_request",
    "refresh_required_buckets",
    "refresh_required_buckets_sync",
]
