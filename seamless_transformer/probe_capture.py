"""Probe capture helpers for seamless-probe and execution-record buckets."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import importlib.metadata
import io
import locale
import os
import platform
import resource
import socket
import sys
import time
from datetime import datetime, timezone
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


def _selected_env_vars(*prefixes: str, names: tuple[str, ...] = ()) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in os.environ.items():
        if key in names or any(key.startswith(prefix) for prefix in prefixes):
            result[key] = value
    return dict(sorted(result.items()))


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


def _affinity_count() -> int | None:
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return None


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


def _python_packages() -> list[dict[str, str]]:
    packages: list[dict[str, str]] = []
    try:
        distributions = importlib.metadata.distributions()
    except Exception:
        return packages
    for dist in distributions:
        name = dist.metadata.get("Name") or dist.metadata.get("Summary")
        if not name:
            continue
        packages.append({"name": str(name), "version": str(dist.version)})
    packages.sort(key=lambda item: item["name"].lower())
    return packages


def _numpy_show_config() -> str | None:
    try:
        import numpy as np
    except Exception:
        return None
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
                "logical_cores": os.cpu_count(),
                "affinity_cores": _affinity_count(),
            },
            "memory_total_bytes": _memory_total_bytes(),
        }
    )
    return payload


def _build_environment_payload(request: dict[str, Any]) -> dict[str, Any]:
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
                    "CUDA_VISIBLE_DEVICES",
                    "CONDA_DEFAULT_ENV",
                    "CONDA_PREFIX",
                ),
            ),
            "python_packages": _python_packages(),
        }
    )
    return payload


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
        }
    )
    return payload


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
    return payload


def _build_queue_node_payload(request: dict[str, Any]) -> dict[str, Any]:
    payload = _common_payload("queue_node")
    payload.update(
        {
            "label": request.get("label"),
            "hostname": socket.gethostname(),
            "queue_checksum": request.get("queue_checksum"),
            "requested_node": request.get("requested_node"),
            "environment_variables": _selected_env_vars(
                "OMP_",
                "GOMP_",
                "KMP_",
                "MKL_",
                "OPENBLAS_",
                names=("CUDA_VISIBLE_DEVICES",),
            ),
            "resource_limits": _resource_limits(),
            "cgroup_memory_limit_bytes": _cgroup_memory_limit_bytes(),
            "affinity_cores": _affinity_count(),
        }
    )
    return payload


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
