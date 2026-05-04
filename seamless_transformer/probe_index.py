"""Shared execution-record probe-index helpers."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import socket
import sys
import sysconfig
from pathlib import Path
from typing import Any

from seamless import Checksum
from seamless_transformer import record_utils
from seamless_transformer.transformation_utils import resolve_env_checksum

from seamless_config.cluster import get_cluster
from seamless_config.select import (
    check_remote_redundancy,
    get_execution,
    get_queue,
    get_remote,
    get_selected_cluster,
)
from seamless_transformer.record_runtime import get_record_mode

try:
    from seamless_remote import database_remote
except ImportError:  # pragma: no cover - optional dependency
    database_remote = None


class RecordBucketError(RuntimeError):
    """Raised when record-mode bucket preconditions are not satisfied."""


RECORD_PROBE_DUNDER = "__record_probe__"


def is_record_probe(
    transformation_dict: dict[str, Any] | None = None,
    tf_dunder: dict[str, Any] | None = None,
) -> bool:
    for payload in (tf_dunder, transformation_dict):
        if not isinstance(payload, dict):
            continue
        if payload.get(RECORD_PROBE_DUNDER):
            return True
    return False


def _resolve_remote_target(execution: str) -> str | None:
    return record_utils._resolve_remote_target(
        execution,
        get_remote=get_remote,
        get_selected_cluster=get_selected_cluster,
        check_remote_redundancy=check_remote_redundancy,
    )


def _read_boot_id() -> str | None:
    path = Path("/proc/sys/kernel/random/boot_id")
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except Exception:
        return None


def _path_mtime_ns(path: Path) -> int | None:
    try:
        return int(path.stat().st_mtime_ns)
    except Exception:
        return None


def _queue_config_digest(cluster_name: str, queue_name: str) -> str:
    cluster = get_cluster(cluster_name)
    queues = cluster.queues or {}
    queue = queues[queue_name]
    payload = {
        "cluster": cluster_name,
        "cluster_type": cluster.type,
        "queue": dataclasses.asdict(queue),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest


def _load_env_dict(
    transformation_dict: dict[str, Any], tf_dunder: dict[str, Any] | None
) -> tuple[str | None, dict[str, Any]]:
    env_checksum = resolve_env_checksum(transformation_dict, tf_dunder)
    if env_checksum is None:
        return None, {}
    env_checksum = Checksum(env_checksum).hex()
    try:
        env_buffer = Checksum(env_checksum).resolve()
    except Exception:
        env_buffer = None
    if env_buffer is None:
        return env_checksum, {}
    try:
        env_dict = env_buffer.get_value("plain")
    except Exception:
        env_dict = {}
    if not isinstance(env_dict, dict):
        env_dict = {}
    return env_checksum, env_dict


def _environment_label_and_tokens(
    env_dict: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    docker = env_dict.get("docker")
    if isinstance(docker, dict) and isinstance(docker.get("name"), str):
        docker_name = docker["name"]
        return f"docker:{docker_name}", {"docker_name": docker_name}

    if "conda" in env_dict or "conda_environment" in env_dict:
        prefix = sys.prefix
        tokens: dict[str, Any] = {"sys_prefix": prefix}
        conda_history_mtime_ns = _path_mtime_ns(
            Path(prefix) / "conda-meta" / "history"
        )
        if conda_history_mtime_ns is not None:
            tokens["conda_history_mtime_ns"] = conda_history_mtime_ns
        purelib = sysconfig.get_path("purelib")
        if purelib:
            purelib_mtime_ns = _path_mtime_ns(Path(purelib))
            if purelib_mtime_ns is not None:
                tokens["purelib_mtime_ns"] = purelib_mtime_ns
        conda_environment = env_dict.get("conda_environment")
        if isinstance(conda_environment, str) and conda_environment:
            tokens["conda_environment"] = conda_environment
        return f"conda:{prefix}", tokens

    prefix = sys.prefix
    tokens = {"sys_prefix": prefix}
    purelib = sysconfig.get_path("purelib")
    if purelib:
        purelib_mtime_ns = _path_mtime_ns(Path(purelib))
        if purelib_mtime_ns is not None:
            tokens["purelib_mtime_ns"] = purelib_mtime_ns
    return f"python:{prefix}", tokens


def _required_bucket_kinds(execution: str, remote_target: str | None) -> list[str]:
    if execution == "remote" and remote_target == "daskserver":
        return ["node", "environment", "node_env", "queue", "queue_node"]
    return ["node", "environment", "node_env"]


def resolve_probe_plan(
    transformation_dict: dict[str, Any],
    tf_dunder: dict[str, Any] | None = None,
    *,
    execution: str | None = None,
    hostname: str | None = None,
) -> dict[str, Any]:
    execution = execution or get_execution()
    remote_target = _resolve_remote_target(execution)
    hostname = hostname or socket.gethostname()
    cluster_name = get_selected_cluster()
    env_checksum, env_dict = _load_env_dict(transformation_dict, tf_dunder)
    environment_label, environment_tokens = _environment_label_and_tokens(env_dict)

    labels: dict[str, str] = {"node": hostname, "environment": environment_label}
    live_tokens: dict[str, dict[str, Any]] = {
        "node": {"hostname": hostname},
        "environment": environment_tokens,
    }
    boot_id = _read_boot_id()
    if boot_id is not None:
        live_tokens["node"]["boot_id"] = boot_id

    required_kinds = _required_bucket_kinds(execution, remote_target)

    if "queue" in required_kinds:
        if cluster_name is None:
            raise RecordBucketError(
                "Record mode requires a selected cluster for daskserver execution"
            )
        cluster = get_cluster(cluster_name)
        queue_name = get_queue(cluster_name) or cluster.default_queue
        if queue_name is None:
            raise RecordBucketError(
                f"Record mode requires a selected queue for cluster '{cluster_name}'"
            )
        labels["queue"] = f"{cluster_name}/{queue_name}/{remote_target}"
        live_tokens["queue"] = {
            "queue_config_sha256": _queue_config_digest(cluster_name, queue_name)
        }

    return {
        "execution": execution,
        "remote_target": remote_target,
        "hostname": hostname,
        "cluster": cluster_name,
        "env_checksum": env_checksum,
        "required_kinds": required_kinds,
        "labels": labels,
        "live_tokens": live_tokens,
    }


def _format_bucket_list(items: list[tuple[str, str]]) -> str:
    return ", ".join(f"{kind}={label!r}" for kind, label in items)


async def ensure_record_bucket_preconditions(
    transformation_dict: dict[str, Any],
    tf_dunder: dict[str, Any] | None = None,
    *,
    execution: str | None = None,
    hostname: str | None = None,
) -> dict[str, Any] | None:
    if not get_record_mode():
        return None
    if database_remote is None or not database_remote.has_read_server():
        raise RecordBucketError("Record mode requires an active database read server")

    plan = resolve_probe_plan(
        transformation_dict,
        tf_dunder,
        execution=execution,
        hostname=hostname,
    )
    required_kinds = plan["required_kinds"]
    labels = dict(plan["labels"])
    live_tokens = dict(plan["live_tokens"])
    probes: dict[str, dict[str, Any]] = {}
    missing: list[tuple[str, str]] = []
    stale: list[tuple[str, str]] = []

    for kind in ("node", "environment", "queue"):
        if kind not in required_kinds:
            continue
        label = labels[kind]
        probe = await database_remote.get_bucket_probe(kind, label)
        if probe is None:
            missing.append((kind, label))
            continue
        probes[kind] = probe
        if probe.get("freshness_tokens") != live_tokens[kind]:
            stale.append((kind, label))

    if "node_env" in required_kinds and "node" in probes and "environment" in probes:
        labels["node_env"] = (
            f"{probes['node']['bucket_checksum']}:"
            f"{probes['environment']['bucket_checksum']}"
        )
        live_tokens["node_env"] = {
            "node": live_tokens["node"],
            "environment": live_tokens["environment"],
        }
        probe = await database_remote.get_bucket_probe("node_env", labels["node_env"])
        if probe is None:
            missing.append(("node_env", labels["node_env"]))
        else:
            probes["node_env"] = probe
            if probe.get("freshness_tokens") != live_tokens["node_env"]:
                stale.append(("node_env", labels["node_env"]))

    if "queue_node" in required_kinds and "queue" in probes and "node" in probes:
        labels["queue_node"] = f"{probes['queue']['bucket_checksum']}:{plan['hostname']}"
        live_tokens["queue_node"] = {
            "queue": live_tokens["queue"],
            "node": live_tokens["node"],
        }
        probe = await database_remote.get_bucket_probe(
            "queue_node", labels["queue_node"]
        )
        if probe is None:
            missing.append(("queue_node", labels["queue_node"]))
        else:
            probes["queue_node"] = probe
            if probe.get("freshness_tokens") != live_tokens["queue_node"]:
                stale.append(("queue_node", labels["queue_node"]))

    if missing:
        raise RecordBucketError(
            "Record mode requires bucket probes for: " + _format_bucket_list(missing)
        )
    if stale:
        raise RecordBucketError(
            "Record mode detected stale bucket probes for: "
            + _format_bucket_list(stale)
        )

    required_bucket_checksums = {
        kind: probes[kind]["bucket_checksum"] for kind in required_kinds
    }
    bucket_tokens = {
        kind: probes[kind]["freshness_tokens"] for kind in required_kinds
    }
    return {
        "required_bucket_labels": {kind: labels[kind] for kind in required_kinds},
        "required_bucket_checksums": required_bucket_checksums,
        "live_tokens": {kind: live_tokens[kind] for kind in required_kinds},
        "bucket_tokens": bucket_tokens,
    }


def ensure_record_bucket_preconditions_sync(
    transformation_dict: dict[str, Any],
    tf_dunder: dict[str, Any] | None = None,
    *,
    execution: str | None = None,
    hostname: str | None = None,
) -> dict[str, Any] | None:
    async def _run():
        return await ensure_record_bucket_preconditions(
            transformation_dict,
            tf_dunder,
            execution=execution,
            hostname=hostname,
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

    thread = threading.Thread(target=_runner, daemon=True, name="record-bucket-check")
    thread.start()
    thread.join()
    if error:
        raise error["exc"]
    return result.get("value")


__all__ = [
    "RecordBucketError",
    "RECORD_PROBE_DUNDER",
    "resolve_probe_plan",
    "ensure_record_bucket_preconditions",
    "ensure_record_bucket_preconditions_sync",
    "is_record_probe",
]
