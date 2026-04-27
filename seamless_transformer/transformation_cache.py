"""Transformation cache helpers."""

from typing import Any, Dict

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
import itertools
import os
import os as _os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time

from seamless import Buffer, CacheMissError, Checksum, is_worker

from .remote_job import RemoteJobWritten, parse_remote_job_written
from .probe_index import ensure_record_bucket_preconditions, is_record_probe
from .run import (
    _module_definition_from_payload,
    _resolve_dunder_value,
    get_transformation_inputs_output,
    run_transformation_dict,
)
from . import worker
from seamless_config.select import (
    check_remote_redundancy,
    get_execution,
    get_node,
    get_queue,
    get_record,
    get_remote,
    get_selected_cluster,
)

try:
    from seamless.caching import buffer_writer as _buffer_writer
except ImportError:  # pragma: no cover - optional dependency
    _buffer_writer = None

try:
    from seamless_remote import database_remote, jobserver_remote, buffer_remote
except ImportError:
    database_remote = jobserver_remote = buffer_remote = None

try:
    from seamless_remote.client import close_all_clients as _close_all_clients
except ImportError:  # pragma: no cover - optional dependency
    _close_all_clients = None

# In-process cache of transformation results
_DEBUG = os.environ.get("SEAMLESS_DEBUG_TRANSFORMATION", "").lower() in (
    "1",
    "true",
    "yes",
)
try:
    _SEAMLESS_VERSION = importlib_metadata.version("seamless")
except Exception:  # pragma: no cover - fallback for editable/source-only installs
    try:
        _SEAMLESS_VERSION = importlib_metadata.version("seamless-transformer")
    except Exception:  # pragma: no cover - last resort
        _SEAMLESS_VERSION = "unknown"
_PROCESS_STARTED_AT = datetime.now(timezone.utc)
_EXECUTION_RECORD_COUNTER = itertools.count(1)
_VALIDATION_SNAPSHOT_COUNTS: dict[tuple, int] = {}


def _debug(msg: str) -> None:
    if _DEBUG:
        print(f"[transformation_cache] {msg}", flush=True)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _resolve_remote_target(execution: str) -> str | None:
    if execution != "remote":
        return None
    remote_target = get_remote()
    if remote_target is not None:
        return remote_target
    cluster = get_selected_cluster()
    if cluster is None:
        return None
    try:
        return check_remote_redundancy(cluster)
    except Exception:
        return None


def _validation_snapshot_limit() -> int:
    raw = os.environ.get("SEAMLESS_RECORD_VALIDATION_SNAPSHOT_LIMIT", "1")
    try:
        value = int(raw)
    except Exception:
        return 1
    return max(value, 0)


def _stable_digest(value: Any) -> str:
    import hashlib
    import json

    try:
        payload = json.dumps(value, sort_keys=True, default=repr).encode()
    except TypeError:
        payload = repr(value).encode()
    return hashlib.sha256(payload).hexdigest()


def _compiler_version(binary: str) -> str | None:
    try:
        completed = subprocess.run(
            [binary, "--version"],
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


async def build_compilation_context_checksum(
    transformation_dict: Dict[str, Any],
    tf_dunder: Dict[str, Any] | None,
) -> str | None:
    if not (
        transformation_dict.get("__compiled__")
        or (isinstance(tf_dunder, dict) and tf_dunder.get("__compiled__"))
    ):
        return None
    if buffer_remote is None or not buffer_remote.has_write_server():
        raise RuntimeError(
            "Compiled execution records require an active hashserver write server"
        )

    from seamless_transformer.compiler.compile import complete

    schema_checksum = transformation_dict.get("__schema__")
    if schema_checksum is None and isinstance(tf_dunder, dict):
        schema_checksum = tf_dunder.get("__schema__")
    header_checksum = transformation_dict.get("__header__")
    if header_checksum is None and isinstance(tf_dunder, dict):
        header_checksum = tf_dunder.get("__header__")
    compilation_checksum = transformation_dict.get("__compilation__")
    if compilation_checksum is None and isinstance(tf_dunder, dict):
        compilation_checksum = tf_dunder.get("__compilation__")

    code_checksum = transformation_dict.get("code", (None, None, None))[2]
    objects_checksum = transformation_dict.get("objects", (None, None, None))[2]
    language = transformation_dict.get("__language__")

    code = await Checksum(code_checksum).resolution("text")
    header = _resolve_dunder_value(transformation_dict, tf_dunder, "__header__", "text")
    compilation = _resolve_dunder_value(
        transformation_dict, tf_dunder, "__compilation__", "plain"
    )
    objects = {}
    if objects_checksum is not None:
        objects = await Checksum(objects_checksum).resolution("plain")
        if not isinstance(objects, dict):
            objects = {}

    module_definition = _module_definition_from_payload(
        language, code, header, objects, compilation
    )
    completed = complete(module_definition)
    compiled_module_digest = _stable_digest(completed)

    objects_payload: dict[str, Any] = {}
    for name, obj in sorted(completed["objects"].items()):
        compiler_binary = obj["compiler_binary"]
        compiler_path = shutil.which(compiler_binary) or compiler_binary
        objects_payload[name] = {
            "language": obj["language"],
            "compiler_binary": compiler_binary,
            "compiler_path": compiler_path,
            "compiler_version": _compiler_version(compiler_path),
            "compile_mode": obj.get("compile_mode"),
            "extension": obj.get("extension"),
            "options": list(obj.get("options") or []),
        }

    payload = {
        "schema_version": 1,
        "language": language,
        "target": completed.get("target", "profile"),
        "schema_checksum": str(schema_checksum) if schema_checksum is not None else None,
        "header_checksum": str(header_checksum) if header_checksum is not None else None,
        "code_checksum": str(code_checksum) if code_checksum is not None else None,
        "objects_checksum": (
            str(objects_checksum) if objects_checksum is not None else None
        ),
        "compilation_checksum": (
            str(compilation_checksum) if compilation_checksum is not None else None
        ),
        "object_names": sorted(completed["objects"].keys()),
        "link_options": list(completed.get("link_options", [])),
        "objects": objects_payload,
        "compiled_module_digest": compiled_module_digest,
        "compiled_module_validation_digest": None,
    }
    buffer = Buffer(payload, "plain")
    written = await buffer.write()
    if not written:
        raise RuntimeError("Failed to write compilation context to the hashserver")
    return buffer.get_checksum().hex()


def _validation_snapshot_key(
    execution: str,
    probe_context: dict[str, Any] | None,
    compilation_context: str | None,
) -> tuple:
    required_bucket_checksums = {}
    if isinstance(probe_context, dict):
        required_bucket_checksums = dict(
            probe_context.get("required_bucket_checksums", {}) or {}
        )
    return (
        execution,
        tuple(sorted(required_bucket_checksums.items())),
        compilation_context,
    )


async def build_validation_snapshot_checksum(
    transformation_dict: Dict[str, Any],
    tf_dunder: Dict[str, Any] | None,
    *,
    execution: str,
    probe_context: dict[str, Any] | None,
    compilation_context: str | None,
    bucket_contract_violations: list[str] | None,
    job_contract_violations: list[str] | None,
) -> str | None:
    limit = _validation_snapshot_limit()
    if limit <= 0:
        return None
    if buffer_remote is None or not buffer_remote.has_write_server():
        return None

    key = _validation_snapshot_key(execution, probe_context, compilation_context)
    count = _VALIDATION_SNAPSHOT_COUNTS.get(key, 0)
    if count >= limit:
        return None

    payload = {
        "schema_version": 1,
        "execution_mode": execution,
        "remote_target": _resolve_remote_target(execution),
        "language": transformation_dict.get("__language__", "python"),
        "compiled": bool(
            transformation_dict.get("__compiled__")
            or (isinstance(tf_dunder, dict) and tf_dunder.get("__compiled__"))
        ),
        "requested_cluster": get_selected_cluster(),
        "requested_queue": get_queue(get_selected_cluster()),
        "requested_node": get_node(),
        "probe_context": probe_context or {},
        "compilation_context": compilation_context,
        "bucket_contract_violations": sorted(set(bucket_contract_violations or [])),
        "job_contract_violations": sorted(set(job_contract_violations or [])),
        "hostname": socket.gethostname(),
        "pid": _os.getpid(),
        "cwd": os.getcwd(),
        "tempdir": tempfile.gettempdir(),
        "platform": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_executable": sys.executable,
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "environment": {
            key: _os.environ.get(key)
            for key in (
                "PATH",
                "PYTHONPATH",
                "LD_LIBRARY_PATH",
                "LD_PRELOAD",
                "CONDA_PREFIX",
                "CONDA_DEFAULT_ENV",
                "CUDA_VISIBLE_DEVICES",
            )
            if _os.environ.get(key) is not None
        },
        "sys_path": list(sys.path),
    }
    buffer = Buffer(payload, "plain")
    written = await buffer.write()
    if not written:
        return None
    _VALIDATION_SNAPSHOT_COUNTS[key] = count + 1
    return buffer.get_checksum().hex()


def build_execution_record(
    transformation_dict: Dict[str, Any],
    *,
    tf_checksum: Checksum,
    result_checksum: Checksum,
    tf_dunder: Dict[str, Any] | None,
    execution: str,
    started_at: str,
    finished_at: str,
    wall_time_seconds: float,
    cpu_user_seconds: float,
    cpu_system_seconds: float,
    probe_context: dict[str, Any] | None = None,
    bucket_contract_violations: list[str] | None = None,
    job_contract_violations: list[str] | None = None,
    compilation_context: str | None = None,
    validation_snapshot: str | None = None,
) -> dict[str, Any]:
    meta = transformation_dict.get("__meta__", {}) or {}
    if not isinstance(meta, dict):
        meta = {}
    env_checksum = transformation_dict.get("__env__")
    if env_checksum is None and isinstance(tf_dunder, dict):
        env_checksum = tf_dunder.get("__env__")
    if isinstance(env_checksum, Checksum):
        env_checksum = env_checksum.hex()
    elif env_checksum is not None:
        env_checksum = str(env_checksum)

    remote_target = _resolve_remote_target(execution)
    requested_cluster = get_selected_cluster()
    requested_queue = get_queue(requested_cluster)
    requested_node = get_node()
    language = transformation_dict.get("__language__", "python")
    if language not in ("python", "bash", "compiled"):
        if transformation_dict.get("__compiled__"):
            language = "compiled"
        else:
            language = "python"

    execution_envelope = {
        "requested_cluster": requested_cluster,
        "requested_queue": requested_queue,
        "requested_node": requested_node,
        "actual_remote_target": remote_target,
        "scratch": bool(meta.get("scratch", False)),
        "allow_input_fingertip": bool(meta.get("allow_input_fingertip", False)),
        "language_kind": language,
        "resolved_env_checksum": env_checksum,
        "active_tf_dunder_keys": sorted((tf_dunder or {}).keys()),
    }
    probe_context = probe_context or {}
    bucket_contract_violations = sorted(set(bucket_contract_violations or []))
    job_contract_violations = sorted(set(job_contract_violations or []))
    contract_violations = sorted(
        set(bucket_contract_violations) | set(job_contract_violations)
    )
    required_bucket_checksums = dict(probe_context.get("required_bucket_checksums", {}))
    freshness = {
        "required_bucket_labels": dict(
            probe_context.get("required_bucket_labels", {})
        ),
        "required_bucket_checksums": required_bucket_checksums,
        "live_tokens": dict(probe_context.get("live_tokens", {})),
        "bucket_tokens": dict(probe_context.get("bucket_tokens", {})),
    }

    return {
        "schema_version": 1,
        "checksum_fields": [
            "node",
            "environment",
            "node_env",
            "queue",
            "queue_node",
            "compilation_context",
            "validation_snapshot",
        ],
        "tf_checksum": tf_checksum.hex(),
        "result_checksum": result_checksum.hex(),
        "seamless_version": _SEAMLESS_VERSION,
        "execution_mode": execution,
        "remote_target": remote_target,
        "node": required_bucket_checksums.get("node"),
        "environment": required_bucket_checksums.get("environment"),
        "node_env": required_bucket_checksums.get("node_env"),
        "queue": required_bucket_checksums.get("queue"),
        "queue_node": required_bucket_checksums.get("queue_node"),
        "execution_envelope": execution_envelope,
        "compilation_context": compilation_context,
        "freshness": freshness,
        "bucket_contract_violations": bucket_contract_violations,
        "job_contract_violations": job_contract_violations,
        "contract_violations": contract_violations,
        "validation_snapshot": validation_snapshot,
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_time_seconds": wall_time_seconds,
        "cpu_time_user_seconds": cpu_user_seconds,
        "cpu_time_system_seconds": cpu_system_seconds,
        "memory_peak_bytes": None,
        "gpu_memory_peak_bytes": None,
        "input_total_bytes": None,
        "output_total_bytes": None,
        "compilation_time_seconds": None,
        "hostname": socket.gethostname(),
        "pid": _os.getpid(),
        "process_started_at": _PROCESS_STARTED_AT.replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "worker_execution_index": next(_EXECUTION_RECORD_COUNTER),
        "retry_count": 0,
    }


async def load_bucket_contract_violations(
    probe_context: dict[str, Any] | None,
) -> list[str]:
    if not isinstance(probe_context, dict):
        return []
    required_bucket_checksums = probe_context.get("required_bucket_checksums", {})
    if not isinstance(required_bucket_checksums, dict):
        return []
    violations: set[str] = set()
    for checksum_hex in required_bucket_checksums.values():
        if not checksum_hex:
            continue
        try:
            payload = await Checksum(checksum_hex).resolution("plain")
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        contract_violations = payload.get("contract_violations")
        if not isinstance(contract_violations, list):
            continue
        for code in contract_violations:
            if isinstance(code, str) and code:
                violations.add(code)
    return sorted(violations)


async def _checksum_length(checksum_hex: str | None) -> int | None:
    if not checksum_hex:
        return None
    checksum = Checksum(checksum_hex)
    try:
        if buffer_remote is not None:
            lengths = await buffer_remote.get_buffer_lengths([checksum])
            if lengths and lengths[0] is not None:
                return int(lengths[0])
    except Exception:
        pass
    try:
        buffer = await checksum.resolution()
    except Exception:
        return None
    return len(buffer) if buffer is not None else None


async def compute_record_io_bytes(
    transformation_dict: Dict[str, Any], result_checksum: Checksum
) -> tuple[int | None, int | None]:
    input_total_bytes = 0
    seen: set[str] = set()
    inputs, _output_name, _output_celltype, _output_subcelltype = (
        get_transformation_inputs_output(transformation_dict)
    )
    as_ = transformation_dict.get("__as__", {})
    reverse_as = {mapped: original for original, mapped in as_.items()}
    for input_name in inputs:
        pinname = reverse_as.get(input_name, input_name)
        try:
            _celltype, _subcelltype, checksum_hex = transformation_dict[pinname]
        except Exception:
            continue
        if not checksum_hex:
            continue
        checksum_hex = Checksum(checksum_hex).hex()
        if checksum_hex in seen:
            continue
        seen.add(checksum_hex)
        length = await _checksum_length(checksum_hex)
        if length is None:
            input_total_bytes = None
            break
        input_total_bytes += length

    output_total_bytes = await _checksum_length(result_checksum.hex())
    return input_total_bytes, output_total_bytes


async def _await_buffer_writer(checksum: Checksum) -> None:
    if _buffer_writer is None:
        return
    try:
        await _buffer_writer.await_existing_task(checksum)
    except Exception:
        pass


class TransformationCache:
    """Singleton wrapper for transformation cache helpers."""

    def __init__(self) -> None:
        self._transformation_cache: dict[Checksum, Checksum] = {}
        self._rev_transformation_cache: dict[Checksum, set[Checksum]] = {}
        self._transformation_dunder_cache: dict[Checksum, Dict[str, Any]] = {}

    def _remember_transformation_dunder(
        self, tf_checksum: Checksum, tf_dunder: Dict[str, Any] | None
    ) -> None:
        if not isinstance(tf_dunder, dict) or not tf_dunder:
            return
        tf_checksum = Checksum(tf_checksum)
        self._transformation_dunder_cache[tf_checksum] = deepcopy(tf_dunder)

    def _register_transformation_result(
        self,
        tf_checksum: Checksum,
        result_checksum: Checksum,
        tf_dunder: Dict[str, Any] | None = None,
    ) -> None:
        tf_checksum = Checksum(tf_checksum)
        result_checksum = Checksum(result_checksum)
        self._remember_transformation_dunder(tf_checksum, tf_dunder)
        self._transformation_cache[tf_checksum] = result_checksum
        rev = self._rev_transformation_cache.setdefault(result_checksum, set())
        rev.add(tf_checksum)

    def get_transformation_dunder(self, tf_checksum: Checksum) -> Dict[str, Any]:
        tf_checksum = Checksum(tf_checksum)
        return deepcopy(self._transformation_dunder_cache.get(tf_checksum, {}))

    def get_reverse_transformations(self, result_checksum: Checksum) -> list[Checksum]:
        result_checksum = Checksum(result_checksum)
        rev = self._rev_transformation_cache.get(result_checksum)
        if not rev:
            return []
        return list(rev)

    async def run(
        self,
        transformation_dict: Dict[str, Any],
        *,
        tf_checksum,
        tf_dunder,
        scratch: bool,
        require_value: bool,
        force_local: bool = False,
    ) -> Checksum:
        """Run a transformation and return its result checksum.

        require_value applies to the result: when True, ensure the result checksum
        is resolvable (buffer available locally or via remote) before returning.
        """
        tf_checksum = Checksum(tf_checksum)
        self._remember_transformation_dunder(tf_checksum, tf_dunder)
        record_mode = bool(get_record())
        cached_result = self._transformation_cache.get(tf_checksum)
        if cached_result is not None:
            if require_value:
                try:
                    await cached_result.resolution()
                except CacheMissError:
                    cached_result = None
            if cached_result is not None:
                _debug(f"cache hit {tf_checksum.hex()}")
                if scratch:
                    cached_result.tempref(scratch=True)
                else:
                    if buffer_remote is not None:
                        await buffer_remote.promise(cached_result)
                    cached_result.tempref()
                self._register_transformation_result(
                    tf_checksum, cached_result, tf_dunder=tf_dunder
                )
                return cached_result

        if not force_local and database_remote is not None and not is_worker():
            _debug(f"query remote db for {tf_checksum.hex()}")
            remote_result = await database_remote.get_transformation_result(tf_checksum)
            _debug(f"remote db result {remote_result}")
            if remote_result is not None:
                if require_value:
                    try:
                        _debug("waiting for result resolution")
                        # TODO: be more lazy and only evaluate the *potential* checksum resolution
                        await remote_result.resolution()
                    except CacheMissError:
                        _debug("result resolution cache miss")
                        remote_result = None
                if remote_result is not None:
                    _debug("using remote result")
                    if scratch:
                        remote_result.tempref(scratch=True)
                    else:
                        remote_result.tempref()
                    self._register_transformation_result(
                        tf_checksum, remote_result, tf_dunder=tf_dunder
                    )
                    return remote_result

        execution = "process" if force_local else get_execution()
        record_probe = is_record_probe(transformation_dict, tf_dunder)
        meta = (
            transformation_dict.get("__meta__")
            if isinstance(transformation_dict, dict)
            else None
        )
        if isinstance(meta, dict) and meta.get("local") is True:
            execution = "process"
        remote_target = _resolve_remote_target(execution)
        if record_mode:
            if database_remote is None or not database_remote.has_write_server():
                raise RuntimeError(
                    "Record mode requires an active database write server"
                )
        _debug(
            f"execution={execution} has_spawned()={worker.has_spawned()} is_worker={is_worker()}"
        )
        started_at = _utcnow_iso()
        wall_start = time.perf_counter()
        cpu_start = os.times()
        probe_context = None
        compilation_context = None
        if execution == "remote" and not force_local:
            # NOTE: this branch is only hit if no seamless Dask client has been defined
            if jobserver_remote is None:
                raise RuntimeError(
                    "Remote execution requested but seamless_remote is not installed"
                )
            if buffer_remote is None or database_remote is None:
                raise RuntimeError(
                    "Remote execution requires hashserver and database server"
                )
            if not buffer_remote.has_write_server():
                raise RuntimeError("Remote execution requires an active hashserver")
            if not database_remote.has_write_server():
                raise RuntimeError(
                    "Remote execution requires an active database server"
                )
            _debug("dispatching transformation to remote jobserver")

            ### NOTE: flushing the entire buffer_writer queue, just to be sure that
            ###   the jobserver has it available.
            ### TODO: flush only the buffers that are required by the transformation
            ### This is not trivial in case of deep checksums
            from seamless.caching import buffer_writer

            buffer_writer.flush()
            ### /NOTE

            result_checksum = await jobserver_remote.run_transformation(
                transformation_dict,
                tf_checksum=tf_checksum,
                tf_dunder=tf_dunder,
                scratch=scratch,
            )
            if isinstance(result_checksum, dict):
                probe_context = result_checksum.get("probe_context")
                compilation_context = result_checksum.get("compilation_context")
                result_checksum = result_checksum.get("result_checksum")
            if isinstance(result_checksum, str):
                remote_job_dir = parse_remote_job_written(result_checksum)
                if remote_job_dir is not None:
                    raise RemoteJobWritten(remote_job_dir)
                raise RuntimeError(result_checksum)
            result_checksum = Checksum(result_checksum)
        elif worker.has_spawned() and not is_worker() and not force_local:
            _debug("dispatching transformation to worker pool")
            result_checksum = await worker.dispatch_to_workers(
                transformation_dict,
                tf_checksum=tf_checksum,
                tf_dunder=tf_dunder,
                scratch=scratch,
            )
            if isinstance(result_checksum, str):
                remote_job_dir = parse_remote_job_written(result_checksum)
                if remote_job_dir is not None:
                    raise RemoteJobWritten(remote_job_dir)
                raise RuntimeError(result_checksum)
            result_checksum = Checksum(result_checksum)
        elif is_worker() and not force_local:
            assert not worker.has_spawned()
            _debug("forwarding transformation request to parent")
            result_checksum = await worker.forward_to_parent(
                transformation_dict,
                tf_checksum=tf_checksum,
                tf_dunder=tf_dunder,
                scratch=scratch,
            )
            if isinstance(result_checksum, str):
                remote_job_dir = parse_remote_job_written(result_checksum)
                if remote_job_dir is not None:
                    raise RemoteJobWritten(remote_job_dir)
                raise RuntimeError(result_checksum)
            try:
                result_checksum = Checksum(result_checksum)
            except Exception as exc:
                raise RuntimeError(
                    f"Invalid checksum from parent: {result_checksum!r}"
                ) from exc
        else:
            _debug("running transformation in-process")
            loop = asyncio.get_running_loop()
            result_checksum = await loop.run_in_executor(
                None,
                run_transformation_dict,
                transformation_dict,
                tf_checksum,
                tf_dunder,
                scratch,
                require_value,
            )
            remote_job_dir = parse_remote_job_written(result_checksum)
            if remote_job_dir is not None:
                raise RemoteJobWritten(remote_job_dir)
            result_checksum = Checksum(result_checksum)

        finished_at = _utcnow_iso()
        wall_time_seconds = round(time.perf_counter() - wall_start, 6)
        cpu_end = os.times()
        cpu_user_seconds = round(cpu_end.user - cpu_start.user, 6)
        cpu_system_seconds = round(cpu_end.system - cpu_start.system, 6)

        if require_value:
            try:
                _debug("ensuring result is resolvable")
                await result_checksum.resolution()
            except Exception:
                _debug("result resolution failed; will continue")

        if scratch:
            result_checksum.tempref(scratch=True)
        else:
            result_checksum.tempref()

        if database_remote is not None and not is_worker():
            await database_remote.set_transformation_result(
                tf_checksum, result_checksum
            )
            if record_mode and not record_probe:
                if probe_context is None and (
                    execution != "remote" or remote_target != "jobserver"
                ):
                    probe_context = await ensure_record_bucket_preconditions(
                        transformation_dict,
                        tf_dunder,
                        execution=execution,
                    )
                bucket_contract_violations = await load_bucket_contract_violations(
                    probe_context
                )
                input_total_bytes, output_total_bytes = await compute_record_io_bytes(
                    transformation_dict, result_checksum
                )
                if compilation_context is None:
                    compilation_context = await build_compilation_context_checksum(
                        transformation_dict, tf_dunder
                    )
                validation_snapshot = await build_validation_snapshot_checksum(
                    transformation_dict,
                    tf_dunder,
                    execution=execution,
                    probe_context=probe_context,
                    compilation_context=compilation_context,
                    bucket_contract_violations=bucket_contract_violations,
                    job_contract_violations=[],
                )
                record = build_execution_record(
                    transformation_dict,
                    tf_checksum=tf_checksum,
                    result_checksum=result_checksum,
                    tf_dunder=tf_dunder,
                    execution=execution,
                    started_at=started_at,
                    finished_at=finished_at,
                    wall_time_seconds=wall_time_seconds,
                    cpu_user_seconds=cpu_user_seconds,
                    cpu_system_seconds=cpu_system_seconds,
                    probe_context=probe_context,
                    bucket_contract_violations=bucket_contract_violations,
                    compilation_context=compilation_context,
                    validation_snapshot=validation_snapshot,
                )
                record["execution_envelope"]["scratch"] = bool(scratch)
                record["input_total_bytes"] = input_total_bytes
                record["output_total_bytes"] = output_total_bytes
                await database_remote.set_execution_record(
                    tf_checksum, result_checksum, record
                )
            # TODO:
            #     buffer_cache.guarantee_buffer_info(
            #         result_checksum, output_celltype, sync_to_remote=True
            #     )

        self._register_transformation_result(
            tf_checksum, result_checksum, tf_dunder=tf_dunder
        )
        await _await_buffer_writer(result_checksum)

        return result_checksum

    def run_sync(
        self,
        transformation_dict: Dict[str, Any],
        *,
        tf_checksum,
        tf_dunder,
        scratch: bool,
        require_value: bool,
        force_local: bool = False,
    ) -> Checksum:
        tf_checksum = Checksum(tf_checksum)
        self._remember_transformation_dunder(tf_checksum, tf_dunder)
        cached_result = self._transformation_cache.get(tf_checksum)
        if cached_result is not None:
            if require_value:
                try:
                    cached_result.resolve()
                except CacheMissError:
                    cached_result = None
            if cached_result is not None:
                self._register_transformation_result(
                    tf_checksum, cached_result, tf_dunder=tf_dunder
                )
                if scratch:
                    cached_result.tempref(scratch=True)
                else:
                    cached_result.tempref()
                return cached_result

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                self.run(
                    transformation_dict,
                    tf_checksum=tf_checksum,
                    tf_dunder=tf_dunder,
                    scratch=scratch,
                    require_value=require_value,
                    force_local=force_local,
                ),
                loop,
            )
            return future.result()

        new_loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(
                self.run(
                    transformation_dict,
                    tf_checksum=tf_checksum,
                    tf_dunder=tf_dunder,
                    scratch=scratch,
                    require_value=require_value,
                    force_local=force_local,
                )
            )
        finally:
            if _close_all_clients is not None:
                try:
                    _close_all_clients()
                except Exception:
                    pass
            asyncio.set_event_loop(None)
            try:
                new_loop.run_until_complete(new_loop.shutdown_asyncgens())
            except Exception:
                pass
            try:
                new_loop.run_until_complete(new_loop.shutdown_default_executor())
            except Exception:
                pass
            new_loop.close()


_transformation_cache_instance: TransformationCache | None = None


def get_transformation_cache() -> TransformationCache:
    global _transformation_cache_instance
    if _transformation_cache_instance is None:
        _transformation_cache_instance = TransformationCache()
    return _transformation_cache_instance


async def run(
    transformation_dict: Dict[str, Any],
    *,
    tf_checksum,
    tf_dunder,
    scratch: bool,
    require_value: bool,
    force_local: bool = False,
) -> Checksum:
    return await get_transformation_cache().run(
        transformation_dict,
        tf_checksum=tf_checksum,
        tf_dunder=tf_dunder,
        scratch=scratch,
        require_value=require_value,
        force_local=force_local,
    )


def run_sync(
    transformation_dict: Dict[str, Any],
    *,
    tf_checksum,
    tf_dunder,
    scratch: bool,
    require_value: bool,
    force_local: bool = False,
) -> Checksum:
    return get_transformation_cache().run_sync(
        transformation_dict,
        tf_checksum=tf_checksum,
        tf_dunder=tf_dunder,
        scratch=scratch,
        require_value=require_value,
        force_local=force_local,
    )


async def recompute_from_transformation_checksum(
    tf_checksum: Checksum | str,
    *,
    scratch: bool = True,
    require_value: bool = True,
) -> Checksum | None:
    try:
        tf_checksum_obj = Checksum(tf_checksum)
        transformation_dict = await tf_checksum_obj.resolution(celltype="plain")
    except Exception:
        return None
    if not isinstance(transformation_dict, dict):
        return None
    cache = get_transformation_cache()
    tf_dunder = cache.get_transformation_dunder(tf_checksum_obj)
    return await cache.run(
        transformation_dict,
        tf_checksum=tf_checksum_obj,
        tf_dunder=tf_dunder,
        scratch=scratch,
        require_value=require_value,
        force_local=True,
    )


def recompute_from_transformation_checksum_sync(
    tf_checksum: Checksum | str,
    *,
    scratch: bool = True,
    require_value: bool = True,
) -> Checksum | None:
    try:
        tf_checksum_obj = Checksum(tf_checksum)
        transformation_dict = tf_checksum_obj.resolve(celltype="plain")
    except Exception:
        return None
    if not isinstance(transformation_dict, dict):
        return None
    cache = get_transformation_cache()
    tf_dunder = cache.get_transformation_dunder(tf_checksum_obj)
    return cache.run_sync(
        transformation_dict,
        tf_checksum=tf_checksum_obj,
        tf_dunder=tf_dunder,
        scratch=scratch,
        require_value=require_value,
        force_local=True,
    )


def get_reverse_transformations(result_checksum: Checksum) -> list[Checksum]:
    return get_transformation_cache().get_reverse_transformations(result_checksum)


def register_transformation_result(
    tf_checksum: Checksum,
    result_checksum: Checksum,
    tf_dunder: Dict[str, Any] | None = None,
) -> None:
    get_transformation_cache()._register_transformation_result(  # type: ignore[attr-defined]
        tf_checksum, result_checksum, tf_dunder=tf_dunder
    )


__all__ = [
    "TransformationCache",
    "get_transformation_cache",
    "run",
    "run_sync",
    "recompute_from_transformation_checksum",
    "recompute_from_transformation_checksum_sync",
    "build_execution_record",
    "get_reverse_transformations",
    "register_transformation_result",
]
