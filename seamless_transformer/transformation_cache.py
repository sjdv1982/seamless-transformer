"""Transformation cache helpers."""

from typing import Any, Dict

import asyncio
import concurrent.futures
from copy import deepcopy
from dataclasses import dataclass
import os
import subprocess
import time
import threading

from seamless import Buffer, CacheMissError, Checksum, is_worker

from . import record_utils as _record_utils
from . import worker
from .probe_index import ensure_record_bucket_preconditions, is_record_probe
from .record_assembly import (
    _COMPILED_VALIDATION_CACHE,
    _VALIDATION_SNAPSHOT_COUNTS,
    _canonical_path_entries,
    _canonical_sys_path,
    _current_umask,
    _determinant_env_live,
    _filesystem_facts_live,
    _gpu_policy_live,
    _normalize_job_validation_payload,
    _read_proc_self_maps_paths,
    _system_library_roots,
    build_compilation_context_checksum,
    build_execution_record,
    build_minimal_execution_record,
    build_validation_snapshot_checksum,
    collect_compilation_runtime_metadata,
    collect_job_validation,
    compute_record_io_bytes,
    load_bucket_contract_violations,
    start_gpu_memory_sampler,
    stop_gpu_memory_sampler,
)
from .record_runtime import get_record_mode
from .record_utils import (
    _memory_peak_bytes,
    _process_create_time_epoch,
    _utcnow_iso,
)
from .remote_job import RemoteJobWritten, parse_remote_job_written
from .run import run_transformation_dict
from seamless_config.select import (
    check_remote_redundancy,
    get_execution,
    get_node,
    get_queue,
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

_DEBUG = os.environ.get("SEAMLESS_DEBUG_TRANSFORMATION", "").lower() in (
    "1",
    "true",
    "yes",
)


def _debug(msg: str) -> None:
    if _DEBUG:
        print(f"[transformation_cache] {msg}", flush=True)


def _resolve_remote_target(execution: str) -> str | None:
    return _record_utils._resolve_remote_target(
        execution,
        get_remote=get_remote,
        get_selected_cluster=get_selected_cluster,
        check_remote_redundancy=check_remote_redundancy,
    )


@dataclass
class _ActiveSubmission:
    envelope_checksum: str
    future: concurrent.futures.Future
    canceled: bool = False


class TransformationCancelledError(RuntimeError):
    pass


def _dunder_envelope_checksum(
    tf_dunder: Dict[str, Any] | None, *, scratch: bool
) -> str:
    payload = {
        "tf_dunder": deepcopy(tf_dunder) if isinstance(tf_dunder, dict) else {},
        "scratch": bool(scratch),
    }
    return Buffer(payload, "plain").get_checksum().hex()


async def _await_with_active_cancellation(awaitable, active_submission):
    """Await backend work, but let checksum cancellation release the owner wait."""

    if active_submission is None:
        return await awaitable

    work_future = asyncio.ensure_future(awaitable)
    cancel_future = asyncio.wrap_future(active_submission.future)
    cancel_future.add_done_callback(_retrieve_future_exception)
    try:
        done, _pending = await asyncio.wait(
            {work_future, cancel_future}, return_when=asyncio.FIRST_COMPLETED
        )
    except asyncio.CancelledError:
        work_future.cancel()
        cancel_future.cancel()
        raise
    if cancel_future in done:
        work_future.cancel()
        try:
            await work_future
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        return cancel_future.result()

    return work_future.result()


def _retrieve_future_exception(future) -> None:
    try:
        future.exception()
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


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
        self._active_submissions: dict[Checksum, _ActiveSubmission] = {}
        self._active_lock = threading.RLock()

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
        store_execution_record: bool = True,
        strict_dunder: bool = False,
    ) -> Checksum:
        """Run a transformation and return its result checksum.

        require_value applies to the result: when True, ensure the result checksum
        is resolvable (buffer available locally or via remote) before returning.
        """
        tf_checksum = Checksum(tf_checksum)
        self._remember_transformation_dunder(tf_checksum, tf_dunder)
        record_mode = get_record_mode()
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

        return await self._run_active_or_execute(
            transformation_dict,
            tf_checksum=tf_checksum,
            tf_dunder=tf_dunder,
            scratch=scratch,
            require_value=require_value,
            force_local=force_local,
            store_execution_record=store_execution_record,
            strict_dunder=strict_dunder,
            record_mode=record_mode,
        )

    async def _run_active_or_execute(
        self,
        transformation_dict: Dict[str, Any],
        *,
        tf_checksum: Checksum,
        tf_dunder,
        scratch: bool,
        require_value: bool,
        force_local: bool,
        store_execution_record: bool,
        strict_dunder: bool,
        record_mode: bool,
    ) -> Checksum:
        envelope_checksum = _dunder_envelope_checksum(tf_dunder, scratch=scratch)
        active: _ActiveSubmission
        owner = False
        with self._active_lock:
            active = self._active_submissions.get(tf_checksum)
            if active is not None and active.future.done():
                self._active_submissions.pop(tf_checksum, None)
                active = None
            if active is not None:
                if strict_dunder and active.envelope_checksum != envelope_checksum:
                    raise RuntimeError(
                        "Transformation "
                        f"{tf_checksum.hex()} is already running with a different "
                        "dunder envelope; wait for it to finish or cancel it "
                        "before strict re-submission"
                    )
            else:
                active = _ActiveSubmission(
                    envelope_checksum=envelope_checksum,
                    future=concurrent.futures.Future(),
                )
                self._active_submissions[tf_checksum] = active
                owner = True

        if not owner:
            return await asyncio.wrap_future(active.future)

        try:
            result = await self._run_uncached(
                transformation_dict,
                tf_checksum=tf_checksum,
                tf_dunder=tf_dunder,
                scratch=scratch,
                require_value=require_value,
                force_local=force_local,
                store_execution_record=store_execution_record,
                strict_dunder=strict_dunder,
                record_mode=record_mode,
                active_submission=active,
            )
        except BaseException as exc:
            if not active.future.done():
                active.future.set_exception(exc)
            raise
        else:
            if not active.future.done():
                active.future.set_result(result)
            return result
        finally:
            with self._active_lock:
                if self._active_submissions.get(tf_checksum) is active:
                    self._active_submissions.pop(tf_checksum, None)

    def cancel_by_checksum(self, tf_checksum: Checksum | str) -> bool:
        tf_checksum = Checksum(tf_checksum)
        canceled = False
        with self._active_lock:
            active = self._active_submissions.pop(tf_checksum, None)
        if active is not None and not active.future.done():
            active.canceled = True
            active.future.set_exception(
                TransformationCancelledError("Transformation was canceled")
            )
            canceled = True
        try:
            canceled = worker.cancel_by_checksum(tf_checksum) or canceled
        except Exception:
            pass
        try:
            from seamless_dask.transformer_client import get_seamless_dask_client
        except Exception:
            dask_client = None
        else:
            dask_client = get_seamless_dask_client()
        if dask_client is not None:
            cancel = getattr(dask_client, "cancel_by_checksum", None)
            if callable(cancel):
                try:
                    canceled = bool(cancel(tf_checksum)) or canceled
                except Exception:
                    pass
        return canceled

    def transformation_status(self, tf_checksum: Checksum | str) -> str:
        tf_checksum = Checksum(tf_checksum)
        with self._active_lock:
            active = self._active_submissions.get(tf_checksum)
        if active is None:
            return "not-running"
        if active.canceled:
            return "canceled"
        if not active.future.done():
            return "running"
        try:
            active.future.result()
        except TransformationCancelledError:
            return "canceled"
        except Exception:
            return "failed"
        return "done"

    async def _run_uncached(
        self,
        transformation_dict: Dict[str, Any],
        *,
        tf_checksum: Checksum,
        tf_dunder,
        scratch: bool,
        require_value: bool,
        force_local: bool,
        store_execution_record: bool,
        strict_dunder: bool,
        record_mode: bool,
        active_submission: _ActiveSubmission | None = None,
    ) -> Checksum:
        execution = "process" if force_local else get_execution()
        meta = (
            transformation_dict.get("__meta__")
            if isinstance(transformation_dict, dict)
            else None
        )
        if isinstance(meta, dict) and meta.get("local") is True:
            execution = "process"
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
        job_validation_payload = None
        runtime_metadata = None
        gpu_memory_peak_bytes = None
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

            result_checksum = await _await_with_active_cancellation(
                jobserver_remote.run_transformation(
                    transformation_dict,
                    tf_checksum=tf_checksum,
                    tf_dunder=tf_dunder,
                    scratch=scratch,
                    strict_dunder=strict_dunder,
                ),
                active_submission,
            )
            if isinstance(result_checksum, dict):
                remote_job_written = result_checksum.get("remote_job_written")
                if isinstance(remote_job_written, str):
                    result_checksum = remote_job_written
                else:
                    probe_context = result_checksum.get("probe_context")
                    compilation_context = result_checksum.get("compilation_context")
                    job_validation_payload = result_checksum.get("job_validation")
                    runtime_metadata = result_checksum.get("record_runtime")
                    result_checksum = result_checksum.get("result_checksum")
            if isinstance(result_checksum, str):
                remote_job_dir = parse_remote_job_written(result_checksum)
                if remote_job_dir is not None:
                    raise RemoteJobWritten(remote_job_dir)
                raise RuntimeError(result_checksum)
            result_checksum = Checksum(result_checksum)
        elif worker.has_spawned() and not is_worker() and not force_local:
            _debug("dispatching transformation to worker pool")
            result_checksum = await _await_with_active_cancellation(
                worker.dispatch_to_workers(
                    transformation_dict,
                    tf_checksum=tf_checksum,
                    tf_dunder=tf_dunder,
                    scratch=scratch,
                    strict_dunder=strict_dunder,
                ),
                active_submission,
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
            result_checksum = await _await_with_active_cancellation(
                worker.forward_to_parent(
                    transformation_dict,
                    tf_checksum=tf_checksum,
                    tf_dunder=tf_dunder,
                    scratch=scratch,
                    strict_dunder=strict_dunder,
                ),
                active_submission,
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
            gpu_sampler = start_gpu_memory_sampler()
            try:
                result_checksum = await _await_with_active_cancellation(
                    loop.run_in_executor(
                        None,
                        run_transformation_dict,
                        transformation_dict,
                        tf_checksum,
                        tf_dunder,
                        scratch,
                        require_value,
                    ),
                    active_submission,
                )
            finally:
                gpu_memory_peak_bytes = stop_gpu_memory_sampler(gpu_sampler)
            remote_job_dir = parse_remote_job_written(result_checksum)
            if remote_job_dir is not None:
                raise RemoteJobWritten(remote_job_dir)
            result_checksum = Checksum(result_checksum)

        finished_at = _utcnow_iso()
        wall_time_seconds = round(time.perf_counter() - wall_start, 6)
        cpu_end = os.times()
        cpu_user_seconds = round(cpu_end.user - cpu_start.user, 6)
        cpu_system_seconds = round(cpu_end.system - cpu_start.system, 6)
        if isinstance(runtime_metadata, dict):
            started_at = runtime_metadata.get("started_at", started_at)
            finished_at = runtime_metadata.get("finished_at", finished_at)
            wall_time_seconds = runtime_metadata.get(
                "wall_time_seconds", wall_time_seconds
            )
            cpu_user_seconds = runtime_metadata.get(
                "cpu_user_seconds", cpu_user_seconds
            )
            cpu_system_seconds = runtime_metadata.get(
                "cpu_system_seconds", cpu_system_seconds
            )

        if require_value:
            try:
                _debug("ensuring result is resolvable")
                await result_checksum.resolution()
            except Exception:
                _debug("result resolution failed; will continue")

        if active_submission is not None and active_submission.canceled:
            raise TransformationCancelledError("Transformation was canceled")

        if scratch:
            result_checksum.tempref(scratch=True)
        else:
            result_checksum.tempref()

        if database_remote is not None and not is_worker():
            await database_remote.set_transformation_result(
                tf_checksum, result_checksum
            )
            record_probe = is_record_probe(transformation_dict, tf_dunder)
            if store_execution_record and not record_probe:
                record_runtime_metadata = dict(runtime_metadata or {})
                record_runtime_metadata.setdefault(
                    "memory_peak_bytes", _memory_peak_bytes()
                )
                if gpu_memory_peak_bytes is not None:
                    record_runtime_metadata.setdefault(
                        "gpu_memory_peak_bytes", gpu_memory_peak_bytes
                    )
                if record_mode:
                    remote_target = _resolve_remote_target(execution)
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
                    if compilation_context is None:
                        compilation_context = await build_compilation_context_checksum(
                            transformation_dict, tf_dunder
                        )
                    if execution == "remote" and remote_target == "jobserver":
                        job_validation = _normalize_job_validation_payload(
                            job_validation_payload
                        )
                    else:
                        job_validation = await collect_job_validation(
                            transformation_dict,
                            tf_dunder,
                            compilation_context=compilation_context,
                            probe_context=probe_context,
                        )
                        job_validation = _normalize_job_validation_payload(
                            job_validation
                        )
                    job_contract_violations = job_validation[
                        "job_contract_violations"
                    ]
                    record_runtime_metadata.setdefault(
                        "process_create_time_epoch", _process_create_time_epoch()
                    )
                    if "compilation_time_seconds" not in record_runtime_metadata:
                        record_runtime_metadata.update(
                            await collect_compilation_runtime_metadata(
                                transformation_dict, tf_dunder
                            )
                        )
                    input_total_bytes, output_total_bytes = (
                        await compute_record_io_bytes(
                            transformation_dict, result_checksum
                        )
                    )
                    validation_snapshot = await build_validation_snapshot_checksum(
                        transformation_dict,
                        tf_dunder,
                        execution=execution,
                        probe_context=probe_context,
                        compilation_context=compilation_context,
                        bucket_contract_violations=bucket_contract_violations,
                        job_contract_violations=job_contract_violations,
                        job_validation_diagnostics=job_validation["diagnostics"],
                        runtime_metadata=record_runtime_metadata,
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
                        job_contract_violations=job_contract_violations,
                        compilation_context=compilation_context,
                        validation_snapshot=validation_snapshot,
                        runtime_metadata=record_runtime_metadata,
                    )
                    record["execution_envelope"]["scratch"] = bool(scratch)
                    record["input_total_bytes"] = input_total_bytes
                    record["output_total_bytes"] = output_total_bytes
                else:
                    record = build_minimal_execution_record(
                        tf_checksum=tf_checksum,
                        result_checksum=result_checksum,
                        execution=execution,
                        wall_time_seconds=wall_time_seconds,
                        cpu_user_seconds=cpu_user_seconds,
                        cpu_system_seconds=cpu_system_seconds,
                        runtime_metadata=record_runtime_metadata,
                    )
                await database_remote.set_execution_record(
                    tf_checksum, result_checksum, record
                )
            # TODO:
            #     buffer_cache.guarantee_buffer_info(
            #         result_checksum, output_celltype, sync_to_remote=True
            #     )

        if active_submission is not None and active_submission.canceled:
            raise TransformationCancelledError("Transformation was canceled")

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
        store_execution_record: bool = True,
        strict_dunder: bool = False,
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
                    store_execution_record=store_execution_record,
                    strict_dunder=strict_dunder,
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
                    store_execution_record=store_execution_record,
                    strict_dunder=strict_dunder,
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
    store_execution_record: bool = True,
    strict_dunder: bool = False,
) -> Checksum:
    return await get_transformation_cache().run(
        transformation_dict,
        tf_checksum=tf_checksum,
        tf_dunder=tf_dunder,
        scratch=scratch,
        require_value=require_value,
        force_local=force_local,
        store_execution_record=store_execution_record,
        strict_dunder=strict_dunder,
    )


async def is_cached(tf_checksum: Checksum | str) -> bool:
    """Return whether a transformation checksum is present in the database cache."""
    if database_remote is None:
        raise RuntimeError(
            "Transformation.is_cached() requires seamless-remote to be installed"
        )
    if not database_remote.has_read_server():
        raise RuntimeError(
            "Transformation.is_cached() requires seamless.config.init() "
            "and an active database read server"
        )
    result = await database_remote.get_transformation_result(Checksum(tf_checksum))
    return result is not None


def is_cached_sync(tf_checksum: Checksum | str) -> bool:
    """Synchronous wrapper for database cache lookup."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        raise RuntimeError(
            "Cannot block on is_cached() from within the running event loop. "
            "Use 'await tf.is_cached_async()' instead."
        )

    new_loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(new_loop)
        return new_loop.run_until_complete(is_cached(tf_checksum))
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


def run_sync(
    transformation_dict: Dict[str, Any],
    *,
    tf_checksum,
    tf_dunder,
    scratch: bool,
    require_value: bool,
    force_local: bool = False,
    store_execution_record: bool = True,
    strict_dunder: bool = False,
) -> Checksum:
    return get_transformation_cache().run_sync(
        transformation_dict,
        tf_checksum=tf_checksum,
        tf_dunder=tf_dunder,
        scratch=scratch,
        require_value=require_value,
        force_local=force_local,
        store_execution_record=store_execution_record,
        strict_dunder=strict_dunder,
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
        store_execution_record=False,
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
        store_execution_record=False,
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
    "build_compilation_context_checksum",
    "get_transformation_cache",
    "is_cached",
    "is_cached_sync",
    "run",
    "run_sync",
    "recompute_from_transformation_checksum",
    "recompute_from_transformation_checksum_sync",
    "build_execution_record",
    "build_minimal_execution_record",
    "build_validation_snapshot_checksum",
    "collect_compilation_runtime_metadata",
    "collect_job_validation",
    "compute_record_io_bytes",
    "get_reverse_transformations",
    "load_bucket_contract_violations",
    "register_transformation_result",
    "start_gpu_memory_sampler",
    "stop_gpu_memory_sampler",
]
