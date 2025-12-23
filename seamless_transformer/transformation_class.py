"""Transformation class ported from the legacy Seamless direct API.

This version relies on core Seamless primitives (Buffer, Checksum, CacheMissError)
for checksum/buffer handling. Workflow-specific helpers that are not present in
this repository (cancel/undo/execution_metadata, fingertip resolution, etc.)
are intentionally omitted or stubbed.
"""

from __future__ import annotations

import asyncio
import threading
import traceback
import concurrent.futures as _cf
from typing import Any, Dict, Generic, Optional, TYPE_CHECKING, TypeVar

from seamless import Checksum, Buffer, ensure_open, is_worker
from seamless.util.get_event_loop import get_event_loop
from .transformation_utils import tf_get_buffer
from . import worker

try:  # Optional Dask integration
    from seamless_dask.transformation_mixin import TransformationDaskMixin
    from seamless_dask.transformer_client import get_seamless_dask_client
except Exception:  # pragma: no cover - allow operation without seamless-dask

    class TransformationDaskMixin:  # type: ignore
        _dask_futures = None

        def _dask_client(self):
            raise RuntimeError("Dask integration is unavailable")

        def _compute_with_dask(self, require_value: bool):
            raise RuntimeError("Dask integration is unavailable")

        async def _compute_with_dask_async(self, require_value: bool):
            raise RuntimeError("Dask integration is unavailable")

        def _ensure_dask_futures(self, *args, **kwargs):
            raise RuntimeError("Dask integration is unavailable")

    def get_seamless_dask_client():
        return None


T = TypeVar("T")


if TYPE_CHECKING:
    from .pretransformation import PreTransformation
    from seamless_dask.client import SeamlessDaskClient


class TransformationError(RuntimeError):
    pass


def running_in_jupyter() -> bool:
    """Function to detect Jupyter-like environments:

    - That have default running event loop. This prevents
    sync evaluation because that blocks on coroutines running in the same loop

    - That support top-level await as a go-to alternative
    """
    import sys
    import asyncio

    try:
        import pyodide

        return True
    except ImportError:
        pass

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running() and hasattr(loop, "_asyncio_runner"):
            return True
    except Exception:
        return False

    try:
        from IPython import get_ipython
    except ImportError:
        return False

    ip = get_ipython()
    if ip is None:
        return False
    # ipykernel shells: ZMQInteractiveShell, module ipykernel.zmqshell
    return (
        "ipykernel" in sys.modules
        and hasattr(ip, "kernel")
        and type(ip).__name__ == "ZMQInteractiveShell"
    )


def loop_is_nested(loop: asyncio.AbstractEventLoop) -> bool:
    return getattr(loop, "_nest_patched", False)


_LOOP_THREADS: dict[asyncio.AbstractEventLoop, threading.Thread] = {}
_COMPUTE_EXECUTOR = _cf.ThreadPoolExecutor()


def _ensure_loop_running(loop: asyncio.AbstractEventLoop) -> None:
    """Run the given event loop in a background thread if it is not already running."""

    if loop.is_running():
        return
    thread = _LOOP_THREADS.get(loop)
    if thread and thread.is_alive():
        return

    def _runner():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=_runner, daemon=True, name="seamless-loop")
    _LOOP_THREADS[loop] = thread
    thread.start()


class Transformation(TransformationDaskMixin, Generic[T]):
    """Resolve and evaluate transformation checksums, sync or async.

    Lifecycle:
    - .construct() / await .construction():
        Evaluate dependencies and builds the transformation dict
    - .compute() / await .computation() :
        Run the transformation dict and return the result checksum.
    - .run() /  await .task() :
        compute, then resolve the result checksum into a value.
    """

    # Task that drives _computation() (dependencies + evaluation)
    _computation_task: Optional[asyncio.Task] = None

    def __init__(
        self,
        result_celltype,
        constructor_sync,
        constructor_async,
        evaluator_sync,
        evaluator_async,
        upstream_dependencies: Dict[str, "Transformation"] | None = None,
        *,
        destructor=None,
        meta=None,
        pretransformation: "PreTransformation | None" = None,
        tf_dunder: dict | None = None,
    ) -> None:
        self._result_celltype = result_celltype
        self._upstream_dependencies = (upstream_dependencies or {}).copy()
        self._constructor_sync = constructor_sync
        self._constructor_async = constructor_async
        self._transformation_checksum: Optional[Checksum] = None
        self._constructed = False
        self._scratch = False

        self._evaluator_sync = evaluator_sync
        self._evaluator_async = evaluator_async
        self._result_checksum: Optional[Checksum] = None
        self._evaluated = False
        self._exception = None
        self._meta = meta
        self._destructor = destructor
        self._pretransformation = pretransformation
        self._tf_dunder = tf_dunder or {}
        self._dask_futures: TransformationFutures | None = None
        self._computation_task: Optional[asyncio.Task] = None
        self._computation_future: Optional[asyncio.Future] = None

    def _prefer_local_execution(self) -> bool:
        try:
            return self._meta is not None and self._meta.get("local") is True
        except Exception:
            return False

    @property
    def scratch(self) -> bool:
        return self._scratch

    @scratch.setter
    def scratch(self, value: bool) -> None:
        self._scratch = value

    def construct(self) -> Checksum | None:
        """
        Evaluate dependencies and calculate the transformation checksum from the inputs
        In case of failure, set .exception and return None
        """
        ensure_open("transformation construct")
        if self._constructed:
            return self._transformation_checksum
        try:
            tf_checksum_raw = self._constructor_sync(self)
            if tf_checksum_raw is None:
                raise ValueError("Cannot obtain transformation checksum")
            tf_checksum = Checksum(tf_checksum_raw)
            self._transformation_checksum = tf_checksum
        except (AssertionError, TransformationError):
            self._exception = traceback.format_exc().strip("\n") + "\n"
        except Exception:
            self._exception = traceback.format_exc(limit=0).strip("\n") + "\n"
        finally:
            self._constructed = True
        return self._transformation_checksum

    async def construction(self) -> Checksum | None:
        """
        Evaluate dependencies and calculate the transformation checksum from the inputs
        In case of failure, set .exception and return None
        """
        ensure_open("transformation construction")
        if self._constructed:
            return self._transformation_checksum
        try:
            tf_checksum_raw = await self._constructor_async(self)
            if tf_checksum_raw is None:
                raise ValueError("Cannot obtain transformation checksum")
            tf_checksum = Checksum(tf_checksum_raw)
            self._transformation_checksum = tf_checksum
        except (AssertionError, TransformationError):
            self._exception = traceback.format_exc().strip("\n") + "\n"
        except Exception:
            self._exception = traceback.format_exc(limit=0).strip("\n") + "\n"
        finally:
            self._constructed = True
        return self._transformation_checksum

    def _evaluate(self) -> Checksum | None:
        if self._evaluated:
            return self._result_checksum
        self.construct()
        if self._exception is not None:
            return None
        try:
            result_checksum_raw = self._evaluator_sync(self, require_value=True)
            if result_checksum_raw is None:
                raise ValueError("Result is empty")
            result_checksum = Checksum(result_checksum_raw)
            self._result_checksum = result_checksum
            self._exception = None
        except (AssertionError, TransformationError):
            self._exception = traceback.format_exc().strip("\n") + "\n"
        except Exception:
            self._exception = traceback.format_exc(limit=0).strip("\n") + "\n"
        finally:
            self._evaluated = True
        return self._result_checksum

    async def _evaluation(self, require_value: bool) -> Checksum | None:
        if self._evaluated:
            return self._result_checksum
        await self.construction()
        if self._exception is not None:
            return None
        try:
            result_checksum_raw = await self._evaluator_async(
                self, require_value=require_value
            )
            if result_checksum_raw is None:
                raise ValueError("Result is empty")
            result_checksum = Checksum(result_checksum_raw)
            self._result_checksum = result_checksum
            self._exception = None
        except (AssertionError, TransformationError):
            self._exception = traceback.format_exc().strip("\n") + "\n"
        except Exception:
            self._exception = traceback.format_exc(limit=0).strip("\n") + "\n"
        finally:
            self._evaluated = True
        return self._result_checksum

    def _run_dependencies(self) -> None:
        all_evaluated = True
        for depname, dep in self._upstream_dependencies.items():
            if not dep._evaluated:
                all_evaluated = False
        if all_evaluated:
            return
        try:
            loop = get_event_loop()
            self._verify_sync_construct(loop)
            self.start(loop=loop)
            for depname, dep in self._upstream_dependencies.items():
                dep.start(loop=loop)
            for depname, dep in self._upstream_dependencies.items():
                dep.compute()
                if dep.exception is not None:
                    msg = "Dependency '{}' has an exception:\n{}"
                    raise RuntimeError(msg.format(depname, dep.exception))
        except (AssertionError, TransformationError):
            self._exception = traceback.format_exc().strip("\n") + "\n"
        except Exception:
            self._exception = traceback.format_exc(limit=0).strip("\n") + "\n"

    async def _run_dependencies_async(self, require_value: bool) -> None:
        tasks = {}
        loop = get_event_loop()
        for depname, dep in self._upstream_dependencies.items():
            tasks[depname] = loop.create_task(
                dep.computation(require_value=require_value)
            )
        if tasks:
            await asyncio.gather(*tasks.values(), return_exceptions=True)
        for task in tasks.values():
            self._future_cleanup(task)
        try:
            for depname in tasks:
                dep = self._upstream_dependencies[depname]
                if dep.exception is not None:
                    msg = "Dependency '{}' has an exception:\n{}"
                    raise RuntimeError(msg.format(depname, dep.exception))
        except (AssertionError, TransformationError):
            self._exception = traceback.format_exc().strip("\n") + "\n"
        except Exception:
            self._exception = traceback.format_exc(limit=0).strip("\n") + "\n"

    @property
    def meta(self):
        return self._meta

    @meta.setter
    def meta(self, meta):
        if self._meta is None:
            self._meta = {}
        self._meta.update(meta)
        for k in list(self._meta.keys()):
            if self._meta[k] is None:
                self._meta.pop(k)
        return self._meta

    def _verify_sync(self, task_loop0, err_msg: str, jupyter_err_msg: str):
        task_loops: dict[str | None, Any] = {None: task_loop0}
        for depname, dep in self._upstream_dependencies.items():
            assert isinstance(dep, Transformation)
            dep_task_loop = None
            if dep._computation_future is not None:
                try:
                    dep_task_loop = dep._computation_future.get_loop()
                except (
                    AttributeError
                ):  # not an asyncio Future, but a concurrent.futures Future
                    pass
            elif dep._computation_task is not None:
                dep_task_loop = dep._computation_task.get_loop()
            if dep_task_loop is not None:
                task_loops[depname] = dep_task_loop

        for dep_name, task_loop in task_loops.items():
            if task_loop is None:
                continue
            if task_loop.is_running():
                if loop_is_nested(task_loop):
                    # nest_asyncio is running. Deadlocks are now the user's responsibility, not ours
                    return
                try:
                    running_loop = asyncio.get_running_loop()
                except RuntimeError:
                    running_loop = None
                if running_loop is task_loop:
                    err = jupyter_err_msg if running_in_jupyter() else err_msg
                    if dep_name is not None:
                        err = f"Dependency {dep_name}: " + err
                    raise RuntimeError(err)

    def _verify_sync_construct(self, task_loop):
        return self._verify_sync(
            task_loop0=task_loop,
            err_msg="Cannot block on construct() from within the running event loop",
            jupyter_err_msg="'tf.construct()' is not supported within Jupyter. Use 'await tf.construction()' instead.",
        )

    def _verify_sync_compute(self, task_loop):
        return self._verify_sync(
            task_loop0=task_loop,
            err_msg="Cannot block on compute() from within the running event loop",
            jupyter_err_msg="'tf.compute()' is not supported within Jupyter. Use 'await tf.computation()' instead.",
        )

    def _verify_sync_run(self, task_loop):
        return self._verify_sync(
            task_loop0=task_loop,
            err_msg="Cannot block on run() from within the running event loop",
            jupyter_err_msg="'tf.run()' is not supported within Jupyter. Use 'await tf.task()' instead.",
        )

    def _verify_sync_call(self, task_loop):
        return self._verify_sync(
            task_loop0=task_loop,
            err_msg="Cannot block on func() from within the running event loop",
            jupyter_err_msg="""'func()' is not supported within Jupyter. Use instead:"

    from seamless.transformer import delayed
    await delayed(func).task()

""",
        )

    def _compute(self, api_origin: str) -> Checksum | None:
        ensure_open("transformation compute")
        if self._evaluated:
            return self._result_checksum
        if self._computation_task is None and self._computation_future is None:
            dask_client = get_seamless_dask_client()
            if dask_client is not None and not self._prefer_local_execution():
                return self._compute_with_dask(require_value=True)
            task_loop = get_event_loop()
            self.start(loop=task_loop)

        if self._computation_future is not None:
            assert self._computation_task is None
            try:
                task_loop = self._computation_future.get_loop()
            except (
                AttributeError
            ):  # not an asyncio Future, but a concurrent.futures Future
                task_loop = None
        else:
            assert self._computation_task is not None
            task_loop = self._computation_task.get_loop()

        if api_origin == "call":
            self._verify_sync_call(task_loop)
        elif api_origin == "run":
            self._verify_sync_run(task_loop)
        else:
            self._verify_sync_compute(task_loop)

        if self._computation_future is not None:
            self._computation_future.result()
            self._computation_future = None
            return self._result_checksum

        assert self._computation_task is not None
        if task_loop.is_running():
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None
            if not loop_is_nested(running_loop):
                assert running_loop is not task_loop  # must have been detected earlier

            async def _await_task(task):
                return await task

            fut = asyncio.run_coroutine_threadsafe(
                _await_task(self._computation_task), task_loop
            )
            fut.result()
            self._computation_task = None
            return self._result_checksum

        assert self._computation_task is not None
        task_loop.run_until_complete(self._computation_task)
        self._computation_task = None
        return self._result_checksum

    def compute(self) -> Checksum | None:
        """Run the transformation and return the checksum.

        First, constructs the transformation; then, evaluate its result.
        Returns the result checksum
        In case of failure, set .exception and return None.

        It is made sure that the result value will be available upon Checksum.resolve().
        """
        return self._compute(api_origin="compute")

    async def _computation(self, require_value: bool) -> Checksum | None:
        dask_client = get_seamless_dask_client()
        if dask_client is not None and not self._prefer_local_execution():
            return await self._compute_with_dask_async(require_value=require_value)
        await self._run_dependencies_async(require_value=require_value)
        await self._evaluation(require_value=require_value)
        return self._result_checksum

    def _compute_in_thread(self, require_value: bool) -> None:
        """Run the computation in a dedicated event loop in a worker thread."""

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._computation(require_value=require_value))
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()

    async def computation(self, require_value: bool = False) -> Checksum | None:
        """Run the transformation and return the checksum.

        First, constructs the transformation; then, evaluate its result.
        Returns the result checksum
        In case of failure, set .exception and return None.

        If require_value is True, it is made sure that the value will be available too.
        (If only the checksum is available, the transformation will be recomputed.)
        """
        ensure_open("transformation computation")
        dask_client = get_seamless_dask_client()
        if dask_client is not None and not self._prefer_local_execution():
            if self._computation_task is not None:
                await self._computation_task
                self._computation_task = None
                return self._result_checksum
            return await self._compute_with_dask_async(require_value=require_value)
        if self._computation_future is not None:
            await asyncio.wrap_future(self._computation_future)
            self._computation_future = None
            return self._result_checksum
        if self._computation_task is None:
            return await self._computation(require_value=require_value)
        task_loop = self._computation_task.get_loop()
        if task_loop.is_running():
            if asyncio.get_running_loop() is task_loop:
                await self._computation_task
            else:

                async def _await_task(task):
                    return await task

                fut = asyncio.run_coroutine_threadsafe(
                    _await_task(self._computation_task), task_loop
                )
                await asyncio.wrap_future(fut)
        else:
            task_loop.run_until_complete(self._computation_task)
        self._computation_task = None
        return self._result_checksum

    def _future_cleanup(self, fut) -> None:
        """Swallow task exceptions to avoid 'Task exception was never retrieved'."""
        try:
            fut.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    def start(
        self, *, loop: asyncio.AbstractEventLoop | None = None
    ) -> "Transformation[T]":
        """Ensure the computation task is scheduled; return self for chaining."""
        ensure_open("transformation start")
        for _depname, dep in self._upstream_dependencies.items():
            dep.start()
        dask_client = get_seamless_dask_client()
        if dask_client is not None and not self._prefer_local_execution():
            if self._computation_task is None:
                loop = loop or get_event_loop()
                self._computation_task = loop.create_task(
                    self._compute_with_dask_async(require_value=True)
                )
                self._computation_task.add_done_callback(self._future_cleanup)
            return self
        if self._computation_task is not None or self._computation_future is not None:
            return self
        if is_worker() or worker.has_spawned():
            # Offload to a thread with its own event loop for concurrency.
            self._computation_future = _COMPUTE_EXECUTOR.submit(
                self._compute_in_thread, True
            )
            return self
        loop = loop or get_event_loop()
        _ensure_loop_running(loop)
        self._computation_task = loop.create_task(self._computation(require_value=True))
        self._computation_task.add_done_callback(self._future_cleanup)
        return self

    @property
    def transformation_checksum(self) -> Checksum:
        if self._transformation_checksum is None:
            if self._exception is not None:
                raise TransformationError(
                    "Transformation construction returned an exception"
                )
            raise TransformationError("Transformation has not been constructed")
        return self._transformation_checksum

    @property
    def result_checksum(self) -> Checksum:
        if self._transformation_checksum is None:
            if self._exception is not None:
                raise TransformationError(
                    "Transformation construction returned an exception:\n"
                    + self._exception
                )
            raise TransformationError("Transformation has not been constructed")
        if self._result_checksum is not None:
            # If a result exists, prefer it and clear any stale exception.
            self._exception = None
            return self._result_checksum

        if self._exception is not None:
            raise TransformationError(
                "Transformation returned an exception:\n" + self._exception
            )

        if self._computation_task is not None:
            task = self._computation_task
            if task.done():
                self._future_cleanup(task)
                self._computation_task = None
                if self._result_checksum is not None:
                    return self._result_checksum
                if self._exception is not None:
                    raise TransformationError(
                        "Transformation returned an exception:\n" + self._exception
                    )
            raise TransformationError("Transformation is still computing")

        raise TransformationError("Transformation has not been computed")

    @property
    def buffer(self) -> Buffer:
        buf = self.result_checksum.resolve()
        assert isinstance(buf, Buffer)
        return buf

    @property
    def value(self) -> T:
        buf = self.buffer
        return buf.get_value(self.celltype)

    async def _run(self) -> T:
        await self.computation(require_value=True)
        checksum = self.result_checksum  # Will raise an exception if there is one
        buf = await checksum.resolution()
        assert isinstance(buf, Buffer)
        return await buf.get_value_async(self.celltype)

    def task(self) -> asyncio.Task[T]:
        """Create a Task Run the transformation and returns the result,

        First runs .compute, then resolve the result checksum into a value.
        Raise RuntimeError in case of an exception."""
        ensure_open("transformation task")
        return get_event_loop().create_task(self._run())

    def run(self) -> T:
        """Run the transformation and returns the result,

        First runs .compute, then resolve the result checksum into a value.
        Raise RuntimeError in case of an exception."""

        ensure_open("transformation run")
        self._compute(api_origin="run")
        return self.value

    @property
    def celltype(self):
        return self._result_celltype

    @property
    def exception(self):
        return self._exception

    @property
    def logs(self):
        raise NotImplementedError("transformation logs are not ported in this codebase")

    def clear_exception(self) -> None:
        self._exception = None
        if self._constructed and self._transformation_checksum is None:
            self._constructed = False
        elif self._evaluated and self._result_checksum is None:
            self._evaluated = False

    @property
    def status(self) -> str:
        try:
            if self._exception is not None:
                return "Status: exception"
            if self._evaluated:
                assert self._result_checksum is not None
                return "Status: OK"
            if self._computation_task is not None:
                return "Status: pending"
            return "Status: ready"
        except Exception:
            return "Status: unknown exception"

    def __del__(self):
        if self._destructor is not None:
            self._destructor(self)
        try:
            from seamless_dask.permissions import shutdown_permission_manager

            shutdown_permission_manager()
        except Exception:
            pass


def transformation_from_pretransformation(
    pre_transformation: "PreTransformation",
    *,
    upstream_dependencies: dict,
    meta: dict,
    scratch: bool,
    tf_dunder: dict | None = None,
) -> Transformation[T]:
    """Build a Transformation from a PreTransformation"""
    from .transformation_utils import tf_get_buffer
    from .transformation_cache import run_sync, run

    tf_dunder = tf_dunder or {}

    def constructor_sync(transformation_obj):  # pylint: disable=unused-argument
        nonlocal tf_dunder
        transformation_obj._run_dependencies()
        if transformation_obj.exception is not None:
            raise RuntimeError(transformation_obj.exception)
        pre_transformation.prepare_transformation()
        tf_buffer = tf_get_buffer(pre_transformation.pretransformation_dict)
        tf_checksum = tf_buffer.get_checksum()
        tf_buffer.tempref()
        ### tf_dunder = extract_dunder(pre_transformation.pretransformation_dict)
        # TODO: populate tf_dunder from transformation metadata if needed

        return tf_checksum

    async def constructor_async(transformation_obj):
        await transformation_obj._run_dependencies_async(require_value=True)
        if transformation_obj.exception is not None:
            raise RuntimeError(transformation_obj.exception)
        return constructor_sync(transformation_obj)

    def evaluator_sync(
        transformation_obj: Transformation, require_value: bool
    ) -> Checksum:
        # Currently, require_value will always be true for sync evaluation
        scratch = False if require_value else transformation_obj.scratch
        return run_sync(
            pre_transformation.pretransformation_dict,
            tf_checksum=transformation_obj.transformation_checksum,
            tf_dunder=tf_dunder,
            scratch=scratch,
            require_fingertip=require_value,
        )

    async def evaluator_async(transformation_obj, require_value: bool) -> Checksum:
        scratch = False if require_value else transformation_obj.scratch
        return await run(
            pre_transformation.pretransformation_dict,
            tf_checksum=transformation_obj.transformation_checksum,
            tf_dunder=tf_dunder,
            scratch=scratch,
            require_fingertip=require_value,
        )

    def destructor(transformation_obj):  # pylint: disable=unused-argument
        pre_transformation.release()

    tf = Transformation(
        pre_transformation.result_celltype,
        constructor_sync,
        constructor_async,
        evaluator_sync,
        evaluator_async,
        upstream_dependencies=upstream_dependencies,
        meta=meta,
        destructor=destructor,
        pretransformation=pre_transformation,
        tf_dunder=tf_dunder,
    )
    tf.scratch = scratch
    return tf


__all__ = ["Transformation", "transformation_from_pretransformation"]
