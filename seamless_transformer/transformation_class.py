"""Transformation class ported from the legacy Seamless direct API.

This version relies on core Seamless primitives (Buffer, Checksum, CacheMissError)
for checksum/buffer handling. Workflow-specific helpers that are not present in
this repository (cancel/undo/execution_metadata, fingertip resolution, etc.)
are intentionally omitted or stubbed.
"""

from __future__ import annotations

import asyncio
import traceback
from typing import Dict, Optional, TYPE_CHECKING

from seamless import Checksum, Buffer

if TYPE_CHECKING:
    from .pretransformation import PreTransformation


class TransformationError(RuntimeError):
    pass


def get_event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

    return loop


class Transformation:
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
        if self._constructed:
            return self._transformation_checksum
        try:
            tf_checksum_raw = self._constructor_sync(self)
            if tf_checksum_raw is None:
                raise ValueError("Cannot obtain transformation checksum")
            tf_checksum = Checksum(tf_checksum_raw)
            self._transformation_checksum = tf_checksum
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
        if self._constructed:
            return self._transformation_checksum
        try:
            tf_checksum_raw = await self._constructor_async(self)
            if tf_checksum_raw is None:
                raise ValueError("Cannot obtain transformation checksum")
            tf_checksum = Checksum(tf_checksum_raw)
            self._transformation_checksum = tf_checksum
        except AssertionError:
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
        except Exception:
            self._exception = traceback.format_exc(limit=0).strip("\n") + "\n"
        finally:
            self._evaluated = True
        return self._result_checksum

    def _run_dependencies(self) -> None:
        try:
            loop = get_event_loop()
            self.start(loop=loop)
            for depname, dep in self._upstream_dependencies.items():
                dep.compute()
                if dep.exception is not None:
                    msg = "Dependency '{}' has an exception:\n{}"
                    raise RuntimeError(msg.format(depname, dep.exception))
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

    def compute(self) -> Checksum | None:
        """Run the transformation and return the checksum.

        First, constructs the transformation; then, evaluate its result.
        Returns the result checksum
        In case of failure, set .exception and return None.

        It is made sure that the result value will be available upon Checksum.resolve().
        """
        if self._evaluated:
            return self._result_checksum
        if self._computation_task is None:
            task_loop = get_event_loop()
            self.start(loop=task_loop)
        else:
            task_loop = self._computation_task.get_loop()

        if not task_loop.is_running():
            assert self._computation_task is not None
            task_loop.run_until_complete(self._computation_task)
            self._computation_task = None
        else:
            self._run_dependencies()
            if self._exception is None:
                self._evaluate()
            if self._computation_task is not None:
                self._computation_task.cancel()
                self._computation_task = None
        return self._result_checksum

    async def _computation(self, require_value: bool) -> Checksum | None:
        await self._run_dependencies_async(require_value=require_value)
        await self._evaluation(require_value=require_value)
        return self._result_checksum

    async def computation(self, require_value: bool = False) -> Checksum | None:
        """Run the transformation and return the checksum.

        First, constructs the transformation; then, evaluate its result.
        Returns the result checksum
        In case of failure, set .exception and return None.

        If require_value is True, it is made sure that the value will be available too.
        (If only the checksum is available, the transformation will be recomputed.)
        """
        if self._computation_task is not None:
            await self._computation_task
            return self._result_checksum
        else:
            return await self._computation(require_value=require_value)

    def _future_cleanup(self, fut) -> None:
        """Swallow task exceptions to avoid 'Task exception was never retrieved'."""
        try:
            fut.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    def start(self, *, loop: asyncio.AbstractEventLoop | None = None) -> "Transformation":
        """Ensure the computation task is scheduled; return self for chaining."""
        for _depname, dep in self._upstream_dependencies.items():
            dep.start()
        if self._computation_task is not None:
            return self
        loop = loop or get_event_loop()
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
        if self._result_checksum is None:
            if self._computation_task is not None:
                raise TransformationError("Transformation is still computing")
            else:
                assert self._exception is not None
                raise TransformationError(
                    "Transformation returned an exception:\n" + self._exception
                )
        else:
            assert self._exception is None
        return self._result_checksum

    @property
    def buffer(self) -> Buffer:
        buf = self.result_checksum.resolve()
        assert isinstance(buf, Buffer)
        return buf

    @property
    def value(self):
        buf = self.buffer
        return buf.get_value(self.celltype)

    async def _run(self):
        await self.computation(require_value=True)
        checksum = self.result_checksum  # Will raise an exception if there is one
        buf = await checksum.resolution()
        assert isinstance(buf, Buffer)
        return await buf.get_value_async(self.celltype)

    def task(self) -> asyncio.Task:
        """Create a Task Run the transformation and returns the result,

        First runs .compute, then resolve the result checksum into a value.
        Raise RuntimeError in case of an exception."""
        return get_event_loop().create_task(self._run())

    def run(self):
        """Run the transformation and returns the result,

        First runs .compute, then resolve the result checksum into a value.
        Raise RuntimeError in case of an exception."""

        self.compute()
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


def transformation_from_pretransformation(
    pre_transformation: "PreTransformation",
    *,
    upstream_dependencies: dict,
    meta: dict,
    scratch: bool,
) -> Transformation:
    """Build a Transformation from a PreTransformation"""
    from .transformation_utils import tf_get_buffer
    from .transformation_cache import run_sync, run

    tf_dunder = {}

    def constructor_sync(transformation_obj):  # pylint: disable=unused-argument
        nonlocal tf_dunder
        pre_transformation.prepare_transformation()
        tf_buffer = tf_get_buffer(pre_transformation.pretransformation_dict)
        tf_checksum = tf_buffer.get_checksum()
        tf_buffer.tempref()
        ### tf_dunder = extract_dunder(pre_transformation.pretransformation_dict)
        tf_dunder = {}  # TODO

        return tf_checksum

    async def constructor_async(transformation_obj):
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
    )
    tf.scratch = scratch
    return tf


__all__ = ["Transformation", "transformation_from_pretransformation"]
