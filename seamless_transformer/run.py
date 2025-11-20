"""Execution helpers for direct transformers."""

from typing import Any, Dict, List, Tuple

import asyncio
import multiprocessing
import warnings

from seamless import Buffer, Checksum
from .cached_compile import exec_code
from .injector import transformer_injector as injector
from .transformation_namespace import build_transformation_namespace_sync
from .execute_forked import run_forked
from .transformation_utils import (
    unpack_deep_structure,
    is_deep_celltype,
    pack_deep_structure,
)

PACK_DEEP_RESULTS = False
_BUFFER_WRITER_HOOK_ACTIVE = False


def run_transformation_dict_in_process(
    transformation_dict: Dict[str, Any], tf_checksum, tf_dunder, scratch: bool
) -> Checksum:
    """Execute a transformation dict in-process.

    Ported from seamless.workflow.core.direct.run.run_transformation_dict_in_process.
    Many responsibilities (metadata, module compilation)
    are still pending and therefore commented out for now.
    """

    """
    STUB
    from seamless.workflow.core.transformation import (
        execution_metadata0,
        get_global_info,
    )
    from seamless.compiler import (
        compilers as default_compilers,
        languages as default_languages,
    )
    from seamless.workflow.core.transformation import build_all_modules
    from seamless.workflow.core.cache.transformation_cache import transformation_cache
    from seamless.checksum.database_client import database
    /STUB
    """

    cached_result = _transformation_cache.get(tf_checksum)
    if cached_result is not None:
        cached_result.tempref()
        return cached_result

    # result_checksum = database.get_transformation_result(tf_checksum)
    # result_checksum = Checksum(result_checksum)
    # if result_checksum:
    #     return result_checksum

    # get_global_info()
    # execution_metadata = deepcopy(execution_metadata0)
    # if "Executor" not in execution_metadata:
    #     execution_metadata["Executor"] = "seamless-in-process"

    transformation: Dict[str, Any] = {}
    transformation.update(transformation_dict)
    transformation.update(tf_dunder)

    if transformation.get("__language__") == "bash":
        raise NotImplementedError("Bash transformers are not supported yet")

    env_checksum0 = transformation.get("__env__")
    if env_checksum0 is not None:
        raise NotImplementedError("Environments are not supported yet")

    inputs, output_name, output_celltype, output_subcelltype = (
        get_transformation_inputs_output(transformation)
    )
    tf_namespace = build_transformation_namespace_sync(transformation)
    code, namespace, modules_to_build, _ = tf_namespace

    module_workspace = {}
    """
    STUB
    compilers = transformation.get("__compilers__", default_compilers)
    languages = transformation.get("__languages__", default_languages)
    build_all_modules(
        modules_to_build,
        module_workspace,
        compilers=compilers,
        languages=languages,
        module_debug_mounts=None,
    )
    /STUB
    """
    assert code is not None

    from . import transformer

    namespace["transformer"] = transformer
    namespace.pop(output_name, None)
    with injector.active_workspace(module_workspace, namespace):
        exec_code(
            code,
            "transformer-in-process",
            namespace,
            inputs,
            output_name,
            with_ipython_kernel=False,
        )
        try:
            result = namespace[output_name]
        except KeyError:
            msg = "Output variable name '%s' undefined" % output_name
            raise RuntimeError(msg) from None

    if result is None:
        raise RuntimeError("Result is empty")

    if is_deep_celltype(output_celltype):
        if not PACK_DEEP_RESULTS:
            raise NotImplementedError(
                "Packing deep transformation results is not supported yet"
            )
        result = pack_deep_structure(result, output_celltype)

    result_buffer = Buffer(result, output_celltype)
    result_checksum = result_buffer.get_checksum()

    if not scratch:
        result_checksum.tempref()

    # database.set_transformation_result(tf_checksum, result_checksum)
    # if not scratch:
    #     buffer_cache.guarantee_buffer_info(
    #         result_checksum, output_celltype, sync_to_remote=True
    #     )
    #     buffer_cache.cache_buffer(result_checksum, result_buffer)
    #     buffer_remote.write_buffer(result_checksum, result_buffer)

    _transformation_cache[tf_checksum] = result_checksum
    return result_checksum


def get_transformation_inputs_output(
    transformation: Dict[str, Any],
) -> Tuple[List[str], str, str, Any]:
    """Return sorted inputs and output descriptors from a transformation dict."""

    inputs: List[str] = []
    as_ = transformation.get("__as__", {})
    for pinname in sorted(transformation.keys()):
        if pinname in (
            "__compilers__",
            "__languages__",
            "__env__",
            "__as__",
            "__meta__",
            "__format__",
        ):
            continue
        if pinname in ("__language__", "__output__", "__code_checksum__"):
            continue
        if pinname == "code":
            continue
        celltype, subcelltype, _ = transformation[pinname]
        if (celltype, subcelltype) == ("plain", "module"):
            continue
        pinname_as = as_.get(pinname, pinname)
        inputs.append(pinname_as)

    outputpin = transformation["__output__"]
    if len(outputpin) == 3:
        outputname, output_celltype, output_subcelltype = outputpin
    else:
        outputname, output_celltype, output_subcelltype, hash_pattern = outputpin
        if hash_pattern == {"*": "#"}:
            output_celltype = "deepcell"
        elif hash_pattern == {"*": "##"}:
            output_celltype = "deepfolder"
    return inputs, outputname, output_celltype, output_subcelltype


async def run_transformation_dict_forked(
    transformation_dict: Dict[str, Any], tf_checksum, tf_dunder, scratch: bool
) -> Checksum:
    """Execute a transformation dict in a forked subprocess."""

    transformation = dict(transformation_dict)
    if tf_dunder:
        transformation.update(tf_dunder)

    cached_result = _transformation_cache.get(tf_checksum)
    if cached_result is not None:
        cached_result.tempref()
        return cached_result

    if transformation.get("__language__") == "bash":
        raise NotImplementedError("Bash transformers are not supported yet")
    if transformation.get("__env__") is not None:
        raise NotImplementedError("Environments are not supported yet")

    inputs, output_name, output_celltype, output_subcelltype = (
        get_transformation_inputs_output(transformation)
    )
    tf_namespace = build_transformation_namespace_sync(transformation)
    code, namespace, modules_to_build, _ = tf_namespace
    if modules_to_build:
        raise NotImplementedError("Module builds not yet supported")
    assert code is not None

    def _run() -> Checksum:
        if multiprocessing.get_start_method(allow_none=True) is None:
            multiprocessing.set_start_method("fork")
        queue: multiprocessing.Queue = multiprocessing.JoinableQueue()
        result_value = None
        result_checksum = None
        with warnings.catch_warnings():
            if _BUFFER_WRITER_HOOK_ACTIVE:
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        "This process .* is multi-threaded, use of fork\\(\\) may "
                        "lead to deadlocks in the child\\."
                    ),
                    category=DeprecationWarning,
                    module=r"multiprocessing",
                )
            proc = multiprocessing.Process(
                target=run_forked,
                args=(
                    "transformer-forked",
                    code,
                    False,
                    injector,
                    {},
                    "transformer-forked",
                    namespace,
                    {},  # deep structures already unpacked
                    inputs,
                    output_name,
                    output_celltype,
                    scratch,
                    queue,
                ),
                kwargs={"tf_checksum": tf_checksum},
                daemon=False,
            )
            proc.start()
            try:
                while True:
                    status, msg = queue.get()
                    queue.task_done()
                    if isinstance(status, tuple) and status[1] == "checksum":
                        if status[0] == 0:
                            result_checksum = Checksum(msg)
                            break
                        continue
                    if status == 0:
                        result_value = msg
                        break
                    if status == 1:
                        raise RuntimeError(msg)
                    if status in (2, 3, 4, 5, 6, 7, 8):
                        continue
                queue.join()
            finally:
                proc.join()
        if proc.exitcode and proc.exitcode != 0 and result_checksum is None:
            raise RuntimeError(f"Forked transformer exited with code {proc.exitcode}")

        if result_checksum is None:
            if result_value is None:
                raise RuntimeError("Forked transformation produced no result")
            packed_result = result_value
            if is_deep_celltype(output_celltype):
                if not PACK_DEEP_RESULTS:
                    raise NotImplementedError(
                        "Packing deep transformation results is not supported yet"
                    )
                packed_result = pack_deep_structure(result_value, output_celltype)
            buf = Buffer(packed_result, output_celltype)
            result_checksum = buf.get_checksum()
        if not scratch:
            result_checksum.tempref()
        _transformation_cache[tf_checksum] = result_checksum
        return result_checksum

    return _run()


def run_transformation_dict_forked_sync(
    transformation_dict: Dict[str, Any], tf_checksum, tf_dunder, scratch: bool
) -> Checksum:
    """Synchronous wrapper for the forked executor."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        future = asyncio.run_coroutine_threadsafe(
            run_transformation_dict_forked(
                transformation_dict, tf_checksum, tf_dunder, scratch
            ),
            loop,
        )
        return future.result()
    new_loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(new_loop)
        return new_loop.run_until_complete(
            run_transformation_dict_forked(
                transformation_dict, tf_checksum, tf_dunder, scratch
            )
        )
    finally:
        asyncio.set_event_loop(None)
        new_loop.close()


def run_transformation_dict(
    transformation_dict: Dict[str, Any],
    tf_checksum,
    tf_dunder,
    scratch: bool,
    *,
    in_process: bool,
) -> Checksum:
    if in_process:
        return run_transformation_dict_in_process(
            transformation_dict, tf_checksum, tf_dunder, scratch
        )
    return run_transformation_dict_forked_sync(
        transformation_dict, tf_checksum, tf_dunder, scratch
    )


_transformation_cache: dict[Checksum, Checksum] = {}

__all__ = [
    "run_transformation_dict_in_process",
    "get_transformation_inputs_output",
    "run_transformation_dict_forked",
    "run_transformation_dict_forked_sync",
    "run_transformation_dict",
]


def mark_buffer_writer_hook_installed() -> None:
    global _BUFFER_WRITER_HOOK_ACTIVE
    _BUFFER_WRITER_HOOK_ACTIVE = True
