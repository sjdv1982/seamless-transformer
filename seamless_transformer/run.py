"""Execution helpers for direct transformers."""

from typing import Any, Dict, List, Tuple

from seamless import Buffer, Checksum
from .cached_compile import exec_code
from .injector import transformer_injector as injector
from .transformation_namespace import build_transformation_namespace_sync
from .transformer import transformer


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

    inputs, output_name, output_celltype, output_subcelltype, output_hash_pattern = (
        get_transformation_inputs_output(transformation)
    )
    tf_namespace = build_transformation_namespace_sync(transformation)
    code, namespace, modules_to_build, deep_structures_to_unpack = tf_namespace

    if deep_structures_to_unpack:
        raise NotImplementedError("Deep structures are not supported yet")

    if output_hash_pattern is not None:
        raise NotImplementedError("Hash patterns are not supported yet")

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
) -> Tuple[List[str], str, str, Any, Any]:
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
        output_hash_pattern = None
    else:
        outputname, output_celltype, output_subcelltype, output_hash_pattern = outputpin

    return inputs, outputname, output_celltype, output_subcelltype, output_hash_pattern


_transformation_cache: dict[Checksum, Checksum] = {}

__all__ = ["run_transformation_dict_in_process", "get_transformation_inputs_output"]
