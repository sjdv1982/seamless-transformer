"""Execution helpers for direct transformers."""

from typing import Any, Dict, List, Tuple
from contextlib import contextmanager
import os
import shutil
import threading

from seamless import Buffer, Checksum
from .cached_compile import exec_code
from .remote_job import (
    RemoteJobWritten,
    encode_remote_job_written,
    get_write_remote_job,
)
from .injector import transformer_injector as injector
from .module_builder import build_all_modules
from .transformation_namespace import build_transformation_namespace_sync
from .transformation_utils import (
    is_deep_celltype,
    merge_transformation_meta,
    pack_deep_structure,
    resolve_env_checksum,
)
from .execute_bash import execute_bash, write_bash_job

PACK_DEEP_RESULTS = True

_DRIVER_CONTEXT = threading.local()


def is_driver_context() -> bool:
    return bool(getattr(_DRIVER_CONTEXT, "active", False))


@contextmanager
def _driver_context(active: bool):
    prev = getattr(_DRIVER_CONTEXT, "active", False)
    _DRIVER_CONTEXT.active = bool(active)
    try:
        yield
    finally:
        _DRIVER_CONTEXT.active = prev


def _execute(
    module_workspace, code, identifier, namespace, inputs, output_name, driver_active
):
    with _driver_context(driver_active):
        with injector.active_workspace(module_workspace, namespace):
            _inject_main_globals(module_workspace, namespace)
            exec_code(
                code,
                identifier,
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
    return result


def _inject_main_globals(module_workspace, namespace):
    main_mod = module_workspace.get("main")
    if main_mod is None:
        return
    for name, value in main_mod.__dict__.items():
        if name.startswith("__"):
            continue
        namespace[name] = value


def run_transformation_dict(
    transformation_dict: Dict[str, Any],
    tf_checksum: Checksum,
    tf_dunder: Dict[str, Any] | None = None,
    scratch: bool = False,
    require_value: bool = False,
) -> Checksum | str:
    """Execute a transformation dict.

    tf_checksum is the checksum of the transformation dict
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
    /STUB
    """

    # get_global_info()
    # execution_metadata = deepcopy(execution_metadata0)
    # if "Executor" not in execution_metadata:
    #     execution_metadata["Executor"] = "seamless-in-process"

    if os.environ.get("SEAMLESS_DEBUG_FORK"):
        print(
            "[transformer-run] running transformation in-process",
            flush=True,
        )

    transformation: Dict[str, Any] = {}
    transformation.update(transformation_dict)
    meta = merge_transformation_meta(transformation, tf_dunder)
    if meta:
        transformation["__meta__"] = meta
    driver_active = bool(meta.get("driver")) if isinstance(meta, dict) else False

    env_checksum0 = resolve_env_checksum(transformation, tf_dunder)
    env_dict = {}
    if env_checksum0 is not None:
        env_buffer = Checksum(env_checksum0).resolve()
        if env_buffer is not None:
            env_dict = env_buffer.get_value("plain")
        if not isinstance(env_dict, dict):
            env_dict = {}

    inputs, output_name, output_celltype, output_subcelltype = (
        get_transformation_inputs_output(transformation)
    )

    tf_namespace = build_transformation_namespace_sync(transformation)
    code, namespace, modules_to_build, _ = tf_namespace
    language = transformation.get("__language__", "python")
    if language == "python" and isinstance(code, str) and len(code) == 64:
        # Defensive fallback: recover code text if a checksum hex string leaked through.
        tf_dict_fallback = None
        try:
            tf_dict_fallback = tf_checksum.resolve(celltype="plain")
        except Exception:
            tf_dict_fallback = None
        if isinstance(tf_dict_fallback, dict):
            code_text = tf_dict_fallback.get("__code_text__")
            if isinstance(code_text, str):
                code = code_text
        if isinstance(code, str) and len(code) == 64:
            try:
                buf = Checksum(code).resolve()
                if buf is not None:
                    code = buf.get_value("python")
            except Exception:
                pass

    module_workspace = {}
    if modules_to_build:
        build_all_modules(
            modules_to_build,
            module_workspace,
            module_debug_mounts=None,
        )
    assert code is not None

    namespace.pop(output_name, None)

    identifier = "transformer-in-process"
    code_checksum = transformation.get("__code_checksum__")
    if code_checksum is None:
        code_checksum = tf_checksum
    try:
        checksum_hex = code_checksum.hex()
    except AttributeError:
        checksum_hex = str(code_checksum) if code_checksum is not None else None
    if checksum_hex:
        identifier = f"{identifier}-{checksum_hex}"

    if is_deep_celltype(output_celltype):
        namespace["OUTPUTPIN"] = output_celltype

    if language == "bash":
        remote_job_dir = get_write_remote_job(meta)
        if remote_job_dir is not None:
            job_directory = _write_remote_bash_job(
                remote_job_dir,
                code,
                sorted(namespace.get("PINS", {}).keys()),
                env_dict.get("conda_environment", ""),
                namespace.get("PINS", {}),
                namespace.get("FILESYSTEM", {}),
            )
            return encode_remote_job_written(job_directory)
        namespace["execute_bash"] = execute_bash
        namespace["bashcode"] = code
        pins = namespace.get("PINS", {})
        namespace["pins_"] = sorted(pins.keys())
        namespace["conda_environment_"] = env_dict.get("conda_environment", "")
        code = (
            f"{output_name} = execute_bash("
            "bashcode, pins_, conda_environment_, PINS, FILESYSTEM, OUTPUTPIN)"
        )
        inputs = []

    result = _execute(
        module_workspace,
        code,
        identifier,
        namespace,
        inputs,
        output_name,
        driver_active,
    )

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
        # Keep the result buffer around so resolve() can find it.
        if len(result_buffer):
            result_buffer.tempref()
    elif require_value:
        if len(result_buffer):
            result_buffer.tempref(scratch=True)

    return result_checksum


def _write_remote_bash_job(
    remote_job_dir: str,
    bashcode: str,
    pins_: List[str],
    conda_environment: str,
    pins: Dict[str, Any],
    filesystem: Dict[str, Any],
) -> str:
    """Materialize a bash transformation job and stop before execution."""

    job_directory = os.path.abspath(os.path.expanduser(remote_job_dir))
    old_cwd = os.getcwd()
    try:
        os.makedirs(job_directory)
    except PermissionError as exc:
        raise PermissionError(
            "Cannot create remote job directory "
            f"'{job_directory}': {exc}. "
            "The path is interpreted on the execution host that materializes the "
            "job directory, typically the remote jobserver host; choose a path "
            "that is writable there, for example ~/..., /tmp/..., or your "
            "cluster scratch directory."
        ) from exc
    try:
        os.chdir(job_directory)
        write_bash_job(
            bashcode,
            pins_,
            conda_environment,
            pins,
            filesystem,
        )
    except Exception:
        os.chdir(old_cwd)
        shutil.rmtree(job_directory, ignore_errors=True)
        raise
    finally:
        try:
            os.chdir(old_cwd)
        except Exception:
            pass
    return job_directory


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
        if pinname in (
            "__language__",
            "__output__",
            "__code_checksum__",
            "__code_text__",
        ):
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


__all__ = [
    "run_transformation_dict",
    "get_transformation_inputs_output",
    "is_driver_context",
    "RemoteJobWritten",
]
