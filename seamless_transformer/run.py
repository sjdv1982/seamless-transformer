"""Execution helpers for direct transformers."""

from typing import Any, Dict, List, Tuple
from contextlib import contextmanager
import os
import re
import shutil
import threading
import yaml

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
    TRANSFORMATION_EXECUTION_DUNDER_KEYS,
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

    if _is_compiled_transformation(transformation, tf_dunder):
        result = call_compiled_transform(
            transformation,
            tf_dunder or {},
            code,
            namespace,
            output_celltype,
            meta,
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
            if len(result_buffer):
                result_buffer.tempref()
        elif require_value:
            if len(result_buffer):
                result_buffer.tempref(scratch=True)
        return result_checksum

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
        if pinname in TRANSFORMATION_EXECUTION_DUNDER_KEYS:
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
        if pinname == "objects":
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


def _is_compiled_transformation(transformation, tf_dunder):
    if transformation.get("__compiled__"):
        return True
    return isinstance(tf_dunder, dict) and bool(tf_dunder.get("__compiled__"))


def _get_execution_dunder(transformation, tf_dunder, key):
    value = transformation.get(key)
    if value is None and isinstance(tf_dunder, dict):
        value = tf_dunder.get(key)
    return value


def _resolve_dunder_value(transformation, tf_dunder, key, celltype):
    checksum_hex = _get_execution_dunder(transformation, tf_dunder, key)
    if checksum_hex is None:
        raise RuntimeError(f"Compiled transformation is missing {key}")
    buffer = Checksum(checksum_hex).resolve()
    if buffer is None:
        raise RuntimeError(f"Cannot resolve compiled transformation {key}: {checksum_hex}")
    return buffer.get_value(celltype)


def _numpy_dtype(dtype_spec):
    try:
        import numpy as np
    except ImportError:
        raise ImportError(
            "numpy is required for compiled transformers. Install it with: pip install numpy"
        ) from None

    if hasattr(dtype_spec, "fields"):
        fields = []
        for field in dtype_spec.fields:
            field_dtype = _numpy_dtype(field.dtype)
            if field.shape:
                fields.append((field.name, field_dtype, field.shape))
            else:
                fields.append((field.name, field_dtype))
        return np.dtype(fields, align=True)

    dtype_name = dtype_spec.name
    mapping = {
        "int8": np.dtype("int8"),
        "int16": np.dtype("int16"),
        "int32": np.dtype("int32"),
        "int64": np.dtype("int64"),
        "uint8": np.dtype("uint8"),
        "uint16": np.dtype("uint16"),
        "uint32": np.dtype("uint32"),
        "uint64": np.dtype("uint64"),
        "float32": np.dtype("float32"),
        "float64": np.dtype("float64"),
        "bool": np.dtype("bool"),
        "char": np.dtype("S1"),
        "complex64": np.dtype("complex64"),
        "complex128": np.dtype("complex128"),
    }
    return mapping[dtype_name]


def _is_struct_dtype(dtype_spec) -> bool:
    return hasattr(dtype_spec, "fields")


def _numpy_dtype_matches(actual, expected) -> bool:
    if actual == expected:
        return True
    if actual.fields is None or expected.fields is None:
        return False
    if actual.itemsize != expected.itemsize:
        return False

    expected_names = set(expected.names or ())
    for name in actual.names or ():
        if name in expected_names:
            continue
        field_dtype = actual.fields[name][0]
        if field_dtype.kind != "V" or field_dtype.fields is not None:
            return False

    for name in expected.names or ():
        if name not in actual.fields:
            return False
        actual_field, actual_offset = actual.fields[name][:2]
        expected_field, expected_offset = expected.fields[name][:2]
        if actual_offset != expected_offset:
            return False
        actual_base, actual_shape = actual_field.subdtype or (actual_field, ())
        expected_base, expected_shape = expected_field.subdtype or (expected_field, ())
        if actual_shape != expected_shape:
            return False
        if not _numpy_dtype_matches(actual_base, expected_base):
            return False
    return True


def _is_native_dtype(dtype) -> bool:
    return dtype.isnative


def _require_native_dtype(dtype, name: str):
    if not _is_native_dtype(dtype):
        raise TypeError(f"{name} must have native byte order")


def _camel_case(value: str) -> str:
    return "".join(part.capitalize() for part in re.split(r"_+", value) if part)


def _struct_type_name(parameter) -> str:
    return f"{_camel_case(parameter.name)}Struct"


def _copy_struct_array_as_dtype(array, expected):
    import numpy as np

    source = np.ascontiguousarray(array)
    target = np.empty(source.shape, dtype=expected, order="C")
    target.view(np.uint8).reshape(-1)[:] = source.view(np.uint8).reshape(-1)
    return target


def _shape_for_parameter(parameter, resolved_wildcards, output_maxima=None):
    if parameter.shape is None:
        return ()
    output_maxima = output_maxima or {}
    shape = []
    for dim in parameter.shape:
        if isinstance(dim, int):
            shape.append(dim)
        elif dim in resolved_wildcards:
            shape.append(resolved_wildcards[dim])
        elif dim in output_maxima:
            shape.append(output_maxima[dim])
        else:
            raise ValueError(f"Wildcard {dim!r} is unresolved")
    return tuple(shape)


def _resolve_input_wildcards(sig, namespace):
    resolved = {}
    for parameter in sig.inputs:
        if parameter.shape is None:
            continue
        value = namespace[parameter.name]
        shape = getattr(value, "shape", None)
        if shape is None:
            raise TypeError(f"Input {parameter.name!r} must be a numpy array")
        if len(shape) != len(parameter.shape):
            raise ValueError(
                f"Input {parameter.name!r} has rank {len(shape)}, expected {len(parameter.shape)}"
            )
        for index, dim in enumerate(parameter.shape):
            actual = int(shape[index])
            if isinstance(dim, int):
                if actual != dim:
                    raise ValueError(
                        f"Input {parameter.name!r} dimension {index} is {actual}, expected {dim}"
                    )
            else:
                previous = resolved.get(dim)
                if previous is None:
                    resolved[dim] = actual
                elif previous != actual:
                    raise ValueError(
                        f"Wildcard {dim!r} has inconsistent sizes: {previous} and {actual}"
                    )
    return resolved


def _coerce_scalar_input(ffi, value, parameter, keepalive):
    try:
        import numpy as np
    except ImportError:
        raise ImportError(
            "numpy is required for compiled transformers. Install it with: pip install numpy"
        ) from None

    expected = _numpy_dtype(parameter.dtype)
    if _is_struct_dtype(parameter.dtype):
        if isinstance(value, np.ndarray):
            if value.shape != ():
                raise TypeError(f"Input {parameter.name!r} must be a scalar structured value")
            if not _numpy_dtype_matches(value.dtype, expected):
                raise TypeError(
                    f"Input {parameter.name!r} dtype is {value.dtype}, expected {expected}"
                )
            array = value
        elif isinstance(value, np.void):
            if not _numpy_dtype_matches(value.dtype, expected):
                raise TypeError(
                    f"Input {parameter.name!r} dtype is {value.dtype}, expected {expected}"
                )
            array = np.asarray(value)
        else:
            raise TypeError(
                f"Input {parameter.name!r} must be a numpy structured scalar with dtype {expected}"
            )
        if not _numpy_dtype_matches(array.dtype, expected):
            raise TypeError(
                f"Input {parameter.name!r} dtype is {array.dtype}, expected {expected}"
            )
        _require_native_dtype(array.dtype, parameter.name)
        if array.dtype != expected:
            array = _copy_struct_array_as_dtype(array, expected)
        ptr = ffi.new(f"{_struct_type_name(parameter)} *")
        ffi.memmove(ptr, array.tobytes(), expected.itemsize)
        keepalive.extend([array, ptr])
        return ptr[0]

    if isinstance(value, np.generic):
        if value.dtype != expected:
            raise TypeError(
                f"Input {parameter.name!r} dtype is {value.dtype}, expected {expected}"
            )
        _require_native_dtype(value.dtype, parameter.name)
        return value.item()
    return value


def _coerce_array_input(ffi, value, parameter):
    try:
        import numpy as np
    except ImportError:
        raise ImportError(
            "numpy is required for compiled transformers. Install it with: pip install numpy"
        ) from None

    expected = _numpy_dtype(parameter.dtype)
    array = np.asarray(value)
    if not _numpy_dtype_matches(array.dtype, expected):
        raise TypeError(
            f"Input {parameter.name!r} dtype is {array.dtype}, expected {expected}"
        )
    _require_native_dtype(array.dtype, parameter.name)
    if _is_struct_dtype(parameter.dtype):
        if not array.flags.c_contiguous or not array.flags.aligned or array.dtype != expected:
            array = _copy_struct_array_as_dtype(array, expected)
    else:
        array = np.require(array, dtype=expected, requirements=["C", "ALIGNED"])
    if not array.flags.aligned:
        raise TypeError(f"Input {parameter.name!r} is not aligned")
    ctype = _array_ctype(parameter)
    return array, ffi.from_buffer(ctype, array)


def _array_ctype(parameter):
    from seamless_signature.c_header import SCALAR_C_TYPES

    if _is_struct_dtype(parameter.dtype):
        base = _struct_type_name(parameter)
    else:
        base = parameter.dtype.name
    if parameter.element_shape:
        suffix = "x".join(str(dim) for dim in parameter.element_shape)
        return f"{base}_{suffix}[]"
    if _is_struct_dtype(parameter.dtype):
        return f"{base}[]"
    return f"{SCALAR_C_TYPES[parameter.dtype.name]}[]"


def _output_scalar_pointer(ffi, parameter):
    from seamless_signature.c_header import SCALAR_C_TYPES

    if _is_struct_dtype(parameter.dtype):
        return ffi.new(f"{_struct_type_name(parameter)} *")
    return ffi.new(f"{SCALAR_C_TYPES[parameter.dtype.name]} *")


def _struct_scalar_from_pointer(ffi, ptr, parameter):
    try:
        import numpy as np
    except ImportError:
        raise ImportError(
            "numpy is required for compiled transformers. Install it with: pip install numpy"
        ) from None

    expected = _numpy_dtype(parameter.dtype)
    data = bytes(ffi.buffer(ptr, expected.itemsize))
    return np.frombuffer(data, dtype=expected, count=1)[0]


def _allocate_output_array(ffi, parameter, shape):
    try:
        import numpy as np
    except ImportError:
        raise ImportError(
            "numpy is required for compiled transformers. Install it with: pip install numpy"
        ) from None

    expected = _numpy_dtype(parameter.dtype)
    array = np.empty(shape, dtype=expected, order="C")
    if not array.flags.aligned:
        raise TypeError(f"Output {parameter.name!r} allocation is not aligned")
    return array, ffi.from_buffer(_array_ctype(parameter), array)


def _trim_output_array(parameter, array, output_sizes):
    if not output_sizes or parameter.shape is None:
        return array
    slices = []
    changed = False
    for dim in parameter.shape:
        if isinstance(dim, str) and dim in output_sizes:
            slices.append(slice(0, output_sizes[dim]))
            changed = True
        else:
            slices.append(slice(None))
    if not changed:
        return array
    return array[tuple(slices)]


def _module_definition_from_payload(language, code, header, objects, compilation):
    main_compilation = {}
    object_compilations = {}
    if isinstance(compilation, dict):
        main_compilation = compilation.get("main") or {}
        object_compilations = compilation.get("objects") or {}
    module_definition = {
        "type": "compiled",
        "target": main_compilation.get("target", "profile"),
        "link_options": main_compilation.get("link_options", []),
        "public_header": {"language": "c", "code": header},
        "objects": {
            "main": {
                "language": language,
                "code": code,
                "options": main_compilation.get("options"),
                "debug_options": main_compilation.get("debug_options"),
            }
        },
    }
    for name, obj in (objects or {}).items():
        override = object_compilations.get(name, {})
        module_definition["objects"][name] = {
            "language": obj["language"],
            "code": obj["code"],
            "options": override.get("options"),
            "debug_options": override.get("debug_options"),
        }
    return module_definition


def call_compiled_transform(
    transformation: Dict[str, Any],
    tf_dunder: Dict[str, Any],
    code: str,
    namespace: Dict[str, Any],
    output_celltype: str,
    meta: Dict[str, Any],
):
    """Build and call a compiled transformer through CFFI."""

    from seamless_signature import Signature
    from seamless_transformer.compiler import build_compiled_module

    schema_text = _resolve_dunder_value(transformation, tf_dunder, "__schema__", "text")
    header = _resolve_dunder_value(transformation, tf_dunder, "__header__", "text")
    compilation = _resolve_dunder_value(
        transformation, tf_dunder, "__compilation__", "plain"
    )
    sig = Signature.from_dict(yaml.safe_load(schema_text))
    objects = namespace.get("objects", {})
    language = transformation.get("__language__")
    module_definition = _module_definition_from_payload(
        language, code, header, objects, compilation
    )
    module = build_compiled_module(module_definition)
    ffi = module.ffi
    lib = module.lib

    resolved_wildcards = _resolve_input_wildcards(sig, namespace)
    metavars = {}
    if isinstance(meta, dict):
        metavars = dict(meta.get("metavars") or {})
    output_maxima = {}
    for wildcard in sig.output_wildcards:
        key = f"max{wildcard}"
        if key not in metavars:
            raise ValueError(f"Missing compiled transformer metavar {key!r}")
        output_maxima[wildcard] = int(metavars[key])

    keepalive = []
    call_args = []
    call_args.extend(int(resolved_wildcards[name]) for name in sig.input_wildcards)
    call_args.extend(int(output_maxima[name]) for name in sig.output_wildcards)

    for parameter in sig.inputs:
        value = namespace[parameter.name]
        if parameter.shape is None:
            call_args.append(_coerce_scalar_input(ffi, value, parameter, keepalive))
        else:
            array, cdata = _coerce_array_input(ffi, value, parameter)
            keepalive.append(array)
            call_args.append(cdata)

    output_size_ptrs = {}
    for wildcard in sig.output_wildcards:
        ptr = ffi.new("unsigned int *")
        output_size_ptrs[wildcard] = ptr
        call_args.append(ptr)

    output_values = {}
    for parameter in sig.outputs:
        if parameter.shape is None:
            ptr = _output_scalar_pointer(ffi, parameter)
            output_values[parameter.name] = ("scalar", ptr, parameter)
            call_args.append(ptr)
        else:
            shape = _shape_for_parameter(parameter, resolved_wildcards, output_maxima)
            array, cdata = _allocate_output_array(ffi, parameter, shape)
            keepalive.append(array)
            output_values[parameter.name] = ("array", array, parameter)
            call_args.append(cdata)

    status = lib.transform(*call_args)
    if status != 0:
        raise RuntimeError(f"Compiled transform returned non-zero status {status}")

    output_sizes = {
        wildcard: int(ptr[0]) for wildcard, ptr in output_size_ptrs.items()
    }
    result = {}
    for name, (kind, value, parameter) in output_values.items():
        if kind == "scalar":
            if _is_struct_dtype(parameter.dtype):
                item = _struct_scalar_from_pointer(ffi, value, parameter)
            else:
                item = value[0]
                try:
                    item = item.item()
                except AttributeError:
                    pass
            result[name] = item
        else:
            result[name] = _trim_output_array(parameter, value, output_sizes)
    if len(sig.outputs) == 1:
        return result[sig.outputs[0].name]
    if output_celltype not in ("mixed", "deepcell"):
        raise TypeError("multi-output compiled transformers require 'mixed' or 'deepcell'")
    return result


__all__ = [
    "run_transformation_dict",
    "get_transformation_inputs_output",
    "is_driver_context",
    "RemoteJobWritten",
    "call_compiled_transform",
]
