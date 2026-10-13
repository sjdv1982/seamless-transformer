"""The built-in Julia language handler."""

from .. import register_language_handler


def julia(
    code,
    pins,
    filesystem,
    output_name,
    output_celltype,
    metavars,
    environment,
    process_lock,
):
    import hashlib
    import json
    import os
    import tempfile
    from collections.abc import Mapping
    from pathlib import Path

    import numpy as np
    from seamless import Buffer
    from seamless_transformer.execute_bash import execute_bash
    from seamless_transformer.languages import run_in_interpreter_thread

    if not isinstance(metavars, Mapping):
        metavars = {}
    backend = metavars.get("BACKEND", "cli")
    if backend not in ("cli", "mem"):
        raise ValueError(f"Unknown Julia backend: {backend!r}")
    if backend == "mem" and not process_lock:
        raise RuntimeError(
            "The mem backend requires the process lock; "
            "use BACKEND='cli' instead"
        )
    if not isinstance(pins, Mapping):
        raise TypeError("Julia handler pins must be a mapping")
    if not isinstance(filesystem, Mapping):
        filesystem = {}

    deep_celltypes = {"deepcell", "deepfolder", "folder"}
    file_celltypes = {"mixed", "plain", "text", "bytes", "binary"}
    if output_celltype in deep_celltypes:
        raise ValueError(
            f"Julia handler cannot return deep or folder celltype {output_celltype!r}"
        )
    environment = environment if isinstance(environment, Mapping) else {}
    if backend == "mem":
        from seamless_transformer.languages import check_in_memory_environment

        check_in_memory_environment(environment, "julia")

    from seamless_transformer.compression_utils import (
        decompress_bytes,
        strip_compression_suffix,
    )

    def read_pin_bytes(value, celltype, entry):
        if entry.get("filesystem"):
            path = os.fspath(value)
            with open(path, "rb") as f:
                data = f.read()
            _, suffix = strip_compression_suffix(path)
            if suffix:
                data = decompress_bytes(data, suffix)
            return data
        if isinstance(value, Buffer):
            return value.content
        buffer_celltype = celltype if celltype in file_celltypes else "plain"
        return Buffer(value, buffer_celltype).content

    def encode_tree(value):
        if isinstance(value, np.ndarray):
            array = np.asarray(value)
            if array.dtype.kind in "iu":
                if array.dtype.kind == "u":
                    dtype = "uint64"
                    array = array.astype(np.uint64, copy=False)
                else:
                    dtype = "int64"
                    array = array.astype(np.int64, copy=False)
            elif array.dtype.kind == "f":
                dtype = "float64"
                array = array.astype(np.float64, copy=False)
            elif array.dtype.kind == "b":
                dtype = "bool"
                array = array.astype(np.bool_, copy=False)
            else:
                raise TypeError(
                    f"Julia handler cannot transport array dtype {array.dtype}"
                )
            return {
                "__seamless_array__": True,
                "dtype": dtype,
                "shape": list(array.shape),
                "values": array.ravel(order="F").tolist(),
            }
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Mapping):
            return {str(k): encode_tree(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode_tree(v) for v in value]
        if isinstance(value, (bytes, bytearray, memoryview)):
            return {
                "__seamless_bytes__": True,
                "values": list(bytes(value)),
            }
        return value

    pin_specs = []
    pin_data = {}
    pin_is_filesystem = {}
    native_arrays = {}
    for index, (name, value) in enumerate(pins.items()):
        if not isinstance(name, str):
            raise TypeError("Julia handler pin names must be strings")
        entry = filesystem.get(name, {})
        if not isinstance(entry, Mapping):
            raise TypeError(
                f"Julia handler filesystem entry for {name!r} must be a mapping"
            )
        celltype = entry.get("celltype", "mixed")
        if celltype in deep_celltypes:
            raise ValueError(
                f"Julia handler cannot accept deep or folder pin celltype {celltype!r}"
            )
        on_file = bool(entry.get("filesystem", False))
        native_array = None
        container_celltype = celltype in {"binary", "mixed"}
        if on_file:
            if celltype not in file_celltypes:
                raise TypeError(
                    f"Julia handler cannot transport file pin {name!r} "
                    f"with celltype {celltype!r}"
                )
            if not isinstance(value, (str, os.PathLike)):
                raise TypeError(f"Julia handler file pin {name!r} must be a path")
            path = os.fspath(value)
            if backend == "mem" and container_celltype:
                _, suffix = strip_compression_suffix(path)
                if not suffix:
                    with open(path, "rb") as f:
                        array_file = f.read(6) == b"\x93NUMPY"
                    if array_file:
                        native_array = np.load(
                            path, mmap_mode="c", allow_pickle=False
                        )
            if native_array is None:
                raw = read_pin_bytes(value, celltype, entry)
            else:
                raw = b""
        else:
            raw = read_pin_bytes(value, celltype, entry)

        if (
            native_array is None
            and celltype == "mixed"
            and raw.startswith(b"\x94SEAMLESS-MIXED")
        ):
            if backend == "cli":
                raise RuntimeError(
                    "The Julia CLI backend cannot read a Seamless-mixed pin container"
                )
            from seamless.util.mixed.io import deserialize

            raw = (
                b"SEAMLESS-JULIA-JSON\n"
                + json.dumps(
                    encode_tree(deserialize(raw)[0]),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
        elif (
            native_array is None
            and container_celltype
            and raw.startswith(b"\x93NUMPY")
            and backend == "mem"
        ):
            native_array = np.load(__import__("io").BytesIO(raw), allow_pickle=False)

        if native_array is not None:
            array = native_array
            if array.dtype.kind in "iu":
                dtype = np.uint64 if array.dtype.kind == "u" else np.int64
            elif array.dtype.kind == "f":
                dtype = np.float64
            elif array.dtype.kind == "b":
                dtype = np.bool_
            else:
                raise TypeError(
                    f"Julia handler cannot transport array dtype {array.dtype}"
                )
            if array.dtype != dtype:
                array = array.astype(
                    dtype,
                    copy=True,
                    order="F" if array.ndim > 1 else "K",
                )
            elif array.ndim > 1 and not array.flags.f_contiguous:
                array = np.asfortranarray(array)
            native_arrays[name] = array
            raw = b""

        pin_data[name] = raw
        pin_is_filesystem[name] = bool(entry.get("filesystem", False))
        pin_file = f"seamless_pin_{index:04d}"
        pin_specs.append({"name": name, "file": pin_file, "celltype": celltype})

    key_material = json.dumps(
        [code, output_name, sorted(pins)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    config = {
        "cache_key": hashlib.sha256(key_material).hexdigest(),
        "output_name": output_name,
        "output_celltype": output_celltype,
        "pins": pin_specs,
    }
    runtime_source = r'''module SeamlessRuntime
using JSON

const ISOLATIONS = Dict{String,Module}()
const NPY_MAGIC = UInt8[0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]
const MIXED_MAGIC = UInt8[0x94; codeunits("SEAMLESS-MIXED")]
const EMBEDDED_MAGIC = Vector{UInt8}(codeunits("SEAMLESS-JULIA-JSON\n"))

function has_prefix(data::Vector{UInt8}, prefix::Vector{UInt8})
    length(data) < length(prefix) && return false
    return data[1:length(prefix)] == prefix
end

function canonical_array(value::AbstractArray)
    if eltype(value) === Bool
        return value
    elseif eltype(value) === Int64
        return value
    elseif eltype(value) <: Signed
        return Int64.(value)
    elseif eltype(value) === UInt64
        return value
    elseif eltype(value) <: Unsigned
        return UInt64.(value)
    elseif eltype(value) === Float64
        return value
    elseif eltype(value) <: AbstractFloat
        return Float64.(value)
    end
    error("Julia handler supports numeric arrays with boolean, integer or float elements")
end

function array_from_pyarray(value::AbstractArray)
    element_type = eltype(value)
    dimensions = ndims(value)
    if !(element_type === Bool || element_type === Int64 ||
         element_type === UInt64 || element_type === Float64)
        error("Julia handler received a non-canonical NumPy array")
    end
    array = unsafe_wrap(
        Array{element_type,dimensions}, pointer(value), size(value); own=false
    )
    finalizer(array) do _
        GC.@preserve value nothing
    end
    return array
end

function embedded_value(value)
    if value isa AbstractDict
        if get(value, "__seamless_array__", false) === true
            shape = Tuple(Int.(value["shape"]))
            dtype = value["dtype"]
            values = value["values"]
            if dtype == "bool"
                return reshape(Bool.(values), shape)
            elseif dtype == "int64"
                return reshape(Int64.(values), shape)
            elseif dtype == "uint64"
                return reshape(UInt64.(values), shape)
            elseif dtype == "float64"
                return reshape(Float64.(values), shape)
            end
            error("Unknown embedded array dtype: $dtype")
        elseif get(value, "__seamless_bytes__", false) === true
            return UInt8.(value["values"])
        end
        return Dict{String,Any}(key => embedded_value(item) for (key, item) in value)
    elseif value isa Vector
        return Any[embedded_value(item) for item in value]
    end
    return value
end

function read_npy(data::Vector{UInt8})
    path = tempname()
    try
        open(path, "w") do io
            write(io, data)
        end
        if !isdefined(@__MODULE__, :NPZ)
            Core.eval(@__MODULE__, :(import NPZ))
        end
        npzread = getfield(@__MODULE__, :NPZ).npzread
        return canonical_array(Base.invokelatest(npzread, path))
    finally
        isfile(path) && rm(path; force=true)
    end
end

function decode_pin(data::Vector{UInt8}, celltype::String)
    if celltype == "bytes"
        return data
    elseif celltype == "text"
        value = String(data)
        return endswith(value, "\n") ? chop(value; tail=1) : value
    elseif celltype == "mixed" && has_prefix(data, EMBEDDED_MAGIC)
        text = String(data[length(EMBEDDED_MAGIC)+1:end])
        return embedded_value(JSON.parse(text))
    elseif celltype == "binary"
        return read_npy(data)
    elseif celltype == "mixed"
        if has_prefix(data, MIXED_MAGIC)
            error("Julia CLI backend cannot read a Seamless mixed pin container")
        elseif has_prefix(data, NPY_MAGIC)
            return read_npy(data)
        end
        return JSON.parse(String(data))
    end
    return JSON.parse(String(data))
end

function unwrap(ex)
    while ex isa Expr
        if ex.head === :macrocall && length(ex.args) >= 4 &&
           ex.args[1] isa GlobalRef && ex.args[1].name === Symbol("@doc")
            ex = ex.args[end]
        else
            break
        end
    end
    return ex
end

function callname(sig)
    while sig isa Expr && sig.head in (:where, :(::))
        sig = sig.args[1]
    end
    return (sig isa Expr && sig.head === :call) ? sig.args[1] : nothing
end

function defines_transform(ex)
    ex = unwrap(ex)
    ex isa Expr || return false
    ex.head in (:function, :(=)) || return false
    return !isempty(ex.args) && callname(ex.args[1]) === :transform
end

is_transform(code::String) = any(defines_transform, Meta.parseall(code).args)

function call_by_name(f, pins::Dict{String,Any})
    candidates = []
    for method in methods(f)
        positional = Base.method_argnames(method)[2:end]
        all(name -> haskey(pins, String(name)), positional) || continue
        args = Any[pins[String(name)] for name in positional]
        applicable(f, args...) || continue
        keywords = Base.kwarg_decl(method)
        kwargs = Pair{Symbol,Any}[
            name => pins[String(name)] for name in keywords
            if haskey(pins, String(name))
        ]
        score = (length(positional) + length(kwargs), length(positional))
        push!(
            candidates,
            (score=score, args=args, kwargs=kwargs, positional=positional),
        )
    end
    isempty(candidates) &&
        error("No Julia transform method matches the available named pins")
    best_score = maximum(candidate.score for candidate in candidates)
    best = filter(candidate -> candidate.score == best_score, candidates)
    mappings = unique([
        (
            Tuple(String(name) for name in candidate.positional),
            Tuple(first(pair) for pair in candidate.kwargs),
        )
        for candidate in best
    ])
    length(mappings) == 1 ||
        error("Julia transform has ambiguous parameter names for the available pins")
    candidate = first(best)
    return f(candidate.args...; candidate.kwargs...)
end

function array_payload(result)
    array = canonical_array(result)
    dtype = eltype(array) === Bool ? "bool" :
            eltype(array) === Int64 ? "int64" :
            eltype(array) === UInt64 ? "uint64" : "float64"
    return Dict(
        "__seamless_array__" => true,
        "dtype" => dtype,
        "shape" => collect(size(array)),
        "values" => collect(vec(array)),
    )
end

function encode_result(result, output_celltype::String)
    if output_celltype == "bytes"
        value = if result isa AbstractString
            Vector{UInt8}(codeunits(result))
        elseif result isa AbstractVector{UInt8}
            collect(UInt8, result)
        else
            error("Julia bytes output must be a string or Vector{UInt8}")
        end
        return JSON.json(Dict("__seamless_bytes__" => true, "values" => collect(value)))
    elseif output_celltype in ("binary", "mixed") &&
           result isa AbstractArray && eltype(result) <: Number
        return JSON.json(array_payload(result))
    end
    return JSON.json(result)
end

function run_values(
    cache_key::String,
    code::String,
    pins::Dict{String,Any},
    output_name::String,
    output_celltype::String,
)
    if is_transform(code)
        mod = get!(ISOLATIONS, cache_key) do
            m = Module(:SeamlessTF)
            Base.include_string(m, code, "transformer.jl")
            m
        end
        f = Base.invokelatest(getfield, mod, :transform)
        result = Base.invokelatest(call_by_name, f, pins)
    else
        mod = Module(:SeamlessTF)
        for (name, value) in pins
            Core.eval(mod, :($(Symbol(name)) = $(QuoteNode(value))))
        end
        Base.include_string(mod, code, "transformer.jl")
        Base.invokelatest(isdefined, mod, Symbol(output_name)) ||
            error("Julia transformer did not define '$output_name'")
        result = Base.invokelatest(getfield, mod, Symbol(output_name))
    end
    return encode_result(result, output_celltype)
end

function run(
    cache_key::String,
    code::String,
    raw_pins::Dict{String,Vector{UInt8}},
    pin_celltypes::Dict{String,String},
    output_name::String,
    output_celltype::String,
)
    pins = Dict{String,Any}(
        name => decode_pin(data, get(pin_celltypes, name, "mixed"))
        for (name, data) in raw_pins
    )
    return run_values(cache_key, code, pins, output_name, output_celltype)
end

function run_native(
    cache_key::String,
    code::String,
    raw_pins::Dict{String,Vector{UInt8}},
    pin_celltypes::Dict{String,String},
    native_arrays::Dict{String,Any},
    output_name::String,
    output_celltype::String,
)
    pins = Dict{String,Any}(
        name => decode_pin(data, get(pin_celltypes, name, "mixed"))
        for (name, data) in raw_pins
    )
    for (name, array) in native_arrays
        pins[name] = canonical_array(array)
    end
    return run_values(cache_key, code, pins, output_name, output_celltype)
end

function read_job(config_path::String, code_path::String)
    config = JSON.parsefile(config_path)
    raw_pins = Dict{String,Vector{UInt8}}()
    pin_celltypes = Dict{String,String}()
    for pin in config["pins"]
        raw_pins[pin["name"]] = read(pin["file"])
        pin_celltypes[pin["name"]] = pin["celltype"]
    end
    return config, read(code_path, String), raw_pins, pin_celltypes
end

function run_files(config_path::String, code_path::String)
    config, code, raw_pins, pin_celltypes = read_job(config_path, code_path)
    return run(
        config["cache_key"],
        code,
        raw_pins,
        pin_celltypes,
        config["output_name"],
        config["output_celltype"],
    )
end

function run_captured(
    config_path::String,
    code_path::String,
    stdout_path::String,
    stderr_path::String,
)
    result = ""
    open(stdout_path, "w") do out
        open(stderr_path, "w") do err
            redirect_stdout(out) do
                redirect_stderr(err) do
                    result = run_files(config_path, code_path)
                    flush(out)
                    flush(err)
                end
            end
        end
    end
    return result
end

function run_captured_native(
    cache_key::String,
    code::String,
    raw_pins::Dict{String,Vector{UInt8}},
    pin_celltypes::Dict{String,String},
    native_arrays::Dict{String,Any},
    output_name::String,
    output_celltype::String,
    stdout_path::String,
    stderr_path::String,
)
    result = ""
    open(stdout_path, "w") do out
        open(stderr_path, "w") do err
            redirect_stdout(out) do
                redirect_stderr(err) do
                    result = run_native(
                        cache_key,
                        code,
                        raw_pins,
                        pin_celltypes,
                        native_arrays,
                        output_name,
                        output_celltype,
                    )
                    flush(out)
                    flush(err)
                end
            end
        end
    end
    return result
end

end # module SeamlessRuntime
'''
    runner_source = r'''include("runtime.jl")
result = SeamlessRuntime.run_files("pins.json", "transformer.jl")
open("RESULT", "w") do io
    write(io, codeunits(result))
end
'''

    if backend == "cli":
        job_pins = {}
        job_filesystem = {}
        for spec in pin_specs:
            name = spec["name"]
            entry = filesystem.get(name, {})
            on_file = pin_is_filesystem[name]
            job_name = spec["file"]
            if on_file:
                job_pins[job_name] = os.fspath(pins[name])
            else:
                job_pins[job_name] = Buffer(pin_data[name], "bytes")
            job_filesystem[job_name] = {
                "mode": "file",
                "optional": True,
                "filesystem": on_file,
                "celltype": "bytes",
            }
            spec["file"] = job_name

        config["pins"] = pin_specs
        job_pins.update(
            {
                "META__FILE__runtime.jl": runtime_source,
                "META__FILE__runner.jl": runner_source,
                "META__FILE__transformer.jl": code,
                "META__FILE__pins.json": json.dumps(config, ensure_ascii=False),
            }
        )
        result = execute_bash(
            "julia --startup-file=no runner.jl",
            sorted(job_pins),
            environment.get("conda_environment", ""),
            job_pins,
            job_filesystem,
            "plain",
        )
    else:
        with tempfile.TemporaryDirectory(prefix="seamless-juliacall-") as tempdir:
            config_path = os.path.join(tempdir, "pins.json")
            code_path = os.path.join(tempdir, "transformer.jl")
            stdout_path = os.path.join(tempdir, "stdout")
            stderr_path = os.path.join(tempdir, "stderr")
            for spec in pin_specs:
                spec["file"] = os.path.join(tempdir, spec["file"])
                Path(spec["file"]).write_bytes(pin_data[spec["name"]])
            Path(config_path).write_text(
                json.dumps(config, ensure_ascii=False), encoding="utf-8"
            )
            Path(code_path).write_text(code, encoding="utf-8")
            Path(stdout_path).touch()
            Path(stderr_path).touch()

            def run_on_owner():
                os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
                os.environ.setdefault("PYTHON_JULIACALL_THREADS", "1")
                try:
                    import ctypes
                    import juliacall

                    jl = juliacall.Main
                    julia_lib = juliacall.CONFIG["lib"]
                    if not getattr(juliacall, "_seamless_owner_atexit", False):
                        import threading

                        original_atexit_hook = julia_lib.jl_atexit_hook
                        atexit_state = {"done": False, "error": None}

                        def run_native_atexit_hook():
                            try:
                                original_atexit_hook(0)
                                return True, None
                            except BaseException as exc:
                                return False, f"{type(exc).__name__}: {exc}"

                        def shutdown_julia_on_owner():
                            if atexit_state["done"]:
                                return
                            succeeded, message = run_in_interpreter_thread(
                                "juliacall", run_native_atexit_hook
                            )
                            atexit_state["done"] = True
                            if not succeeded:
                                atexit_state["error"] = message
                                raise RuntimeError(
                                    f"juliacall native exit hook failed: {message}"
                                )

                        def owner_atexit_hook(status):
                            # juliacall's public atexit hook runs after
                            # ThreadPoolExecutor shutdown. Julia was already
                            # stopped on its owner in threading's earlier hook.
                            return None

                        owner_atexit_hook.argtypes = [ctypes.c_int]
                        owner_atexit_hook.restype = None
                        julia_lib.jl_atexit_hook = owner_atexit_hook
                        threading._register_atexit(shutdown_julia_on_owner)
                        juliacall._seamless_owner_atexit_state = atexit_state
                        juliacall._seamless_owner_atexit = True

                    if not jl.seval("isdefined(Main, :SeamlessRuntime)"):
                        jl.seval(runtime_source)
                    raw_julia = jl.Dict[jl.String, jl.Vector[jl.UInt8]]()
                    celltypes_julia = jl.Dict[jl.String, jl.String]()
                    arrays_julia = jl.Dict[jl.String, jl.Any]()
                    array_from_pyarray = jl.seval(
                        "SeamlessRuntime.array_from_pyarray"
                    )
                    for spec in pin_specs:
                        name = spec["name"]
                        julia_name = jl.String(name)
                        celltypes_julia[julia_name] = jl.String(spec["celltype"])
                        if name in native_arrays:
                            array = native_arrays[name]
                            py_array = jl.PyArray(array, copy=False)
                            arrays_julia[julia_name] = array_from_pyarray(py_array)
                        else:
                            data = np.frombuffer(pin_data[name], dtype=np.uint8)
                            raw_julia[julia_name] = jl.convert(
                                jl.Vector[jl.UInt8], data
                            )
                    run_captured = jl.seval("SeamlessRuntime.run_captured_native")
                    result_json = run_captured._jl_call_nogil(
                        config["cache_key"],
                        code,
                        raw_julia,
                        celltypes_julia,
                        arrays_julia,
                        output_name,
                        output_celltype,
                        stdout_path,
                        stderr_path,
                    )
                    return True, str(result_json), None
                except BaseException as exc:
                    return False, None, f"{type(exc).__name__}: {exc}"

            succeeded, result_json, error_message = run_in_interpreter_thread(
                "juliacall", run_on_owner
            )
            import sys

            stdout_text = Path(stdout_path).read_text(encoding="utf-8")
            stderr_text = Path(stderr_path).read_text(encoding="utf-8")
            if stdout_text:
                sys.stdout.write(stdout_text)
                sys.stdout.flush()
            if stderr_text:
                sys.stderr.write(stderr_text)
                sys.stderr.flush()
            if not succeeded:
                raise RuntimeError(f"juliacall execution failed: {error_message}")
            result = json.loads(result_json)

    if isinstance(result, Mapping) and result.get("__seamless_bytes__") is True:
        return bytes(result["values"])
    if isinstance(result, Mapping) and result.get("__seamless_array__") is True:
        dtype = result["dtype"]
        numpy_dtype = {
            "bool": np.bool_,
            "int64": np.int64,
            "uint64": np.uint64,
            "float64": np.float64,
        }[dtype]
        array = np.asarray(result["values"], dtype=numpy_dtype).reshape(
            tuple(result["shape"]), order="F"
        )
        return np.ascontiguousarray(array)
    return result


register_language_handler("julia", julia, metavars={"BACKEND": "cli"})


__all__ = ["julia"]
