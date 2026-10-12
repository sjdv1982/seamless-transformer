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
    from collections.abc import Mapping

    import numpy as np
    from seamless import Buffer
    from seamless_transformer.execute_bash import execute_bash

    if not isinstance(metavars, Mapping):
        metavars = {}
    backend = metavars.get("BACKEND", "julia")
    if backend == "juliacall":
        if not process_lock:
            raise RuntimeError(
                "The juliacall backend requires the process lock; "
                "use BACKEND='julia' instead"
            )
        raise NotImplementedError("The juliacall backend is not available yet")
    if backend != "julia":
        raise ValueError(f"Unknown Julia backend: {backend!r}")

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

    job_pins = {}
    job_filesystem = {}
    pin_specs = []
    for index, (name, value) in enumerate(pins.items()):
        if not isinstance(name, str):
            raise TypeError("Julia handler pin names must be strings")
        entry = filesystem.get(name, {})
        if not isinstance(entry, Mapping):
            raise TypeError(f"Julia handler filesystem entry for {name!r} must be a mapping")
        celltype = entry.get("celltype", "mixed")
        if celltype in deep_celltypes:
            raise ValueError(
                f"Julia handler cannot accept deep or folder pin celltype {celltype!r}"
            )

        job_name = f"seamless_pin_{index:04d}"
        on_file = bool(entry.get("filesystem", False))
        if on_file:
            if celltype not in file_celltypes:
                raise TypeError(
                    f"Julia handler cannot transport file pin {name!r} "
                    f"with celltype {celltype!r}"
                )
            if not isinstance(value, (str, os.PathLike)):
                raise TypeError(f"Julia handler file pin {name!r} must be a path")
            job_pins[job_name] = os.fspath(value)
        else:
            buffer_celltype = celltype if celltype in file_celltypes else "plain"
            job_pins[job_name] = (
                value if isinstance(value, Buffer) else Buffer(value, buffer_celltype)
            )
        job_filesystem[job_name] = {
            "mode": "file",
            "optional": True,
            "filesystem": on_file,
            "celltype": celltype,
        }
        pin_specs.append(
            {"name": name, "file": job_name, "celltype": celltype}
        )

    key_material = json.dumps(
        [code, output_name, sorted(pins)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    cache_key = hashlib.sha256(key_material).hexdigest()
    config = {
        "cache_key": cache_key,
        "output_name": output_name,
        "output_celltype": output_celltype,
        "pins": pin_specs,
    }

    runtime_source = r'''module SeamlessRuntime
using JSON, NPZ

const ISOLATIONS = Dict{String,Module}()
const NPY_MAGIC = UInt8[0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]
const MIXED_MAGIC = Vector{UInt8}(codeunits("SEAMLESS-MIXED"))

function has_prefix(data::Vector{UInt8}, prefix::Vector{UInt8})
    length(data) < length(prefix) && return false
    return data[1:length(prefix)] == prefix
end

function read_npy(data::Vector{UInt8})
    path = tempname()
    try
        open(path, "w") do io
            write(io, data)
        end
        return NPZ.npzread(path)
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
    elseif celltype == "binary"
        return read_npy(data)
    elseif celltype == "mixed"
        if has_prefix(data, MIXED_MAGIC)
            error("Julia CLI backend cannot read a Seamless mixed pin container")
        elseif has_prefix(data, NPY_MAGIC)
            return read_npy(data)
        else
            return JSON.parse(String(data))
        end
    else
        return JSON.parse(String(data))
    end
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

function encode_result(result, output_celltype::String)
    if output_celltype == "bytes"
        result isa AbstractVector{UInt8} ||
            error("Julia bytes output must be a Vector{UInt8}")
        return "bytes", collect(UInt8, result)
    elseif output_celltype in ("binary", "mixed") &&
           result isa AbstractArray && eltype(result) <: Number
        path = tempname()
        try
            NPZ.npzwrite(path, Array(result))
            return "npy", read(path)
        finally
            isfile(path) && rm(path; force=true)
        end
    else
        return "json", Vector{UInt8}(codeunits(JSON.json(result)))
    end
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

end # module SeamlessRuntime
'''
    runner_source = r'''using JSON
include("runtime.jl")

config = JSON.parsefile("pins.json")
raw_pins = Dict{String,Vector{UInt8}}()
pin_celltypes = Dict{String,String}()
for pin in config["pins"]
    raw_pins[pin["name"]] = read(pin["file"])
    pin_celltypes[pin["name"]] = pin["celltype"]
end

kind, result = SeamlessRuntime.run(
    config["cache_key"],
    read("transformer.jl", String),
    raw_pins,
    pin_celltypes,
    config["output_name"],
    config["output_celltype"],
)
open("RESULT", "w") do io
    write(io, result)
end
'''

    job_pins.update(
        {
            "META__FILE__runtime.jl": runtime_source,
            "META__FILE__runner.jl": runner_source,
            "META__FILE__transformer.jl": code,
            "META__FILE__pins.json": json.dumps(config, ensure_ascii=False),
        }
    )
    pin_names = sorted(job_pins)
    conda_environment = (environment or {}).get("conda_environment", "")
    result = execute_bash(
        "julia --startup-file=no runner.jl",
        pin_names,
        conda_environment,
        job_pins,
        job_filesystem,
        output_celltype,
    )
    if output_celltype in {"binary", "mixed"} and isinstance(result, np.ndarray):
        return np.ascontiguousarray(result)
    return result


register_language_handler("julia", julia, metavars={"BACKEND": "julia"})


__all__ = ["julia"]
