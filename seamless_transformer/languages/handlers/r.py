def r(
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
    import io
    import json
    import os
    import secrets
    import sys
    import tempfile
    from collections.abc import Mapping
    from pathlib import Path

    import numpy as np
    from seamless import Buffer
    from seamless_transformer.execute_bash import execute_bash

    if not isinstance(metavars, Mapping):
        metavars = {}
    backend = metavars.get("BACKEND", "Rscript")
    if backend not in ("Rscript", "ryp"):
        raise ValueError(f"Unknown R backend: {backend!r}")
    if backend == "ryp" and not process_lock:
        raise RuntimeError(
            "The ryp backend requires the process lock; use BACKEND='Rscript' instead"
        )
    if not isinstance(pins, Mapping):
        raise TypeError("R handler pins must be a mapping")
    if not isinstance(filesystem, Mapping):
        filesystem = {}

    deep_celltypes = {"deepcell", "deepfolder", "folder"}
    file_celltypes = {"mixed", "plain", "text", "bytes", "binary"}
    if output_celltype in deep_celltypes:
        raise ValueError(
            f"R handler cannot return deep or folder celltype {output_celltype!r}"
        )
    environment = environment if isinstance(environment, Mapping) else {}
    if backend == "ryp":
        if environment.get("docker"):
            raise RuntimeError("The ryp backend cannot run in a Docker environment")
        conda_environment = environment.get("conda_environment")
        if conda_environment:
            if "/" in str(conda_environment):
                matches = os.path.realpath(os.environ.get("CONDA_PREFIX", "")) == os.path.realpath(
                    str(conda_environment)
                )
            else:
                matches = os.environ.get("CONDA_DEFAULT_ENV") == str(conda_environment)
            if not matches:
                raise RuntimeError(
                    "The ryp backend requires the process to run in "
                    f"conda environment {conda_environment!r}"
                )

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
                dtype = "int64" if array.dtype.kind == "i" else "uint64"
                array = array.astype(dtype, copy=False)
            elif array.dtype.kind == "f":
                dtype = "float64"
                array = array.astype(np.float64, copy=False)
            elif array.dtype.kind == "b":
                dtype = "bool"
                array = array.astype(np.bool_, copy=False)
            else:
                raise TypeError(f"R handler cannot transport array dtype {array.dtype}")
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
            return {"__seamless_bytes__": True, "values": list(bytes(value))}
        return value

    pin_specs = []
    pin_data = {}
    pin_is_filesystem = {}
    for index, (name, value) in enumerate(pins.items()):
        if not isinstance(name, str):
            raise TypeError("R handler pin names must be strings")
        entry = filesystem.get(name, {})
        if not isinstance(entry, Mapping):
            raise TypeError(f"R handler filesystem entry for {name!r} must be a mapping")
        celltype = entry.get("celltype", "mixed")
        if celltype in deep_celltypes:
            raise ValueError(
                f"R handler cannot accept deep or folder pin celltype {celltype!r}"
            )
        on_file = bool(entry.get("filesystem", False))
        if on_file:
            if celltype not in file_celltypes:
                raise TypeError(
                    f"R handler cannot transport file pin {name!r} "
                    f"with celltype {celltype!r}"
                )
            if not isinstance(value, (str, os.PathLike)):
                raise TypeError(f"R handler file pin {name!r} must be a path")
            raw = read_pin_bytes(value, celltype, entry)
        else:
            raw = read_pin_bytes(value, celltype, entry)

        array_type = None
        array_dtype = None
        container_celltype = celltype in {"binary", "mixed"}
        if container_celltype and raw.startswith(b"\x93NUMPY"):
            array = np.load(io.BytesIO(raw), allow_pickle=False)
            if backend == "Rscript" and (array.dtype == np.float32 or array.ndim > 2):
                reason = "float32" if array.dtype == np.float32 else "more than two dimensions"
                raise RuntimeError(
                    f"RcppCNPy cannot read arrays with {reason}; use BACKEND='ryp'"
                )
            array_dtype = str(array.dtype)
            array_type = "integer" if array.dtype.kind in "iub" else "numeric"
            if backend == "Rscript" and array.dtype.kind in "iub":
                # RcppCNPy 0.2.15 misreads NPY int32 when asked for either
                # integer or numeric. R's shared native representation is
                # double, so materialize that canonical dtype for the CLI.
                array_type = "numeric"
                array = array.astype(np.float64)
                normalized = io.BytesIO()
                np.save(normalized, array, allow_pickle=False)
                raw = normalized.getvalue()
                on_file = False
            if backend == "ryp":
                raw = (
                    b"SEAMLESS-R-JSON\n"
                    + json.dumps(
                        encode_tree(array),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
        elif celltype == "mixed" and raw.startswith(b"\x94SEAMLESS-MIXED"):
            if backend == "Rscript":
                raise RuntimeError(
                    "The Rscript backend cannot read a Seamless-mixed pin container"
                )
            from seamless.util.mixed.io import deserialize

            raw = (
                b"SEAMLESS-R-JSON\n"
                + json.dumps(
                    encode_tree(deserialize(raw)[0]),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )

        pin_data[name] = raw
        pin_is_filesystem[name] = on_file
        pin_specs.append(
            {
                "name": name,
                "file": f"seamless_pin_{index:04d}",
                "celltype": celltype,
                "array_type": array_type,
                "array_dtype": array_dtype,
            }
        )

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
    runtime_source = r'''
decode_tree <- function(value) {
    if (is.list(value) && !is.null(value[["__seamless_array__"]]) &&
        isTRUE(value[["__seamless_array__"]])) {
        shape <- as.integer(unlist(value[["shape"]], use.names = FALSE))
        vals <- unlist(value[["values"]], use.names = FALSE)
        dtype <- value[["dtype"]]
        if (identical(dtype, "bool")) vals <- as.logical(vals)
        else vals <- as.numeric(vals)
        if (length(shape) <= 1L) return(vals)
        return(array(vals, dim = shape))
    }
    if (is.list(value) && !is.null(value[["__seamless_bytes__"]]) &&
        isTRUE(value[["__seamless_bytes__"]])) {
        return(as.raw(unlist(value[["values"]], use.names = FALSE)))
    }
    if (is.list(value)) {
        for (i in seq_along(value)) value[[i]] <- decode_tree(value[[i]])
    }
    value
}

decode_pin <- function(data, pin) {
    celltype <- pin[["celltype"]]
    if (identical(celltype, "bytes")) return(data)
    if (identical(celltype, "text")) {
        value <- rawToChar(data)
        if (endsWith(value, "\n")) value <- substr(value, 1L, nchar(value) - 1L)
        return(value)
    }
    marker <- charToRaw("SEAMLESS-R-JSON\n")
    if ((identical(celltype, "binary") || identical(celltype, "mixed")) &&
        length(data) >= length(marker) &&
        identical(data[seq_along(marker)], marker)) {
        text <- rawToChar(data[-seq_along(marker)])
        return(decode_tree(jsonlite::fromJSON(text, simplifyVector = FALSE)))
    }
    if (identical(celltype, "binary") || identical(celltype, "mixed")) {
        is_npy <- length(data) >= 6L &&
            identical(data[1:6], as.raw(c(0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59)))
        if (is_npy) {
            type <- pin[["array_type"]]
            if (is.null(type) || is.na(type)) type <- "numeric"
            value <- RcppCNPy::npyLoad(pin[["file"]], type = type, dotranspose = TRUE)
            dimensions <- dim(value)
            if (identical(pin[["array_dtype"]], "bool")) value <- as.logical(value)
            else value <- as.numeric(value)
            if (!is.null(dimensions)) dim(value) <- dimensions
            return(value)
        }
        if (identical(celltype, "binary")) stop("R binary pin is not a NumPy array")
    }
    jsonlite::fromJSON(rawToChar(data))
}

is_transform <- function(code) {
    expressions <- parse(text = code)
    for (expr in as.list(expressions)) {
        if (!is.call(expr) || length(expr) < 3L) next
        op <- as.character(expr[[1L]])
        if (!(op %in% c("<-", "="))) next
        lhs <- expr[[2L]]
        rhs <- expr[[3L]]
        if (is.symbol(lhs) && identical(as.character(lhs), "transform") &&
            is.call(rhs) && identical(rhs[[1L]], as.name("function"))) return(TRUE)
    }
    FALSE
}

encode_result <- function(value, output_celltype) {
    if (identical(output_celltype, "bytes")) {
        if (is.raw(value)) bytes <- as.integer(value)
        else if (is.character(value) && length(value) == 1L)
            bytes <- as.integer(charToRaw(enc2utf8(value)))
        else if (is.numeric(value) || is.integer(value))
            bytes <- as.integer(value)
        else stop("R bytes output must be raw, numeric or a string")
        payload <- list(TRUE, bytes)
        names(payload) <- c("__seamless_bytes__", "values")
        return(as.character(jsonlite::toJSON(
            payload, auto_unbox = TRUE, null = "null", digits = 17
        )))
    }
    dims <- dim(value)
    is_array <- (is.numeric(value) || is.integer(value) || is.logical(value)) &&
        (length(value) > 1L || !is.null(dims))
    if ((identical(output_celltype, "binary") ||
         identical(output_celltype, "mixed")) && is_array) {
        shape <- if (is.null(dims)) length(value) else dims
        dtype <- if (is.logical(value)) "bool" else "float64"
        values <- if (is.logical(value)) as.list(as.logical(as.vector(value)))
                 else as.list(as.numeric(as.vector(value)))
        payload <- list(
            TRUE, dtype, as.list(as.integer(shape)), values
        )
        names(payload) <- c("__seamless_array__", "dtype", "shape", "values")
        return(as.character(jsonlite::toJSON(
            payload, auto_unbox = TRUE, null = "null", digits = 17
        )))
    }
    as.character(jsonlite::toJSON(
        value, auto_unbox = TRUE, null = "null", digits = 17, force = TRUE
    ))
}

run <- function(config_path, code_path, stdout_path = NULL, stderr_path = NULL) {
    stdout_connection <- if (!is.null(stdout_path)) file(stdout_path, open = "wt") else NULL
    stderr_connection <- if (!is.null(stderr_path)) file(stderr_path, open = "wt") else NULL
    if (!is.null(stdout_connection)) sink(stdout_connection, type = "output")
    if (!is.null(stderr_connection)) sink(stderr_connection, type = "message")
    on.exit({
        if (!is.null(stderr_connection)) sink(type = "message")
        if (!is.null(stdout_connection)) sink(type = "output")
        if (!is.null(stderr_connection)) close(stderr_connection)
        if (!is.null(stdout_connection)) close(stdout_connection)
    }, add = TRUE)

    config_text <- readChar(config_path, file.info(config_path)$size, useBytes = TRUE)
    config <- jsonlite::fromJSON(config_text, simplifyVector = FALSE)
    code <- readChar(code_path, file.info(code_path)$size, useBytes = TRUE)
    pins <- list()
    for (pin in config[["pins"]]) {
        size <- file.info(pin[["file"]])$size
        con <- file(pin[["file"]], open = "rb")
        data <- tryCatch(
            readBin(con, what = "raw", n = size),
            finally = close(con)
        )
        pins[[pin[["name"]]]] <- decode_pin(data, pin)
    }

    if (is_transform(code)) {
        key <- config[["cache_key"]]
        if (!exists(key, envir = ISOLATIONS, inherits = FALSE)) {
            isolation <- new.env(parent = globalenv())
            eval(parse(text = code), envir = isolation)
            assign(key, isolation, envir = ISOLATIONS)
        }
        isolation <- get(key, envir = ISOLATIONS, inherits = FALSE)
        fun <- get("transform", envir = isolation, inherits = FALSE)
        result <- do.call(fun, pins)
    } else {
        isolation <- new.env(parent = globalenv())
        list2env(pins, envir = isolation)
        eval(parse(text = code), envir = isolation)
        name <- config[["output_name"]]
        if (!exists(name, envir = isolation, inherits = FALSE))
            stop(paste0("R transformer did not define '", name, "'"))
        result <- get(name, envir = isolation, inherits = FALSE)
    }
    encode_result(result, config[["output_celltype"]])
}

error_json <- function(error) {
    payload <- list(conditionMessage(error))
    names(payload) <- "__seamless_error__"
    as.character(jsonlite::toJSON(payload, auto_unbox = TRUE))
}

ISOLATIONS <- new.env(parent = emptyenv())
'''
    runner_source = r'''
runtime <- new.env(parent = globalenv())
eval(parse(text = readChar("runtime.R", file.info("runtime.R")$size, useBytes = TRUE)),
     envir = runtime)
result <- runtime$run("pins.json", "transformer.R")
writeBin(charToRaw(enc2utf8(result)), "RESULT")
'''

    if backend == "Rscript":
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
                "META__FILE__runtime.R": runtime_source,
                "META__FILE__runner.R": runner_source,
                "META__FILE__transformer.R": code,
                "META__FILE__pins.json": json.dumps(config, ensure_ascii=False),
            }
        )
        result = execute_bash(
            "Rscript --vanilla runner.R",
            sorted(job_pins),
            environment.get("conda_environment", ""),
            job_pins,
            job_filesystem,
            "plain",
        )
    else:
        import ryp

        token = secrets.token_hex(12)
        input_name = f".seamless_r_input_{token}"
        code_name = f".seamless_r_code_{token}"
        result_name = f".seamless_r_result_{token}"
        with tempfile.TemporaryDirectory(prefix="seamless-ryp-") as tempdir:
            config_path = os.path.join(tempdir, "pins.json")
            code_path = os.path.join(tempdir, "transformer.R")
            stdout_path = os.path.join(tempdir, "stdout")
            stderr_path = os.path.join(tempdir, "stderr")
            for spec in pin_specs:
                spec["file"] = os.path.join(tempdir, spec["file"])
                Path(spec["file"]).write_bytes(pin_data[spec["name"]])
            config["pins"] = pin_specs
            Path(config_path).write_text(
                json.dumps(config, ensure_ascii=False), encoding="utf-8"
            )
            Path(code_path).write_text(code, encoding="utf-8")
            Path(stdout_path).touch()
            Path(stderr_path).touch()
            r_literal = json.dumps
            r_code = f'''
if (!exists(".seamless_r_runtime", envir = .GlobalEnv, inherits = FALSE)) {{
    .runtime <- new.env(parent = globalenv())
    eval(parse(text = {r_literal(runtime_source, ensure_ascii=False)}), envir = .runtime)
    assign(".seamless_r_runtime", .runtime, envir = .GlobalEnv)
}}
assign("{input_name}", {r_literal(config_path)}, envir = .GlobalEnv)
assign("{code_name}", {r_literal(code_path)}, envir = .GlobalEnv)
assign(
    "{result_name}",
    tryCatch(
        get(".seamless_r_runtime", envir = .GlobalEnv)$run(
            get("{input_name}", envir = .GlobalEnv),
            get("{code_name}", envir = .GlobalEnv),
            {r_literal(stdout_path)},
            {r_literal(stderr_path)}
        ),
        error = function(e) get(".seamless_r_runtime", envir = .GlobalEnv)$error_json(e)
    ),
    envir = .GlobalEnv
)
'''
            ryp.r(r_code)
            statement = (
                "local({ "
                f".value <- get(\"{result_name}\", envir = .GlobalEnv, inherits = FALSE); "
                f"rm(list = c(\"{input_name}\", \"{code_name}\", \"{result_name}\"), "
                "envir = .GlobalEnv); .value })"
            )
            result = ryp.to_py(statement)
            stdout_text = Path(stdout_path).read_text(encoding="utf-8")
            stderr_text = Path(stderr_path).read_text(encoding="utf-8")
            if stdout_text:
                sys.stdout.write(stdout_text)
                sys.stdout.flush()
            if stderr_text:
                sys.stderr.write(stderr_text)
                sys.stderr.flush()
            if isinstance(result, np.ndarray) and result.shape == ():
                result = result.item()
            if not isinstance(result, str):
                raise TypeError("ryp did not return the R runtime's JSON text")
            result = json.loads(result)

    if isinstance(result, Mapping) and result.get("__seamless_error__"):
        raise RuntimeError(f"R transformer failed: {result['__seamless_error__']}")
    if isinstance(result, Mapping) and result.get("__seamless_bytes__") is True:
        return bytes(result["values"])
    if isinstance(result, Mapping) and result.get("__seamless_array__") is True:
        dtype = result["dtype"]
        numpy_dtype = {"bool": np.bool_, "float64": np.float64}[dtype]
        array = np.asarray(result["values"], dtype=numpy_dtype).reshape(
            tuple(result["shape"]), order="F"
        )
        return np.ascontiguousarray(array)
    return result


from seamless_transformer.languages import register_language_handler

register_language_handler("r", r, metavars={"BACKEND": "Rscript"})
