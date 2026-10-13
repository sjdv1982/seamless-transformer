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
    from seamless import Buffer, Checksum
    from seamless_transformer.execute_bash import (
        _validate_container_deepfolder_members,
        write_directory_pin,
    )
    from seamless_transformer.languages import execute_cli
    from seamless_transformer.wire_format import (
        decode_jsonarray,
        decode_result_payload,
        read_wire_directory,
        write_checksum_pin_index,
        write_deepcell_pin_index,
    )

    if not isinstance(metavars, Mapping):
        metavars = {}
    backend = metavars.get("BACKEND", "cli")
    if backend not in ("cli", "mem"):
        raise ValueError(f"Unknown R backend: {backend!r}")
    if backend == "mem" and not process_lock:
        raise RuntimeError(
            "The mem backend requires the process lock; use BACKEND='cli' instead"
        )
    if not isinstance(pins, Mapping):
        raise TypeError("R handler pins must be a mapping")
    if not isinstance(filesystem, Mapping):
        filesystem = {}

    deep_celltypes = {"deepcell", "deepfolder", "folder"}
    file_celltypes = {"mixed", "plain", "text", "bytes", "binary"}
    environment = environment if isinstance(environment, Mapping) else {}
    if backend == "mem":
        from seamless_transformer.languages import check_in_memory_environment

        check_in_memory_environment(environment, "r")

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

    def read_wire_files(directory):
        files = {}
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                path = os.path.join(root, filename)
                name = os.path.relpath(path, directory).replace(os.sep, "/")
                with open(path, "rb") as handle:
                    files[name] = handle.read()
        return read_wire_directory(files, output_celltype)

    pin_specs = []
    pin_data = {}
    pin_is_filesystem = {}
    pin_directory_values = {}
    for index, (name, value) in enumerate(pins.items()):
        if not isinstance(name, str):
            raise TypeError("R handler pin names must be strings")
        entry = filesystem.get(name, {})
        if not isinstance(entry, Mapping):
            raise TypeError(f"R handler filesystem entry for {name!r} must be a mapping")
        celltype = entry.get("celltype", "mixed")
        on_file = bool(entry.get("filesystem", False))
        if celltype == "folder":
            pin_directory_values[name] = value
            pin_is_filesystem[name] = False
            pin_specs.append(
                {
                    "name": name,
                    "file": f"seamless_pin_{index:04d}",
                    "celltype": celltype,
                    "array_type": None,
                    "array_dtype": None,
                }
            )
            continue
        if celltype == "deepcell":
            raw = write_deepcell_pin_index(value)
        elif celltype == "checksum":
            raw = write_checksum_pin_index(value)
        elif celltype == "deepfolder":
            if backend == "cli" and environment.get("docker") is not None:
                from seamless_transformer.transformation_namespace import (
                    _get_read_folder_directories,
                )

                _validate_container_deepfolder_members(
                    value, _get_read_folder_directories()
                )
            raw = json.dumps(
                encode_tree(value), ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        elif on_file:
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
            if backend == "cli" and (array.dtype == np.float32 or array.ndim > 2):
                reason = "float32" if array.dtype == np.float32 else "more than two dimensions"
                raise RuntimeError(
                    f"RcppCNPy cannot read arrays with {reason}; use BACKEND='mem'"
                )
            array_dtype = str(array.dtype)
            array_type = "integer" if array.dtype.kind in "iub" else "numeric"
            if backend == "cli" and array.dtype.kind in "iub":
                # RcppCNPy 0.2.15 misreads NPY int32 when asked for either
                # integer or numeric. R's shared native representation is
                # double, so materialize that canonical dtype for the CLI.
                array_type = "numeric"
                array = array.astype(np.float64)
                normalized = io.BytesIO()
                np.save(normalized, array, allow_pickle=False)
                raw = normalized.getvalue()
                on_file = False
            if backend == "mem":
                raw = (
                    b"SEAMLESS-R-JSON\n"
                    + json.dumps(
                        encode_tree(array),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
        elif celltype == "mixed" and raw.startswith(b"\x94SEAMLESS-MIXED"):
            if backend == "cli":
                raise RuntimeError(
                    "The R CLI backend cannot read a Seamless-mixed pin container"
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
seamless_checksum <- function(value) {
    if (!is.character(value) || length(value) != 1L ||
        !grepl("^[0-9a-f]{64}$", value)) {
        stop("seamless_checksum expects 64 lowercase hexadecimal characters")
    }
    structure(value, class = "seamless_checksum")
}

is_seamless_checksum <- function(value) inherits(value, "seamless_checksum")

contains_seamless_checksum <- function(value) {
    if (is_seamless_checksum(value)) return(TRUE)
    if (is.list(value)) {
        return(any(vapply(value, contains_seamless_checksum, logical(1))))
    }
    FALSE
}

as.character.seamless_checksum <- function(x, ...) unclass(x)

print.seamless_checksum <- function(x, ...) {
    cat("Checksum(", unclass(x), ")\n", sep = "")
    invisible(x)
}

decode_wire_pin <- function(data, celltype) {
    index <- jsonlite::fromJSON(rawToChar(data), simplifyVector = FALSE)
    decode_checksum <- function(value, label) {
        if (!is.list(value) || length(value) != 1L ||
            !identical(names(value), "__checksum__")) {
            stop(paste0("Malformed checksum entry ", label))
        }
        seamless_checksum(value[["__checksum__"]])
    }
    if (identical(celltype, "checksum")) {
        return(decode_checksum(index[["value"]], "value"))
    }
    members <- index[["members"]]
    if (!is.list(members)) stop("Malformed deepcell pin index")
    for (key in names(members)) members[[key]] <- decode_checksum(members[[key]], key)
    members
}

read_folder_pin <- function(path) {
    files <- list.files(path, recursive = TRUE, full.names = TRUE, all.files = TRUE)
    files <- files[!dir.exists(files)]
    values <- lapply(files, function(filename) {
        size <- file.info(filename)$size
        con <- file(filename, open = "rb")
        tryCatch(readBin(con, what = "raw", n = size), finally = close(con))
    })
    names(values) <- substring(files, nchar(path) + 2L)
    values
}

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
    if (identical(celltype, "deepcell") || identical(celltype, "checksum"))
        return(decode_wire_pin(data, celltype))
    if (identical(celltype, "deepfolder"))
        return(jsonlite::fromJSON(rawToChar(data), simplifyVector = FALSE))
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

encode_tree <- function(value) {
    if (is.raw(value)) {
        return(list("__seamless_bytes__" = TRUE,
                    values = as.list(as.integer(value))))
    }
    dims <- dim(value)
    is_array <- (is.numeric(value) || is.integer(value) || is.logical(value)) &&
        (length(value) > 1L || !is.null(dims))
    if (is_array) {
        shape <- if (is.null(dims)) length(value) else dims
        dtype <- if (is.logical(value)) "bool" else if (is.integer(value)) "int64" else "float64"
        values <- if (is.logical(value)) as.list(as.logical(as.vector(value)))
                  else if (is.integer(value)) as.list(as.integer(as.vector(value)))
                  else as.list(as.numeric(as.vector(value)))
        payload <- list(TRUE, dtype, as.list(as.integer(shape)), values)
        names(payload) <- c("__seamless_array__", "dtype", "shape", "values")
        return(payload)
    }
    if (is.list(value)) {
        for (i in seq_along(value)) value[[i]] <- encode_tree(value[[i]])
    }
    value
}

is_json_value <- function(value) {
    if (is.object(value)) return(FALSE)
    if (is.null(value) || is.atomic(value)) return(TRUE)
    if (is.list(value)) {
        return(all(vapply(value, is_json_value, logical(1))))
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
    if (identical(output_celltype, "mixed") && is.raw(value)) {
        return(as.character(jsonlite::toJSON(
            encode_tree(value), auto_unbox = TRUE, null = "null", digits = 17
        )))
    }
    if (identical(output_celltype, "mixed")) value <- encode_tree(value)
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

write_wire_result <- function(value, output_celltype, directory) {
    if (!is.list(value) || is.null(names(value)))
        stop("R deep results must be a named list")
    keys <- names(value)
    if (anyNA(keys)) stop("R deep result keys must be strings")
    dir.create(directory, recursive = TRUE, showWarnings = FALSE)
    file_index <- 0L
    write_wire_file <- function(data, suffix) {
        filename <- sprintf("m%06d%s", file_index, suffix)
        file_index <<- file_index + 1L
        writeBin(data, file.path(directory, filename))
        filename
    }
    members <- list()
    json_values <- list()
    for (key in keys) {
        item <- value[[key]]
        if (is_seamless_checksum(item)) {
            members[[key]] <- list("__checksum__" = as.character(item))
        } else if (identical(output_celltype, "folder")) {
            if (is.raw(item)) bytes <- item
            else stop(paste0("Folder member ", key, " is not bytes"))
            members[[key]] <- write_wire_file(bytes, ".bytes")
        } else if (is.raw(item)) {
            members[[key]] <- write_wire_file(item, ".bytes")
        } else {
            dims <- dim(item)
            is_array <- (is.numeric(item) || is.integer(item) || is.logical(item)) &&
                (length(item) > 1L || !is.null(dims))
            if (is_array) {
                shape <- if (is.null(dims)) length(item) else dims
                dtype <- if (is.logical(item)) "bool" else if (is.integer(item)) "int64" else "float64"
                values <- if (is.logical(item)) as.list(as.logical(as.vector(item)))
                          else if (is.integer(item)) as.list(as.integer(as.vector(item)))
                          else as.list(as.numeric(as.vector(item)))
                payload <- list(dtype = dtype, shape = as.list(as.integer(shape)), values = values)
                members[[key]] <- write_wire_file(
                    charToRaw(as.character(jsonlite::toJSON(
                        payload, auto_unbox = TRUE, null = "null", digits = 17
                    ))),
                    ".jsonarray"
                )
            } else {
                if (!is_json_value(item)) {
                    stop(paste0("Unsupported deep result value for member ", key))
                }
                json_values[key] <- list(item)
            }
        }
    }
    member_json <- if (length(members)) {
        as.character(jsonlite::toJSON(members, auto_unbox = TRUE, null = "null"))
    } else "{}"
    json_field <- ""
    if (length(json_values)) {
        json_name <- write_wire_file(
            charToRaw(as.character(jsonlite::toJSON(
                json_values, auto_unbox = TRUE, null = "null", digits = 17, force = TRUE
            ))),
            ".json"
        )
        json_field <- paste0(",\"json\":", as.character(jsonlite::toJSON(json_name, auto_unbox = TRUE)))
    }
    index_text <- paste0("{\"members\":", member_json, json_field, "}")
    writeBin(charToRaw(enc2utf8(index_text)), file.path(directory, "index.json"))
    invisible(NULL)
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
        if (identical(pin[["celltype"]], "folder") && dir.exists(pin[["file"]])) {
            pins[[pin[["name"]]]] <- read_folder_pin(pin[["file"]])
        } else {
            size <- file.info(pin[["file"]])$size
            con <- file(pin[["file"]], open = "rb")
            data <- tryCatch(
                readBin(con, what = "raw", n = size),
                finally = close(con)
            )
            pins[[pin[["name"]]]] <- decode_pin(data, pin)
        }
    }

    if (is_transform(code)) {
        key <- config[["cache_key"]]
        if (!exists(key, envir = ISOLATIONS, inherits = FALSE)) {
            isolation <- new.env(parent = globalenv())
            assign("seamless_checksum", seamless_checksum, envir = isolation)
            assign("is_seamless_checksum", is_seamless_checksum, envir = isolation)
            eval(parse(text = code), envir = isolation)
            assign(key, isolation, envir = ISOLATIONS)
        }
        isolation <- get(key, envir = ISOLATIONS, inherits = FALSE)
        fun <- get("transform", envir = isolation, inherits = FALSE)
        result <- do.call(fun, pins)
    } else {
        isolation <- new.env(parent = globalenv())
        assign("seamless_checksum", seamless_checksum, envir = isolation)
        assign("is_seamless_checksum", is_seamless_checksum, envir = isolation)
        list2env(pins, envir = isolation)
        eval(parse(text = code), envir = isolation)
        name <- config[["output_name"]]
        if (!exists(name, envir = isolation, inherits = FALSE))
            stop(paste0("R transformer did not define '", name, "'"))
        result <- get(name, envir = isolation, inherits = FALSE)
    }
    output_celltype <- config[["output_celltype"]]
    if (output_celltype %in% c("folder", "deepcell")) {
        write_wire_result(result, output_celltype, config[["result_directory"]])
        return("")
    }
    if (contains_seamless_checksum(result) &&
        !(output_celltype %in% c("checksum", "deepcell", "folder"))) {
        stop("A Seamless checksum can only be returned under the 'checksum' or deep celltype")
    }
    if (is_seamless_checksum(result)) {
        if (!identical(output_celltype, "checksum"))
            stop("A Seamless checksum can only be returned under the 'checksum' celltype")
        return(as.character(result))
    }
    encode_result(result, output_celltype)
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
if (!dir.exists("RESULT")) writeBin(charToRaw(enc2utf8(result)), "RESULT")
'''

    if backend == "cli":
        config["result_directory"] = "RESULT"
        job_pins = {}
        job_filesystem = {}
        for spec in pin_specs:
            name = spec["name"]
            job_name = spec["file"]
            if spec["celltype"] == "folder":
                job_pins[job_name] = pin_directory_values[name]
                job_filesystem[job_name] = {
                    "mode": "directory",
                    "optional": True,
                    "filesystem": False,
                    "celltype": "folder",
                    "members_are_checksums": False,
                }
            elif pin_is_filesystem[name]:
                job_pins[job_name] = os.fspath(pins[name])
                job_filesystem[job_name] = {
                    "mode": "file",
                    "optional": True,
                    "filesystem": True,
                    "celltype": "bytes",
                }
            else:
                job_pins[job_name] = Buffer(pin_data[name], "bytes")
                job_filesystem[job_name] = {
                    "mode": "file",
                    "optional": True,
                    "filesystem": False,
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
        cli_outputpin = "folder" if output_celltype in deep_celltypes else "plain"
        result = execute_cli(
            "Rscript --vanilla runner.R",
            sorted(job_pins),
            job_pins,
            job_filesystem,
            cli_outputpin,
            environment,
        )
        if output_celltype in deep_celltypes:
            result = read_wire_directory(result, output_celltype)
        else:
            result = decode_result_payload(result)
    else:
        import ryp

        token = secrets.token_hex(12)
        input_name = f".seamless_r_input_{token}"
        code_name = f".seamless_r_code_{token}"
        result_name = f".seamless_r_result_{token}"
        with tempfile.TemporaryDirectory(prefix="seamless-ryp-") as tempdir:
            config_path = os.path.join(tempdir, "pins.json")
            result_directory = os.path.join(tempdir, "RESULT")
            config["result_directory"] = result_directory
            code_path = os.path.join(tempdir, "transformer.R")
            stdout_path = os.path.join(tempdir, "stdout")
            stderr_path = os.path.join(tempdir, "stderr")
            for spec in pin_specs:
                name = spec["name"]
                spec["file"] = os.path.join(tempdir, spec["file"])
                if spec["celltype"] == "folder":
                    write_directory_pin(
                        spec["file"],
                        pin_directory_values[name],
                        "folder",
                        members_are_checksums=False,
                    )
                else:
                    Path(spec["file"]).write_bytes(pin_data[name])
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
            if output_celltype in deep_celltypes:
                result = read_wire_files(result_directory)
            elif output_celltype == "checksum":
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    pass
            else:
                result = decode_result_payload(json.loads(result))

    if isinstance(result, Mapping) and result.get("__seamless_error__"):
        raise RuntimeError(f"R transformer failed: {result['__seamless_error__']}")
    if output_celltype == "checksum" and isinstance(result, str):
        return Checksum(result)
    if isinstance(result, Mapping) and result.get("__seamless_bytes__") is True:
        return bytes(result["values"])
    if isinstance(result, Mapping) and result.get("__seamless_array__") is True:
        return np.ascontiguousarray(decode_jsonarray(result))
    return result


from seamless_transformer.languages import register_language_handler

register_language_handler("r", r, metavars={"BACKEND": "cli"})
