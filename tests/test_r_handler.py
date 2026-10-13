"""R backends execute successful computations before checksum comparison."""
import gzip
import importlib
import os
import shutil
import subprocess

import numpy as np
import pytest
from seamless import Buffer, Checksum
from seamless_transformer import Transformer
from seamless_transformer.run import run_transformation_dict
from seamless_transformer.transformation_utils import extract_tf_dunder


@pytest.fixture(params=["cli", "mem"])
def backend(request):
    if request.param == "mem":
        if importlib.util.find_spec("ryp") is None:
            pytest.skip("ryp missing")
    else:
        executable = shutil.which("Rscript")
        if executable is None:
            pytest.skip("Rscript missing")
        probe = subprocess.run([executable, "--vanilla", "-e", "library(jsonlite); library(RcppCNPy)"],
                               capture_output=True, text=True, timeout=60)
        if probe.returncode:
            pytest.skip("R packages missing: " + probe.stderr[-1000:])
    return request.param


def builder(backend, code, value=7, celltype="plain", output="plain"):
    tf = Transformer("r")
    tf.metavars.BACKEND = backend
    tf.code = code
    tf.celltypes.value = celltype
    tf.pins.value = value
    tf.celltypes.result = output
    tf.local = True
    return tf


def execution(tf):
    handle = tf.build()
    checksum = handle.construct()
    assert checksum is not None, handle.exception
    payload = checksum.resolve().get_value("plain")
    payload.update(extract_tf_dunder(handle._definition_payload_template))
    payload.update(handle._tf_dunder)
    return checksum, payload


def result_checksum(tf):
    checksum, payload = execution(tf)
    result = run_transformation_dict(payload, checksum)
    assert result is not None
    return Checksum(result)


def run(tf):
    buffer = result_checksum(tf).resolve()
    return buffer.content if tf.celltypes.result == "bytes" else buffer.get_value(tf.celltypes.result)


@pytest.mark.parametrize("code", ["result <- value + 2", "transform <- function(value) value + 2",
                                 "transform = function(value) { value + 2 }",
                                 'transform <- "not a function"; result <- value + 2',
                                 "local({transform <- function(value) value + 999}); result <- value + 2"])
def test_code_styles_ast(backend, code):
    assert run(builder(backend, code)) == 9


CASES = [
    ("plain", {"x": [2, 5]}, "value$x[2]", "plain", 5),
    ("mixed", {"x": 4}, "value$x + 1", "plain", 5),
    ("mixed", np.array([2., 5.]), "value + 1", "mixed", np.array([3., 6.])),
    ("text", "hello\n", 'paste0(value, "!")', "plain", "hello!"),
    ("bytes", b"\x00\xff\x01", "rev(value)", "bytes", b"\x01\xff\x00"),
    ("binary", np.array([[1., 2.], [3., 4.]]), "value + 1", "binary", np.array([[2., 3.], [4., 5.]])),
    ("int", 4, "value", "plain", 4),
    ("float", 2.5, "value", "plain", 2.5),
    ("bool", True, "value", "plain", True),
    ("str", "quoted", "value", "plain", "quoted"),
]


@pytest.mark.parametrize("form", ["value", "file", "gzip"])
@pytest.mark.parametrize("style", ["block", "function"])
@pytest.mark.parametrize("celltype,value,expression,output,expected", CASES)
def test_values_and_actual_checksum_parity(backend, form, style, celltype, value,
                                          expression, output, expected, tmp_path, monkeypatch):
    namespace = importlib.import_module("seamless_transformer.transformation_namespace")
    buffer = Buffer(value, celltype)
    if form != "value":
        path = tmp_path / (buffer.get_checksum().hex() + (".gz" if form == "gzip" else ""))
        path.write_bytes(gzip.compress(buffer.content) if form == "gzip" else buffer.content)
    monkeypatch.setattr(namespace, "_get_read_folder_directories",
                        lambda: [] if form == "value" else [str(tmp_path)])
    code = "result <- " + expression if style == "block" else "transform <- function(value) " + expression
    tf = builder(backend, code, value, celltype, output)
    checksum = result_checksum(tf)
    result = checksum.resolve().content if output == "bytes" else checksum.resolve().get_value(output)
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(result, expected)
        assert result.flags.c_contiguous
    else:
        assert result == expected
        assert type(result) is type(expected)
    if backend == "mem":
        cli = builder("cli", code, value, celltype, output)
        assert execution(cli)[0] == execution(tf)[0]
        assert result_checksum(cli) == checksum


@pytest.mark.parametrize("code", ['stop("r-contract-failure")', "unrelated <- value + 1"])
def test_failure_missing_output(backend, code):
    with pytest.raises(Exception):
        run(builder(backend, code))


def test_function_isolation_built_once_without_pins(backend):
    code = '''
stopifnot(!exists("value", inherits=FALSE))
calls <- 0L
transform <- function(value) {
 calls <<- calls + 1L
 c(value + 2, calls)
}
'''
    for index, value in enumerate((5, 9), 1):
        np.testing.assert_array_equal(run(builder(backend, code, value, output="binary")),
                                      [value + 2, index if backend == "mem" else 1])


@pytest.mark.parametrize("form", ["value", "file", "gzip"])
def test_mutating_pin_does_not_write_through(backend, form, tmp_path, monkeypatch):
    value = np.array([3., 4.])
    buffer = Buffer(value, "binary")
    before = buffer.get_checksum()
    path = tmp_path / (before.hex() + (".gz" if form == "gzip" else ""))
    original = gzip.compress(buffer.content) if form == "gzip" else buffer.content
    if form != "value":
        path.write_bytes(original)
    namespace = importlib.import_module("seamless_transformer.transformation_namespace")
    monkeypatch.setattr(namespace, "_get_read_folder_directories",
                        lambda: [] if form == "value" else [str(tmp_path)])
    tf = builder(backend, "value[1] <- 99; result <- value", value, "binary", "binary")
    np.testing.assert_array_equal(run(tf), [99., 4.])
    np.testing.assert_array_equal(value, [3., 4.])
    assert Buffer(value, "binary").get_checksum() == before
    assert execution(tf)[1]["value"][2] == before.hex()
    if form != "value":
        assert path.read_bytes() == original


@pytest.mark.parametrize("value,celltype,code,expected", [
    ({"array": np.array([1., 2.]), "plain": 3}, "mixed", "sum(value$array) + value$plain", 6),
    (np.ones((2, 3, 4)), "binary", "sum(value)", 24),
    (np.array([1., 2.], dtype=np.float32), "binary", "sum(value)", 3),
])
def test_documented_cli_only_transport_failure(backend, value, celltype, code, expected):
    tf = builder(backend, "result <- " + code, value, celltype)
    if backend == "cli":
        with pytest.raises(Exception):
            run(tf)
    else:
        assert run(tf) == expected


@pytest.mark.parametrize("dtype", ["int32", "int64", "float64"])
def test_array_dtype_result_checksum_parity(backend, dtype):
    if backend != "mem":
        pytest.skip("paired computation once per case")
    value = np.arange(6, dtype=dtype).reshape(2, 3)
    for expression in ("value", "value + 1"):
        code = "result <- " + expression
        assert result_checksum(builder("cli", code, value, "binary", "binary")) == result_checksum(
            builder("mem", code, value, "binary", "binary"))


@pytest.mark.parametrize("environment", [{"docker": {"name": "unused"}},
                                        {"conda_environment": "definitely-not-seamless1"}])
def test_embedded_environment_refusal(backend, environment):
    if backend != "mem":
        pytest.skip("embedded environment contract")
    from seamless_transformer.languages import get_language_handler
    with pytest.raises(Exception, match="docker|Docker|conda|environment"):
        get_language_handler("r").handler(code="result <- value", pins={"value": 7},
            filesystem={"value": {"celltype": "plain", "filesystem": False}},
            output_name="result", output_celltype="plain", metavars={"BACKEND": backend},
            environment=environment, process_lock=True)


def test_embedded_current_environment_and_lock_requirement(backend):
    if backend != "mem":
        pytest.skip("embedded lock contract")
    from seamless_transformer.languages import get_language_handler
    handler = get_language_handler("r").handler
    arguments = dict(code="result <- value", pins={"value": 7},
        filesystem={"value": {"celltype": "plain", "filesystem": False}},
        output_name="result", output_celltype="plain", metavars={"BACKEND": backend},
        environment={"conda_environment": os.environ["CONDA_PREFIX"]})
    assert handler(**arguments, process_lock=True) == 7
    with pytest.raises(Exception, match="Rscript|process.lock|process_lock"):
        handler(**arguments, process_lock=False)


def test_stdout_stderr_transformation_logs(backend):
    from seamless_transformer import has_spawned, spawn
    if not has_spawned():
        spawn(1)
    tf = builder(backend, 'cat("R_CONTRACT_OUT_7942\\n"); cat("R_CONTRACT_ERR_7942\\n", file=stderr()); result <- value')
    tf.local = False
    handle = tf.build()
    assert handle.run() == 7
    assert "R_CONTRACT_OUT_7942" in handle.logs
    assert "R_CONTRACT_ERR_7942" in handle.logs


def test_builtin_registration_does_not_initialize_embedded_interpreters():
    import sys
    probe = subprocess.run([sys.executable, "-c", """
import sys
from seamless_transformer.languages import available_languages
assert available_languages()["r"] == "handler"
assert available_languages()["julia"] == "handler"
assert "ryp" not in sys.modules
assert "juliacall" not in sys.modules
assert "_rinterface" not in sys.modules
"""], capture_output=True, text=True, timeout=60)
    assert probe.returncode == 0, probe.stderr


def test_block_isolation_is_fresh(backend):
    code = 'stopifnot(!exists("previous", inherits=FALSE)); previous <- value; result <- value'
    assert run(builder(backend, code, 5)) == 5
    assert run(builder(backend, code, 9)) == 9


@pytest.mark.parametrize("style", ["block", "function"])
def test_mutation_success_checksum_parity(backend, style):
    if backend != "mem":
        pytest.skip("paired computation once per case")
    value = np.array([3., 4.])
    code = "value[1] <- 99; result <- value" if style == "block" else "transform <- function(value) {value[1] <- 99; value}"
    assert result_checksum(builder("cli", code, value, "binary", "binary")) == result_checksum(
        builder("mem", code, value, "binary", "binary"))
    np.testing.assert_array_equal(value, [3., 4.])


@pytest.mark.parametrize("form", ["value", "file", "gzip", "zstd"])
@pytest.mark.parametrize("value", [
    np.array([1, -2, 2147483647], dtype=np.int32),
    np.array([0, 2**53 + 1, 2**64 - 1], dtype=np.uint64),
    np.array([True, False, True], dtype=np.bool_),
])
def test_extreme_integer_and_logical_normalization_parity(backend, form, value,
                                                        tmp_path, monkeypatch):
    if backend != "mem":
        pytest.skip("paired successful executions once per case")
    buffer = Buffer(value, "binary")
    before = buffer.get_checksum()
    content = buffer.content
    suffix = {"value": "", "file": "", "gzip": ".gz", "zstd": ".zst"}[form]
    if form == "gzip":
        content = gzip.compress(content)
    elif form == "zstd":
        try:
            from compression import zstd
            content = zstd.compress(content)
        except ImportError:
            zstandard = pytest.importorskip("zstandard")
            content = zstandard.ZstdCompressor().compress(content)
    path = tmp_path / (before.hex() + suffix)
    if form != "value":
        path.write_bytes(content)
    namespace = importlib.import_module("seamless_transformer.transformation_namespace")
    monkeypatch.setattr(namespace, "_get_read_folder_directories",
                        lambda: [] if form == "value" else [str(tmp_path)])
    expected = value.copy() if value.dtype.kind == "b" else value.astype(np.float64)
    results = []
    for selected in ("cli", "mem"):
        tf = builder(selected, "result <- value", value, "binary", "binary")
        checksum = result_checksum(tf)
        result = checksum.resolve().get_value("binary")
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == expected.dtype
        assert result.flags.c_contiguous
        results.append(checksum)
    assert results[0] == results[1]
    assert Buffer(value, "binary").get_checksum() == before
    if form != "value":
        assert path.read_bytes() == content


@pytest.mark.parametrize("form", ["value", "file", "gzip"])
@pytest.mark.parametrize("payload_kind", ["npy-int32", "npy-float64", "mixed-container"])
def test_bytes_magic_remains_raw_and_checksum_parity(backend, form, payload_kind,
                                                  tmp_path, monkeypatch):
    if payload_kind == "mixed-container":
        value = Buffer({"array": np.array([1., 2.]), "plain": {"x": 3}}, "mixed").content
    else:
        import io
        stream = io.BytesIO()
        np.save(stream, np.array([1, 2, 3], dtype="int32" if payload_kind == "npy-int32" else "float64"),
                allow_pickle=False)
        value = stream.getvalue()
    buffer = Buffer(value, "bytes")
    before = buffer.get_checksum()
    original = buffer.content
    path = tmp_path / (before.hex() + (".gz" if form == "gzip" else ""))
    stored = gzip.compress(original) if form == "gzip" else original
    if form != "value":
        path.write_bytes(stored)
    namespace = importlib.import_module("seamless_transformer.transformation_namespace")
    monkeypatch.setattr(namespace, "_get_read_folder_directories",
                        lambda: [] if form == "value" else [str(tmp_path)])
    tf = builder(backend, "result <- value", value, "bytes", "bytes")
    checksum = result_checksum(tf)
    assert checksum.resolve().content == value
    assert checksum == before
    if backend == "mem":
        cli = builder("cli", "result <- value", value, "bytes", "bytes")
        assert result_checksum(cli) == checksum
    assert Buffer(value, "bytes").get_checksum() == before
    if form != "value":
        assert path.read_bytes() == stored


@pytest.mark.parametrize("form", ["value", "file"])
@pytest.mark.parametrize("celltype", ["plain", "mixed"])
@pytest.mark.parametrize("nested", [False, True], ids=["top-level", "nested"])
@pytest.mark.parametrize("marker", ["__seamless_array__", "__seamless_bytes__"],
                         ids=["array-tag", "bytes-tag"])
def test_json_marker_dicts_remain_lists_and_checksum_parity(
        backend, form, celltype, nested, marker, tmp_path, monkeypatch):
    # These are user JSON dictionaries, never an internal transport wrapper.
    payload = {marker: True, "values": [17, 23]}
    if marker == "__seamless_array__":
        payload.update(dtype="float64", shape=[2])
    value = {"nested": payload, "sibling": "ordinary"} if nested else payload
    buffer = Buffer(value, celltype)
    if form == "file":
        (tmp_path / buffer.get_checksum().hex()).write_bytes(buffer.content)
    namespace = importlib.import_module("seamless_transformer.transformation_namespace")
    monkeypatch.setattr(namespace, "_get_read_folder_directories",
                        lambda: [] if form == "value" else [str(tmp_path)])
    code = 'candidate <- value[["nested"]]\n' if nested else "candidate <- value\n"
    code += (
        'result <- is.list(candidate) && '
        f'isTRUE(candidate[["{marker}"]]) && '
        'length(candidate[["values"]]) == 2L && '
        'all(candidate[["values"]] == c(17L, 23L))'
    )
    if marker == "__seamless_array__":
        code += (
            ' && identical(candidate[["dtype"]], "float64")'
            ' && length(candidate[["shape"]]) == 1L'
            ' && candidate[["shape"]] == 2L'
        )
    if nested:
        code += ' && identical(value[["sibling"]], "ordinary")'
    tf = builder(backend, code, value, celltype)
    checksum = result_checksum(tf)
    assert checksum.resolve().get_value("plain") is True
    if backend == "mem":
        cli = builder("cli", code, value, celltype)
        assert execution(cli)[0] == execution(tf)[0]
        assert result_checksum(cli) == checksum
