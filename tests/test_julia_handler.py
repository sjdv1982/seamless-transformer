"""Real Julia CLI contract; backend parameterization is extended in phase 4."""
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


@pytest.fixture(scope="module", params=["julia"])
def backend(request):
    executable = shutil.which("julia")
    if executable is None:
        pytest.skip("Julia CLI executable is missing from PATH")
    project = os.environ.get("JULIA_PROJECT", "/tmp/seamless-julia-r-support-project")
    os.environ["JULIA_PROJECT"] = project
    probe = subprocess.run([executable, "--startup-file=no", "-e", "using NPZ, JSON"],
                           capture_output=True, text=True, timeout=180)
    if probe.returncode:
        pytest.skip("Julia CLI needs NPZ and JSON in JULIA_PROJECT: " + probe.stderr[-1000:])
    return request.param


def builder(backend, code, value=7, celltype="plain", output="plain"):
    tf = Transformer("julia")
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


def run(tf):
    checksum, payload = execution(tf)
    result = run_transformation_dict(payload, checksum)
    assert result is not None
    buffer = Checksum(result).resolve()
    # Bytes celltypes expose a Buffer through get_value; assert raw contents.
    if tf.celltypes.result == "bytes":
        return buffer.content
    return buffer.get_value(tf.celltypes.result)


@pytest.mark.parametrize("code", [
    "result = value + 2",
    "function transform(value)\n value + 2\nend",
    "transform(value) = value + 2",
    '"A documented function"\nfunction transform(value::T)::T where T\n value + 2\nend',
    "transform(value)::Int = value + 2",
    "function transform(; value=0)\n value + 2\nend",
    # A nested definition must remain block style.
    "begin\n transform(value) = value + 999\nend\nresult = value + 2",
])
def test_code_styles_and_ast_wrappers(backend, code):
    assert run(builder(backend, code)) == 9


@pytest.mark.parametrize("code", [
    "function transform(a, b=2)\n a + b\nend",
    "transform(a) = a + 1000\ntransform(a, b) = a + b",
])
def test_transform_multiple_methods_match_all_declared_pins(backend, code):
    # Defaults generate several Julia methods. Every actual parameter is
    # declared here; the handler must select the method matching both pins.
    tf = Transformer("julia")
    tf.metavars.BACKEND = backend
    tf.code = code
    tf.celltypes.a = "int"
    tf.celltypes.b = "int"
    tf.pins.a = 7
    tf.pins.b = 3
    tf.celltypes.result = "plain"
    tf.local = True
    assert run(tf) == 10


@pytest.mark.parametrize("form", ["value", "file", "gzip"])
@pytest.mark.parametrize("celltype,value,code,output,expected", [
    ("plain", {"x": [2, 5]}, 'result = value["x"][2]', "plain", 5),
    ("mixed", {"x": 4}, 'result = value["x"] + 1', "plain", 5),
    ("mixed", np.array([2, 5], dtype=np.int64), "result = value .+ 1", "binary", np.array([3, 6])),
    ("text", "hello\n", 'result = value * "!"', "plain", "hello!"),
    ("bytes", b"\x00\xff\x01", "result = reverse(value)", "bytes", b"\x01\xff\x00"),
    ("binary", np.array([[1., 2.], [3., 4.]]), "result = value .+ 1", "binary", np.array([[2., 3.], [4., 5.]])),
])
def test_pin_celltypes_value_and_read_folder(backend, form, celltype, value, code,
                                            output, expected, tmp_path, monkeypatch):
    namespace = importlib.import_module("seamless_transformer.transformation_namespace")
    buffer = Buffer(value, celltype)
    if form != "value":
        content = buffer.content
        suffix = ".gz" if form == "gzip" else ""
        path = tmp_path / (buffer.get_checksum().hex() + suffix)
        path.write_bytes(gzip.compress(content) if suffix else content)
        monkeypatch.setattr(namespace, "_get_read_folder_directories", lambda: [str(tmp_path)])
    else:
        monkeypatch.setattr(namespace, "_get_read_folder_directories", lambda: [])
    tf = builder(backend, code, value, celltype, output)
    _, payload = execution(tf)
    assert payload["__format__"]["value"]["filesystem"] == {"mode": "file", "optional": True}
    result = run(tf)
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(result, expected)
        assert result.flags.c_contiguous
        # NPZ bytes are decoded and serialized by Seamless, never adopted raw.
        assert Buffer(result, output).get_checksum() == Buffer(expected, output).get_checksum()
    else:
        assert result == expected
        assert type(result) is type(expected)


@pytest.mark.parametrize("celltype,value", [("int", 4), ("float", 2.5), ("bool", True), ("str", "quoted")])
def test_scalar_pins(backend, celltype, value):
    assert run(builder(backend, "result = value", value, celltype)) == value


@pytest.mark.parametrize("code", ['error("julia-contract-failure")', "unrelated = value + 1"])
def test_failure_and_missing_result(backend, code):
    with pytest.raises(Exception):
        run(builder(backend, code))


def test_function_definitions_have_no_pins_and_cli_runs_are_fresh(backend):
    code = '''
@assert !isdefined(@__MODULE__, :value)
const calls = Ref(0)
function transform(value)
    calls[] += 1
    [value + 2, calls[]]
end
'''
    # Direct execution bypasses Seamless result caching. CLI interpreters are
    # new per run, so the definition-stage counter restarts each time.
    for value in (5, 9):
        np.testing.assert_array_equal(run(builder(backend, code, value, output="binary")), [value + 2, 1])


@pytest.mark.parametrize("form", ["value", "file", "gzip"])
def test_mutating_array_does_not_write_through(backend, form, tmp_path, monkeypatch):
    value = np.array([3, 4], dtype=np.int64)
    buffer = Buffer(value, "binary")
    before = buffer.get_checksum()
    namespace = importlib.import_module("seamless_transformer.transformation_namespace")
    path = tmp_path / (before.hex() + (".gz" if form == "gzip" else ""))
    original = gzip.compress(buffer.content) if form == "gzip" else buffer.content
    if form != "value":
        path.write_bytes(original)
    monkeypatch.setattr(namespace, "_get_read_folder_directories",
                        lambda: [] if form == "value" else [str(tmp_path)])
    tf = builder(backend, "value[1] = 99; result = value", value, "binary", "binary")
    np.testing.assert_array_equal(run(tf), [99, 4])
    np.testing.assert_array_equal(value, [3, 4])
    assert Buffer(value, "binary").get_checksum() == before
    assert execution(tf)[1]["value"][2] == before.hex()
    if form != "value":
        assert path.read_bytes() == original


def test_mixed_container_is_refused_by_cli(backend):
    with pytest.raises(Exception):
        run(builder(backend, "result = 1", {"array": np.array([1, 2]), "plain": 3}, "mixed"))


@pytest.mark.parametrize("celltype", ["deepcell", "deepfolder", "folder"])
def test_deep_pin_celltypes_are_refused(backend, celltype):
    from seamless_transformer.languages import get_language_handler
    handler = get_language_handler("julia").handler
    with pytest.raises((ValueError, TypeError), match="deep|folder|celltype"):
        handler(code="result = 1", pins={"value": {}},
                filesystem={"value": {"celltype": celltype, "filesystem": False}},
                output_name="result", output_celltype="plain",
                metavars={"BACKEND": backend}, environment=None, process_lock=True)


@pytest.mark.parametrize("celltype", ["deepcell", "folder"])
def test_deep_result_celltypes_are_refused(backend, celltype):
    from seamless_transformer.languages import get_language_handler
    handler = get_language_handler("julia").handler
    with pytest.raises((ValueError, TypeError), match="deep|folder|celltype"):
        handler(code="result = 1", pins={}, filesystem={},
                output_name="result", output_celltype=celltype,
                metavars={"BACKEND": backend}, environment=None, process_lock=True)


def test_stdout_and_stderr_reach_transformation_logs(backend):
    from seamless_transformer import has_spawned, spawn
    # Pytest replaces the calling process streams during capture. The worker
    # installs its own routing streams, as in the existing public logs tests.
    if not has_spawned():
        spawn(1)
    tf = builder(backend, 'println("JULIA_CONTRACT_OUT_7942"); println(stderr, "JULIA_CONTRACT_ERR_7942"); result = value')
    tf.local = False
    handle = tf.build()
    assert handle.run() == 7
    assert "JULIA_CONTRACT_OUT_7942" in handle.logs
    assert "JULIA_CONTRACT_ERR_7942" in handle.logs
