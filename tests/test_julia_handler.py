"""Real Julia CLI contract; backend parameterization is extended in phase 4."""
import gzip
import json
import importlib
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import numpy as np
import pytest

from seamless import Buffer, Checksum
from seamless_transformer import Transformer
from seamless_transformer.run import run_transformation_dict
from seamless_transformer.transformation_utils import extract_tf_dunder


@pytest.fixture(params=["cli", "mem"])
def backend(request):
    # JuliaCall uses its own conda-local project. The CLI stages the tracked
    # Julia project files into each job directory and passes --project=.
    if request.param == "mem":
        if importlib.util.find_spec("juliacall") is None:
            pytest.skip("juliacall is missing")
        return request.param
    project = Path(__file__).resolve().parents[2] / "seamless" / "julia"
    executable = shutil.which("julia")
    if executable is None:
        pytest.skip("Julia CLI executable is missing from PATH")
    probe = subprocess.run([executable, "--startup-file=no", f"--project={project}", "-e", "using NPZ, JSON"],
                           capture_output=True, text=True, timeout=180)
    if probe.returncode:
        pytest.skip("Julia CLI needs NPZ and JSON: " + probe.stderr[-1000:])
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
    for index, value in enumerate((5, 9), 1):
        expected_calls = index if backend == "mem" else 1
        np.testing.assert_array_equal(run(builder(backend, code, value, output="binary")),
                                      [value + 2, expected_calls])


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
    tf = builder(backend, 'result = sum(value["array"]) + value["plain"]',
                 {"array": np.array([1, 2]), "plain": 3}, "mixed")
    if backend == "cli":
        with pytest.raises(Exception):
            run(tf)
    else:
        assert run(tf) == 6


@pytest.mark.parametrize("celltype", ["deepcell", "deepfolder", "folder"])
def test_deep_pin_celltypes_are_delivered_in_their_native_form(backend, celltype):
    from seamless_transformer.languages import get_language_handler
    handler = get_language_handler("julia").handler
    checksum = "a" * 64
    if celltype in ("deepcell", "deepfolder"):
        value = {"k": Checksum(checksum)}
        code = f'result = value["k"] isa Checksum && value["k"].hex == "{checksum}"'
        expected = True
    else:
        value = {"k": b"folder bytes"}
        code = 'result = String(value["k"])'
        expected = "folder bytes"
    assert handler(
        code=code,
        pins={"value": value},
        filesystem={"value": {"celltype": celltype, "filesystem": False}},
        output_name="result",
        output_celltype="plain",
        metavars={"BACKEND": backend},
        environment=None,
        process_lock=True,
    ) == expected


@pytest.mark.parametrize("celltype", ["deepcell", "folder"])
def test_deep_result_celltypes_are_refused(backend, celltype):
    from seamless_transformer.languages import get_language_handler
    handler = get_language_handler("julia").handler
    checksum = "b" * 64
    result = handler(
        code=f'result = Dict("k" => Checksum("{checksum}"))',
        pins={},
        filesystem={},
        output_name="result",
        output_celltype=celltype,
        metavars={"BACKEND": backend},
        environment=None,
        process_lock=True,
    )
    assert result == {"k": Checksum(checksum)}


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


def result_checksum(tf):
    """Execute the body even if this transformation already has a cached result."""
    uses_cli = tf.metavars.BACKEND == "cli"
    previous_project = os.environ.get("JULIA_PROJECT")
    if uses_cli:
        project = Path(__file__).resolve().parents[2] / "seamless" / "julia"
        os.environ["JULIA_PROJECT"] = str(project)
    try:
        checksum, payload = execution(tf)
        result = run_transformation_dict(payload, checksum)
        assert result is not None
        return Checksum(result)
    finally:
        if uses_cli:
            if previous_project is None:
                os.environ.pop("JULIA_PROJECT", None)
            else:
                os.environ["JULIA_PROJECT"] = previous_project


@pytest.mark.parametrize("form", ["value", "file", "gzip"])
@pytest.mark.parametrize("style", ["block", "function"])
@pytest.mark.parametrize("celltype,value,expression,output", [
    ("text", "hello\n", 'value * "!"', "plain"),
    ("plain", {"x": [2, 5]}, 'value["x"][2]', "plain"),
    ("mixed", {"x": 4}, 'value["x"] + 1', "plain"),
    ("mixed", np.array([2, 5], dtype=np.int64), "value .+ 1", "mixed"),
    ("bytes", b"\x00\xff", "reverse(value)", "bytes"),
    ("binary", np.array([[1., 2.], [3., 4.]]), "value .+ 1", "binary"),
    ("int", 7, "value + 1", "plain"),
    ("float", 2.5, "value + 1", "plain"),
    ("bool", True, "value", "plain"),
    ("str", "quoted", "value", "plain"),
])
def test_successful_result_checksum_parity(backend, form, style, celltype, value,
                                          expression, output, tmp_path, monkeypatch):
    if backend != "mem":
        pytest.skip("paired computation is run once per case")
    namespace = importlib.import_module("seamless_transformer.transformation_namespace")
    buffer = Buffer(value, celltype)
    if form != "value":
        content = buffer.content
        path = tmp_path / (buffer.get_checksum().hex() + (".gz" if form == "gzip" else ""))
        path.write_bytes(gzip.compress(content) if form == "gzip" else content)
    monkeypatch.setattr(namespace, "_get_read_folder_directories",
                        lambda: [] if form == "value" else [str(tmp_path)])
    code = "result = " + expression if style == "block" else "transform(value) = " + expression
    cli = builder("cli", code, value, celltype, output)
    embedded = builder("mem", code, value, celltype, output)
    assert execution(cli)[0] == execution(embedded)[0]
    assert result_checksum(cli) == result_checksum(embedded)


@pytest.mark.parametrize("dtype,shape", [("float32", (2, 3)), ("float64", (2, 3, 4)),
                                       ("int32", (3,)), ("int64", (2, 2)),
                                       ("float64", (1,))])
def test_array_dtype_checksum_parity(backend, dtype, shape):
    if backend != "mem":
        pytest.skip("paired computation is run once per case")
    value = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    for expression in ("value", "value .+ 1"):
        code = "result = " + expression
        assert result_checksum(builder("cli", code, value, "binary", "binary")) == result_checksum(
            builder("mem", code, value, "binary", "binary"))


@pytest.mark.parametrize("environment", [{"docker": {"name": "unused"}},
                                        {"conda_environment": "definitely-not-seamless1"}])
def test_embedded_environment_refusal(backend, environment):
    if backend != "mem":
        pytest.skip("embedded environment contract")
    from seamless_transformer.languages import get_language_handler
    with pytest.raises(Exception, match="docker|Docker|conda|environment"):
        get_language_handler("julia").handler(
            code="result = value", pins={"value": 7},
            filesystem={"value": {"celltype": "plain", "filesystem": False}},
            output_name="result", output_celltype="plain",
            metavars={"BACKEND": backend}, environment=environment, process_lock=True)


def test_embedded_current_environment_and_lock_requirement(backend):
    if backend != "mem":
        pytest.skip("embedded lock contract")
    from seamless_transformer.languages import get_language_handler
    handler = get_language_handler("julia").handler
    arguments = dict(code="result = value", pins={"value": 7},
                     filesystem={"value": {"celltype": "plain", "filesystem": False}},
                     output_name="result", output_celltype="plain",
                     metavars={"BACKEND": backend},
                     environment={"conda_environment": os.environ["CONDA_PREFIX"]})
    assert handler(**arguments, process_lock=True) == 7
    with pytest.raises(Exception, match="julia|process.lock|process_lock"):
        handler(**arguments, process_lock=False)


def test_block_isolation_is_fresh(backend):
    code = '@assert !isdefined(@__MODULE__, :previous); previous = value; result = value'
    assert run(builder(backend, code, 5)) == 5
    assert run(builder(backend, code, 9)) == 9


@pytest.mark.parametrize("style", ["block", "function"])
def test_mutation_success_checksum_parity(backend, style):
    if backend != "mem":
        pytest.skip("paired computation once per case")
    value = np.array([3, 4], dtype=np.int64)
    code = "value[1] = 99; result = value" if style == "block" else "transform(value) = (value[1] = 99; value)"
    assert result_checksum(builder("cli", code, value, "binary", "binary")) == result_checksum(
        builder("mem", code, value, "binary", "binary"))
    np.testing.assert_array_equal(value, [3, 4])


def test_stdout_stderr_before_error_reach_transformation_logs(backend):
    from seamless_transformer import has_spawned, spawn
    if not has_spawned():
        spawn(1)
    tf = builder(backend, 'println("JULIA_FAILURE_OUT_8317"); println(stderr, "JULIA_FAILURE_ERR_8317"); error("julia-logged-failure-8317")')
    tf.local = False
    handle = tf.build()
    with pytest.raises(Exception, match="julia-logged-failure-8317"):
        handle.run()
    assert handle.exception is not None
    assert "JULIA_FAILURE_OUT_8317" in handle.logs
    assert "JULIA_FAILURE_ERR_8317" in handle.logs


@pytest.mark.parametrize("layout", ["vector", "C", "F"])
@pytest.mark.parametrize("form", ["value", "file"])
@pytest.mark.parametrize("dtype", ["int32", "uint64", "float64", "bool"])
def test_typed_array_layout_mutation_parity_and_input_immunity(backend, layout, form,
                                                            dtype, tmp_path, monkeypatch):
    if backend != "mem":
        pytest.skip("paired successful executions once per case")
    original = np.arange(1, 7).astype(dtype)
    value = original if layout == "vector" else np.array(original.reshape(2, 3), order=layout)
    buffer = Buffer(value, "binary")
    before = buffer.get_checksum()
    path = tmp_path / before.hex()
    if form == "file":
        path.write_bytes(buffer.content)
    namespace = importlib.import_module("seamless_transformer.transformation_namespace")
    monkeypatch.setattr(namespace, "_get_read_folder_directories",
                        lambda: [] if form == "value" else [str(tmp_path)])
    code = "value[1] = zero(eltype(value)); result = value"
    expected = value.copy()
    expected.flat[0] = 0
    expected = expected.astype({"int32": "int64", "uint64": "uint64",
                                "float64": "float64", "bool": "bool"}[dtype])
    checksums = []
    for selected in ("cli", "mem"):
        checksum = result_checksum(builder(selected, code, value, "binary", "binary"))
        result = checksum.resolve().get_value("binary")
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == expected.dtype
        assert result.flags.c_contiguous
        checksums.append(checksum)
    assert checksums[0] == checksums[1]
    np.testing.assert_array_equal(value, original if layout == "vector" else original.reshape(2, 3))
    assert Buffer(value, "binary").get_checksum() == before
    if form == "file":
        assert path.read_bytes() == buffer.content


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
    tf = builder(backend, "result = value", value, "bytes", "bytes")
    checksum = result_checksum(tf)
    assert checksum.resolve().content == value
    assert checksum == before
    if backend == "mem":
        cli = builder("cli", "result = value", value, "bytes", "bytes")
        assert result_checksum(cli) == checksum
    assert Buffer(value, "bytes").get_checksum() == before
    if form != "value":
        assert path.read_bytes() == stored


def test_julia_cli_readiness_canary():
    from seamless_transformer.languages import get_language_handler

    project = Path(__file__).resolve().parents[2] / "seamless" / "julia"
    executable = shutil.which("julia")
    assert executable is not None, "Julia CLI executable is required for the Phase 2 canary"
    probe = subprocess.run(
        [executable, "--startup-file=no", f"--project={project}", "-e", "using NPZ, JSON"],
        capture_output=True, text=True, timeout=180,
    )
    assert probe.returncode == 0, "Julia CLI requires NPZ and JSON: " + probe.stderr[-1000:]
    code = f"# Phase 2 readiness canary {uuid.uuid4().hex}\nresult = value + 1"
    result = get_language_handler("julia").handler(
        code=code,
        pins={"value": 41},
        filesystem={"value": {"celltype": "plain", "filesystem": False}},
        output_name="result",
        output_celltype="plain",
        metavars={"BACKEND": "cli"},
        environment=None,
        process_lock=True,
    )
    assert result == 42


def test_checksum_class_contract(backend):
    from seamless_transformer.languages import get_language_handler

    checksum = "ab" * 32
    handler = get_language_handler("julia").handler
    result = handler(
        code=(
            f'checksum = Checksum("{checksum}")\n'
            f'result = Dict("text" => string(checksum), '
            '"is_checksum" => checksum isa Checksum, '
            f'"equal" => checksum == Checksum("{checksum}"), '
            f'"plain_is_checksum" => ("{checksum}" isa Checksum))'
        ),
        pins={},
        filesystem={},
        output_name="result",
        output_celltype="plain",
        metavars={"BACKEND": backend},
        environment=None,
        process_lock=True,
    )
    assert result == {
        "text": checksum,
        "is_checksum": True,
        "equal": True,
        "plain_is_checksum": False,
    }
    with pytest.raises(Exception):
        handler(
            code='result = Checksum("invalid")',
            pins={},
            filesystem={},
            output_name="result",
            output_celltype="plain",
            metavars={"BACKEND": backend},
            environment=None,
            process_lock=True,
        )
