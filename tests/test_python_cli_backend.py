import os
import sys

import pytest

from seamless import Buffer, Checksum
from seamless_transformer import Transformer, delayed, direct
from seamless_transformer.run import run_transformation_dict
from seamless_transformer.transformation_utils import extract_tf_dunder


def _python_builders():
    def identity(value):
        return value

    return [
        Transformer("python"),
        direct(identity),
        delayed(identity),
    ]


def test_python_builder_backend_metavar():
    for tf in _python_builders():
        assert tf.metavars.BACKEND == "mem"
        assert "metavars" not in tf.meta

        tf.metavars.BACKEND = "cli"
        assert tf.metavars.BACKEND == "cli"
        assert tf.meta["metavars"] == {"BACKEND": "cli"}

        tf.metavars.BACKEND = "mem"
        assert tf.metavars.BACKEND == "mem"
        assert tf.meta["metavars"] == {"BACKEND": "mem"}


@pytest.mark.parametrize("value", ["ryp", "juliacall", "Rscript", "julia", "invalid", 7, None])
@pytest.mark.parametrize("previous", ["mem", "cli"])
def test_invalid_backend_assignment_keeps_previous_value(value, previous):
    for tf in _python_builders():
        tf.metavars.BACKEND = previous
        with pytest.raises(ValueError, match="BACKEND"):
            tf.metavars.BACKEND = value
        assert tf.metavars.BACKEND == previous


def test_python_backend_metadata_and_identity(capsys):
    import uuid

    nonce = uuid.uuid4().hex
    code = Buffer(
        f"print('python-backend-run-{nonce}')\nresult = 42",
        "text",
    )
    code.tempref()
    transformation = {
        "__language__": "python",
        "code": ("text", None, code.get_checksum().hex()),
        "__output__": ("result", "mixed", None),
    }
    transformation_checksum = Buffer(transformation, "plain").get_checksum()

    results = []
    for backend in ("mem", "cli"):
        results.append(
            run_transformation_dict(
                transformation,
                transformation_checksum,
                tf_dunder={"__meta__": {"metavars": {"BACKEND": backend}}},
                scratch=True,
                require_value=True,
            )
        )

    assert results[0] == results[1] == Buffer(42, "mixed").get_checksum()
    assert capsys.readouterr().out.count(f"python-backend-run-{nonce}") == 2


def test_invalid_backend_in_transformation_dict_fails_at_run():
    code = Buffer("result = 42", "text")
    code.tempref()
    transformation = {
        "__language__": "python",
        "code": ("text", None, code.get_checksum().hex()),
        "__output__": ("result", "mixed", None),
    }
    checksum = Buffer(transformation, "plain").get_checksum()

    with pytest.raises(ValueError, match="BACKEND"):
        run_transformation_dict(
            transformation,
            checksum,
            tf_dunder={"__meta__": {"metavars": {"BACKEND": "memories"}}},
            scratch=True,
            require_value=True,
        )


def _held(value, celltype):
    buffer = Buffer(value, celltype)
    buffer.tempref()
    return buffer


def _run_python(
    code,
    pins=None,
    *,
    backend="cli",
    output="mixed",
    output_name="result",
    environment=None,
    formats=None,
    process_lock=None,
    driver=False,
):
    code_buffer = _held(code, "text")
    transformation = {
        "__language__": "python",
        "code": ("text", None, code_buffer.get_checksum().hex()),
        "__output__": (output_name, output, None),
    }
    if formats is not None:
        transformation["__format__"] = formats
    for name, (value, celltype) in (pins or {}).items():
        if isinstance(value, Buffer):
            pin_buffer = value
        else:
            pin_buffer = _held(value, celltype)
        transformation[name] = (celltype, None, pin_buffer.get_checksum().hex())

    checksum = Buffer(transformation, "plain").get_checksum()
    meta = {"metavars": {"BACKEND": backend}}
    if process_lock is not None:
        meta["process_lock"] = process_lock
    if driver:
        meta["driver"] = True
    tf_dunder = {"__meta__": meta}
    if environment:
        environment_buffer = _held(environment, "plain")
        tf_dunder["__env__"] = environment_buffer.get_checksum().hex()
    result = run_transformation_dict(
        transformation,
        checksum,
        tf_dunder=tf_dunder,
        scratch=True,
        require_value=True,
    )
    return Checksum(result), output


def _resolve_result(result, celltype):
    buffer = Checksum(result).resolve()
    if celltype in ("bytes", "binary"):
        return buffer.content
    return buffer.get_value(celltype)


def _run_handle(handle):
    checksum = handle.construct()
    assert checksum is not None, handle.exception
    payload = checksum.resolve().get_value("plain")
    payload.update(extract_tf_dunder(handle._definition_payload_template))
    payload.update(handle._tf_dunder)
    result = run_transformation_dict(
        payload,
        checksum,
        scratch=True,
        require_value=True,
    )
    return Checksum(result)


def _assert_value_equal(actual, expected):
    import numpy as np

    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_value_equal(actual[key], expected[key])
    elif isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for item, expected_item in zip(actual, expected):
            _assert_value_equal(item, expected_item)
    else:
        assert actual == expected


def test_pin_celltypes_roundtrip():
    import numpy as np

    values = [
        ("plain", {"nested": [1, True, None], "word": "ok"}),
        ("text", "plain text"),
        ("str", "string pin"),
        ("int", 123),
        ("float", 12.5),
        ("bool", True),
        ("bytes", b"\x00\x93NUMPY\xff"),
        ("binary", np.arange(12, dtype=np.int16).reshape((3, 4))),
        ("mixed", np.arange(6, dtype=np.float32).reshape((2, 3))),
        ("mixed", {"array": np.array([1.5, 2.5]), "plain": {"x": 3}}),
        ("text", ""),
        ("bytes", b""),
        ("plain", []),
        ("binary", np.empty((0, 3), dtype=np.uint16)),
    ]

    for index, (celltype, value) in enumerate(values):
        result, output = _run_python(
            "result = value",
            {"value": (_held(value, celltype), celltype)},
        )
        actual = _resolve_result(result, output)
        try:
            _assert_value_equal(actual, value)
        except AssertionError as exc:
            raise AssertionError(f"round trip {index}: {celltype}") from exc


def test_code_modes_and_namespace(monkeypatch, language_runtime_images):
    modes = [
        ("result = 4", {}, 4),
        ("def calculate(value):\n    return value + 5", {"value": (9, "int")}, 14),
        ("lambda value: value + 5", {"value": (9, "int")}, 14),
        ("value + 5", {"value": (9, "int")}, 14),
    ]
    for code, pins, expected in modes:
        result, output = _run_python(code, pins)
        assert _resolve_result(result, output) == expected

    result, output = _run_python(
        "custom = [OUTPUTPIN, PINS['value'], FILESYSTEM['value']['celltype'], "
        "__name__, __package__]",
        {"value": (9, "int")},
        output="plain",
        output_name="custom",
        formats={
            "value": {
                "celltype": "int",
                "filesystem": {"filesystem": False},
            }
        },
    )
    assert _resolve_result(result, output) == [
        "plain",
        9,
        "int",
        "transformer",
        "transformer",
    ]

    import sys

    for runtime in ("plain", "conda", "docker", "podman", "apptainer"):
        monkeypatch.delenv("SEAMLESS_CONTAINER_RUNTIME", raising=False)
        monkeypatch.delenv("SEAMLESS_CONTAINER_PULL", raising=False)
        monkeypatch.delenv("SEAMLESS_APPTAINER_DIR", raising=False)
        if runtime in ("docker", "podman", "apptainer"):
            monkeypatch.setenv("SEAMLESS_CONTAINER_RUNTIME", runtime)
            monkeypatch.setenv("SEAMLESS_CONTAINER_PULL", "never")
            if runtime == "apptainer":
                monkeypatch.setenv(
                    "SEAMLESS_APPTAINER_DIR",
                    str(language_runtime_images["sif_dir"]),
                )
            environment = {
                "docker": {"name": language_runtime_images["python_no_core"]}
            }
        elif runtime == "conda":
            environment = {"conda_environment": sys.prefix}
        else:
            environment = {}

        result, output = _run_python(
            "import sys\nresult = sys.prefix",
            output="plain",
            environment=environment,
        )
        prefix = _resolve_result(result, output)
        if runtime in ("docker", "podman", "apptainer"):
            assert prefix == "/usr/local"
        else:
            assert os.path.realpath(prefix) == os.path.realpath(sys.prefix)


def test_properly_mixed_pin_and_result_with_core():
    import numpy as np

    value = {
        "array": np.array([1.5, 2.5], dtype=np.float32),
        "plain": {"x": 3},
    }
    mixed = _held(value, "mixed")
    assert mixed.content.startswith(b"\x94SEAMLESS-MIXED")
    result, output = _run_python(
        "result = value",
        {"value": (mixed, "mixed")},
        output="mixed",
    )
    assert result == Buffer(value, "mixed").get_checksum()
    _assert_value_equal(_resolve_result(result, output), value)


def test_mixed_bytes_with_npy_magic_preserve_checksum():
    value = b"\x93NUMPY\x01\x00this is still bytes"
    result, output = _run_python(
        f"result = {value!r}",
        output="mixed",
    )
    assert result == Buffer(value, "mixed").get_checksum()
    assert _resolve_result(result, output) == value


def test_python_runner_wire_output_roundtrip():
    import numpy as np

    from seamless_transformer.python_cli_runner import _write_result
    from seamless_transformer.wire_format import read_wire_directory

    checksum = Buffer("wire checksum member", "mixed").get_checksum()
    value = {
        "json": {"a": 1, "b": [True, None]},
        "bytes": b"\x93NUMPYdata",
        "array": np.arange(6, dtype=np.uint8).reshape((2, 3)),
        "checksum": Checksum(checksum),
    }
    files = _write_result(value, "deepcell")
    decoded = read_wire_directory(files, "deepcell")
    assert decoded["json"] == value["json"]
    assert decoded["bytes"] == value["bytes"]
    assert decoded["checksum"] == Checksum(checksum)
    np.testing.assert_array_equal(decoded["array"], value["array"])
    assert decoded["array"].dtype == value["array"].dtype


def test_python_runner_rejects_unsupported_wire_value(monkeypatch):
    import sys

    from seamless_transformer import seamless_core_helper
    from seamless_transformer.python_cli_runner import _write_result

    monkeypatch.setitem(sys.modules, "seamless_core_helper", seamless_core_helper)
    with pytest.raises((TypeError, ValueError)):
        _write_result({"unsupported": object()}, "folder")


def test_without_core_simple_values_and_deepcell_values_run(
    monkeypatch, language_runtime_images
):
    import sys

    for runtime in ("plain", "docker", "podman", "apptainer"):
        monkeypatch.delenv("SEAMLESS_CONTAINER_RUNTIME", raising=False)
        monkeypatch.delenv("SEAMLESS_CONTAINER_PULL", raising=False)
        monkeypatch.delenv("SEAMLESS_APPTAINER_DIR", raising=False)
        if runtime == "plain":
            environment = {}
        else:
            monkeypatch.setenv("SEAMLESS_CONTAINER_RUNTIME", runtime)
            monkeypatch.setenv("SEAMLESS_CONTAINER_PULL", "never")
            if runtime == "apptainer":
                monkeypatch.setenv(
                    "SEAMLESS_APPTAINER_DIR",
                    str(language_runtime_images["sif_dir"]),
                )
            environment = {
                "docker": {"name": language_runtime_images["python_no_core"]}
            }

        result, output = _run_python(
            "result = value + 1",
            {"value": (42, "int")},
            output="int",
            environment=environment,
        )
        assert _resolve_result(result, output) == 43

        deep_result, deep_output = _run_python(
            "result = {'text': 'hello', 'number': 7}",
            output="deepcell",
            environment=environment,
        )
        assert _resolve_result(deep_result, deep_output) == {
            "text": Buffer("hello", "mixed").get_checksum().hex(),
            "number": Buffer(7, "mixed").get_checksum().hex(),
        }


@pytest.mark.parametrize("runtime", ["docker", "podman", "apptainer"])
@pytest.mark.parametrize("kind", ["mixed_pin", "mixed_result", "deepcell_pin", "checksum_pin"])
def test_without_core_required_helper_import_fails(
    runtime, kind, monkeypatch, language_runtime_images, capsys
):
    import numpy as np

    monkeypatch.setenv("SEAMLESS_CONTAINER_RUNTIME", runtime)
    monkeypatch.setenv("SEAMLESS_CONTAINER_PULL", "never")
    monkeypatch.delenv("SEAMLESS_CONTAINER_ARGS", raising=False)
    if runtime == "apptainer":
        monkeypatch.setenv(
            "SEAMLESS_APPTAINER_DIR", str(language_runtime_images["sif_dir"])
        )
    else:
        monkeypatch.delenv("SEAMLESS_APPTAINER_DIR", raising=False)
    environment = {"docker": {"name": language_runtime_images["python_no_core"]}}

    if kind == "mixed_pin":
        value = _held({"array": np.array([1, 2]), "plain": {"x": 3}}, "mixed")
        call = lambda: _run_python(
            "result = value", {"value": (value, "mixed")}, environment=environment
        )
    elif kind == "mixed_result":
        call = lambda: _run_python(
            "import numpy as np\nresult = {'array': np.array([1, 2]), 'plain': {'x': 3}}",
            output="mixed",
            environment=environment,
        )
    elif kind == "deepcell_pin":
        value = _held({"member": Buffer("member", "mixed").get_checksum().hex()}, "plain")
        call = lambda: _run_python(
            "result = list(value)",
            {"value": (value, "deepcell")},
            environment=environment,
        )
    else:
        checksum = Buffer("checksum member", "mixed").get_checksum()
        value = _held(checksum, "checksum")
        call = lambda: _run_python(
            "result = value.hex()",
            {"value": (value, "checksum")},
            output="str",
            environment=environment,
        )

    with pytest.raises(Exception) as excinfo:
        call()
    captured = capsys.readouterr()
    message = str(excinfo.value) + captured.err
    assert "ModuleNotFoundError" in message
    assert "seamless" in message


def test_conda_base_prefix_is_used():
    import sys
    from pathlib import Path

    base_prefix = Path(sys.prefix).parents[1]
    result, output = _run_python(
        "import sys\nresult = sys.prefix",
        output="str",
        environment={"conda_environment": str(base_prefix)},
    )
    assert os.path.realpath(_resolve_result(result, output)) == os.path.realpath(
        base_prefix
    )


def test_modules_and_globals_refused_at_run():
    module = _held({"code": "answer = 42"}, "plain")
    code = _held("result = helper.answer", "text")
    transformation = {
        "__language__": "python",
        "code": ("text", None, code.get_checksum().hex()),
        "helper": ("plain", "module", module.get_checksum().hex()),
        "__output__": ("result", "mixed", None),
    }
    checksum = Buffer(transformation, "plain").get_checksum()
    with pytest.raises(NotImplementedError, match='BACKEND="mem"'):
        run_transformation_dict(
            transformation,
            checksum,
            tf_dunder={"__meta__": {"metavars": {"BACKEND": "cli"}}},
            scratch=True,
            require_value=True,
        )

    tf = Transformer("python")
    tf.code = "result = constant"
    tf.globals.constant = 42
    tf.metavars.BACKEND = "cli"
    with pytest.raises(NotImplementedError, match='BACKEND="mem"'):
        _run_handle(tf.build())


def test_runtime_none_never_runs_host_body(monkeypatch):
    import seamless_transformer.run as run_module

    def fail(*_args, **_kwargs):
        raise AssertionError("in-memory body was called")

    monkeypatch.setattr(run_module, "_execute", fail)
    result, output = _run_python("result = 42")
    assert _resolve_result(result, output) == 42


def test_python_launcher_and_environment_which():
    result, output = _run_python(
        "result = 42",
        output="int",
        environment={"which": ["python3"]},
    )
    assert _resolve_result(result, output) == 42


def test_block_without_output_assignment_fails(capsys):
    with pytest.raises(Exception) as excinfo:
        _run_python("answer = 42")
    captured = capsys.readouterr()
    message = str(excinfo.value) + captured.err
    assert "Output variable name 'result' undefined" in message


def test_exception_has_python_traceback_in_stderr_trailer(capsys):
    with pytest.raises(Exception) as excinfo:
        _run_python("raise ValueError('python-cli-traceback-marker')")
    captured = capsys.readouterr()
    message = str(excinfo.value) + captured.err
    assert "Traceback (most recent call last)" in message
    assert "ValueError: python-cli-traceback-marker" in message


def test_stdout_is_relayed_before_completion(capsys):
    import threading
    import time

    errors = []
    outcome = {}

    def run():
        try:
            outcome["result"] = _run_python(
                "import time\nprint('PYCLI_STREAM_READY', flush=True)\n"
                "time.sleep(1.0)\nprint('PYCLI_STREAM_DONE', flush=True)\nresult = 42"
            )[0]
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    seen = ""
    deadline = time.monotonic() + 10
    relayed_early = False
    while thread.is_alive() and time.monotonic() < deadline:
        seen += capsys.readouterr().out
        if "PYCLI_STREAM_READY" in seen:
            relayed_early = True
            break
        time.sleep(0.02)
    thread.join(timeout=10)
    seen += capsys.readouterr().out
    assert not thread.is_alive()
    assert not errors
    assert relayed_early
    assert "PYCLI_STREAM_DONE" in seen
    assert "result" in outcome


def test_cli_environment_dispatch_rules(monkeypatch):
    for environment in (
        {
            "conda_environment": "/nonexistent/python-cli-env",
            "docker": {"name": "unused-image"},
        },
        {"docker": {"name": "unused-image", "invalid_key": True}},
    ):
        with pytest.raises((RuntimeError, ValueError)):
            _run_python("result = 42", environment=environment)

    nonce = os.urandom(8).hex()
    tf = Transformer("python")
    tf.code = f"result = {nonce!r}"
    tf.local = True
    tf.metavars.BACKEND = "mem"
    first = tf.compute()

    import seamless_transformer.run as run_module

    def fail_if_dispatched(*_args, **_kwargs):
        raise AssertionError("cached result reached environment dispatch")

    monkeypatch.setattr(
        run_module,
        "_validate_environment_dispatch",
        fail_if_dispatched,
    )
    tf.metavars.BACKEND = "cli"
    tf.environment.set_conda_env("/nonexistent/python-cli-env")
    tf.environment.set_docker({"name": "unused-image"})
    assert tf.compute() == first


def test_process_lock_false_allows_overlap():
    import concurrent.futures

    code = """
import time
start = time.monotonic()
time.sleep(0.45)
result = (start, time.monotonic())
"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _run_python,
                code + f"# {os.urandom(8).hex()}\n",
                process_lock=False,
            )
            for _ in range(2)
        ]
        intervals = [
            _resolve_result(future.result(timeout=30)[0], "mixed")
            for future in futures
        ]
    assert intervals[0][0] < intervals[1][1]
    assert intervals[1][0] < intervals[0][1]


@pytest.mark.parametrize(
    "environment",
    [None, pytest.param({"conda_environment": sys.prefix}, id="conda")],
)
def test_nested_submission_releases_parent_lock(environment):
    nonce = int(os.urandom(4).hex(), 16)
    code = f"""
from seamless.transformer import direct

@direct
def nested(value):
    return value + 1

result = nested({nonce!r})
"""
    checksum, celltype = _run_python(code, environment=environment)
    assert _resolve_result(checksum, celltype) == nonce + 1


def test_driver_runs_while_process_lock_is_held(tmp_path):
    import threading
    import time

    ready = tmp_path / "ready"
    release = tmp_path / "release"
    holder_result = {}
    holder_error = {}

    def run_holder():
        try:
            holder_result["value"] = _run_python(
                """
from pathlib import Path
import time
Path(ready).touch()
deadline = time.monotonic() + 20
while not Path(release).exists():
    assert time.monotonic() < deadline, 'driver did not run'
    time.sleep(.01)
result = 'holder-finished'
""",
                {"ready": (str(ready), "text"), "release": (str(release), "text")},
            )
        except BaseException as exc:
            holder_error["value"] = exc

    holder = threading.Thread(target=run_holder, daemon=True)
    holder.start()
    deadline = time.monotonic() + 15
    while not ready.exists() and holder.is_alive() and time.monotonic() < deadline:
        time.sleep(.01)
    assert ready.exists(), holder_error.get("value")

    driver_result = {}
    driver_error = {}

    def run_driver():
        try:
            driver_result["value"] = _run_python(
                "result = 'driver-ran'", driver=True
            )
        except BaseException as exc:
            driver_error["value"] = exc

    driver = threading.Thread(target=run_driver, daemon=True)
    driver.start()
    driver.join(timeout=10)
    release.touch()
    holder.join(timeout=15)
    assert not driver.is_alive(), driver_error.get("value")
    assert not holder.is_alive(), holder_error.get("value")
    assert _resolve_result(driver_result["value"][0], "mixed") == "driver-ran"
    assert _resolve_result(holder_result["value"][0], "mixed") == "holder-finished"


def test_direct_function_defined_inside_cli_body():
    code = """
from seamless.transformer import direct

@direct
def local(value):
    import os
    return (os.getpid(), value + 1)

result = local(41)
"""
    checksum, celltype = _run_python(code)
    pid, value = _resolve_result(checksum, celltype)
    assert pid > 0
    assert value == 42


def test_without_dask_nested_transformation_runs_in_job_process():
    code = """
from seamless.transformer import direct

@direct
def nested():
    import os
    return os.getpid()

import os
parent_pid = os.getpid()
child_pid = nested()
result = (parent_pid, child_pid)
"""
    checksum, celltype = _run_python(code)
    parent_pid, child_pid = _resolve_result(checksum, celltype)
    assert parent_pid == child_pid


@pytest.mark.parametrize("runtime", ["docker", "podman", "apptainer"])
def test_running_container_has_transformation_label(
    runtime, language_runtime_images, monkeypatch
):
    import json
    import subprocess
    import time

    from seamless_config.select import select_execution

    monkeypatch.setenv("SEAMLESS_CONTAINER_RUNTIME", runtime)
    monkeypatch.setenv("SEAMLESS_CONTAINER_PULL", "never")
    if runtime == "apptainer":
        monkeypatch.setenv(
            "SEAMLESS_APPTAINER_DIR", str(language_runtime_images["sif_dir"])
        )
        code = f"""import os, time
nonce = {os.urandom(8).hex()!r}
time.sleep(2)
result = os.environ.get('SEAMLESS_TF_CHECKSUM')
"""
    else:
        code = f"""import time
time.sleep(20)
result = {os.urandom(8).hex()!r}
"""
    select_execution("process")
    tf = Transformer("python")
    tf.code = code
    tf.local = True
    tf.metavars.BACKEND = "cli"
    tf.environment.set_docker(
        {"name": language_runtime_images["python_no_core"]}
    )
    job = tf.transformation()
    assert job.construct() is not None, job.exception
    checksum = job.transformation_checksum.hex()
    container_id = None
    job.start()
    try:
        if runtime == "apptainer":
            # Apptainer has no native container labels; verify the checksum
            # reaches the Python job through its launcher environment.
            assert job.run() == checksum
            return

        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            result = subprocess.run(
                [runtime, "ps", "-q", "--filter", f"label=seamless.tf={checksum}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            if result.stdout.strip():
                container_id = result.stdout.strip().splitlines()[0]
                break
            if job.exception is not None:
                raise AssertionError((job.exception, job.logs))
            time.sleep(0.05)
        assert container_id is not None, (job.status, job.exception, job.logs)
        details = json.loads(
            subprocess.run(
                [runtime, "inspect", container_id],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            ).stdout
        )[0]
        assert details["Config"]["Labels"]["seamless.tf"] == checksum
    finally:
        job.cancel()
        if container_id:
            subprocess.run(
                [runtime, "rm", "-f", container_id],
                capture_output=True,
                text=True,
                timeout=15,
            )
