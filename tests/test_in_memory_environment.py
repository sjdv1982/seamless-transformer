import json
import os
import uuid

import pytest

from seamless import Buffer, Checksum
from seamless_transformer import Transformer
from seamless_transformer.languages import check_in_memory_environment
from seamless_transformer.run import run_transformation_dict


def _run_raw_python(environment, code, *, compiled=False, probe=False):
    code_buffer = Buffer(code, "text")
    transformation = {
        "__language__": "python",
        "__code_text__": code,
        "__output__": ("result", "mixed", None),
        "__meta__": {"metavars": {"BACKEND": "mem"}},
        "code": ("text", "transformer", code_buffer.get_checksum().hex()),
    }
    checksum = Buffer(transformation, "plain").get_checksum()
    environment_buffer = Buffer(environment, "plain")
    environment_buffer.tempref()
    tf_dunder = {"__env__": environment_buffer.get_checksum().hex()}
    if compiled:
        tf_dunder["__compiled__"] = True
    if probe:
        tf_dunder["__record_probe__"] = {"mode": "capture"}
    result = run_transformation_dict(
        transformation,
        Checksum(checksum),
        tf_dunder,
        scratch=True,
        require_value=True,
    )
    return Checksum(result).resolve().get_value("mixed")


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    "environment",
    [
        {"docker": {"name": "python:3.12"}},
        {"conda_environment": "not-the-current-conda-environment"},
    ],
)
def test_raw_python_and_compiled_reject_incompatible_environment_before_body(
    compiled, environment, tmp_path
):
    marker = tmp_path / "body-executed"
    code = f"from pathlib import Path\nPath({str(marker)!r}).touch()\nresult = 42"
    with pytest.raises(RuntimeError, match="Docker image|requires conda environment"):
        _run_raw_python(environment, code, compiled=compiled)
    assert not marker.exists()


def test_conda_name_prefix_and_symlink_match_current_process(tmp_path):
    current_name = os.environ["CONDA_DEFAULT_ENV"]
    current_prefix = os.environ["CONDA_PREFIX"]
    check_in_memory_environment({"conda_environment": current_name}, "python")
    check_in_memory_environment({"conda_environment": current_prefix}, "python")

    prefix_link = tmp_path / "conda-prefix-link"
    prefix_link.symlink_to(current_prefix, target_is_directory=True)
    check_in_memory_environment({"conda_environment": str(prefix_link)}, "python")


def test_missing_required_binary_fails_before_python_body(tmp_path):
    missing = "seamless-missing-binary-" + uuid.uuid4().hex
    marker = tmp_path / "body-executed"
    code = f"from pathlib import Path\nPath({str(marker)!r}).touch()\nresult = 42"
    with pytest.raises(RuntimeError, match="Required environment binaries"):
        _run_raw_python({"which": [missing]}, code)
    assert not marker.exists()


def test_record_probe_bypasses_environment_and_which():
    missing = "seamless-missing-probe-binary-" + uuid.uuid4().hex
    value = _run_raw_python(
        {
            "docker": {"name": "python:3.12"},
            "conda_environment": "not-the-current-conda-environment",
            "which": [missing],
        },
        "result = 42",
        probe=True,
    )
    assert value == 42


def test_python_transformer_builds_and_cached_result_skips_docker_check(tmp_path):
    marker = tmp_path / "calls"
    tf = Transformer("python")
    tf.code = (
        "from pathlib import Path\n"
        f"marker = Path({str(marker)!r})\n"
        "marker.write_text(marker.read_text() + 'x' if marker.exists() else 'x')\n"
        "result = 42"
    )
    tf.celltypes.result = "plain"
    tf.local = True

    first = tf.build()
    first_checksum = first.compute()
    assert first_checksum is not None, first.exception
    assert Checksum(first_checksum).resolve().get_value("plain") == 42
    assert marker.read_text() == "x"

    tf.environment.set_docker({"name": "python:3.12"})
    second = tf.build()
    assert second.construct() == first.construct()
    second_checksum = second.compute()
    assert second_checksum == first_checksum
    assert marker.read_text() == "x"


@pytest.mark.parametrize(
    "language,code",
    [
        ("r", "result <- value + 1"),
        ("julia", "result = value + 1"),
    ],
)
def test_real_embedded_handler_matches_and_enforces_conda(language, code, tmp_path):
    current_name = os.environ["CONDA_DEFAULT_ENV"]
    tf = Transformer(language)
    tf.metavars.BACKEND = "mem"
    tf.code = code
    tf.celltypes.value = "int"
    tf.pins.value = 41
    tf.celltypes.result = "plain"
    tf.environment.set_conda_env(current_name)
    tf.local = True

    matching = tf.build()
    result = matching.compute()
    assert result is not None, matching.exception
    assert Checksum(result).resolve().get_value("plain") == 42

    marker = tmp_path / (language + "-body-ran")
    if language == "julia":
        side_effect = (
            f"open({json.dumps(str(marker))}, \"w\") do io; write(io, \"ran\"); end\n"
            "result = value + 1"
        )
    else:
        side_effect = (
            f"writeLines(\"ran\", {json.dumps(str(marker))})\n"
            "result <- value + 1"
        )
    tf.code = side_effect
    tf.environment.set_conda_env("different-conda-environment")
    mismatched = tf.build()
    try:
        mismatched.compute()
        error = mismatched.exception or ""
    except Exception as exc:
        error = str(exc)
    assert "requires conda environment" in error.lower()
    assert not marker.exists()
