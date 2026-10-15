"""Environment and container contracts for the R and Julia CLI handlers."""

import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import pytest

from seamless import Buffer
from seamless_transformer import Transformer
from seamless_transformer.languages import get_language_handler


RUNTIME_ONLY_BINARY = "seamless-phase3-runtime-only"


def _run(argv, *, timeout=900, env=None):
    result = subprocess.run(
        argv, capture_output=True, text=True, timeout=timeout, env=env
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def _set_runtime(monkeypatch, runtime, images, language="r"):
    monkeypatch.setenv("SEAMLESS_CONTAINER_RUNTIME", runtime)
    monkeypatch.setenv("SEAMLESS_CONTAINER_PULL", "never")
    monkeypatch.delenv("SEAMLESS_CONTAINER_ARGS", raising=False)
    if runtime == "apptainer":
        monkeypatch.setenv("SEAMLESS_APPTAINER_DIR", str(images["sif_dir"]))
    else:
        monkeypatch.delenv("SEAMLESS_APPTAINER_DIR", raising=False)
    return {"docker": {"name": images[language]}}


def _code(language, expression):
    if language == "r":
        return f"result <- {expression}"
    return f"result = {expression}"


def _call_handler(
    language,
    code,
    *,
    pins=None,
    filesystem=None,
    output="plain",
    environment=None,
):
    return get_language_handler(language).handler(
        code=code,
        pins=pins or {},
        filesystem=filesystem or {},
        output_name="result",
        output_celltype=output,
        metavars={"BACKEND": "cli"},
        environment=environment,
        process_lock=True,
    )


@pytest.mark.parametrize("language", ["r", "julia"])
@pytest.mark.parametrize("launcher", ["conda", "docker", "podman", "apptainer"])
def test_launcher_lists_for_conda_and_runtimes(
    language, launcher, monkeypatch, tmp_path
):
    module = importlib.import_module("seamless_transformer.languages")
    captured = []

    def capture(command, pins, values, filesystem, outputpin, environment):
        captured.append((command, environment, outputpin))
        return "launcher-probe"

    monkeypatch.setattr(module, "execute_cli", capture)
    if launcher == "conda":
        environment = {"conda_environment": sys.prefix}
    else:
        environment = {"docker": {"name": "phase3/example:local"}}
        monkeypatch.setenv("SEAMLESS_CONTAINER_PULL", "missing")
        if launcher == "apptainer":
            monkeypatch.delenv("SEAMLESS_APPTAINER_DIR", raising=False)

    assert _call_handler(language, _code(language, '"ok"'), environment=environment) == "launcher-probe"
    assert len(captured) == 1
    command, passed_environment, outputpin = captured[0]
    assert passed_environment == environment
    assert outputpin == "plain"
    expected_command = (
        "Rscript --vanilla runner.R"
        if language == "r"
        else "julia --startup-file=no --project=. runner.jl"
    )
    assert expected_command in command

    from seamless_transformer.container import build_container_command
    from seamless_transformer.execute_bash import _conda_run_command

    if launcher == "conda":
        argv = _conda_run_command(sys.prefix)
        assert "conda" in Path(argv[0]).name.lower() or argv[0].endswith("/conda")
        assert "run" in argv
    else:
        argv = build_container_command(
            launcher, environment["docker"], str(tmp_path / "job"), []
        )
        assert argv[0] == launcher
        if launcher == "apptainer":
            assert "docker://phase3/example:local" in argv
        else:
            assert "phase3/example:local" in argv


@pytest.mark.parametrize("language", ["r", "julia"])
def test_conda_interpreter_observes_requested_prefix(language):
    expression = 'Sys.getenv("CONDA_PREFIX")' if language == "r" else 'ENV["CONDA_PREFIX"]'
    result = _call_handler(
        language,
        _code(language, expression),
        environment={"conda_environment": sys.prefix},
    )
    assert os.path.realpath(result) == os.path.realpath(sys.prefix)


@pytest.mark.parametrize("language", ["r", "julia"])
@pytest.mark.parametrize("runtime", ["docker", "podman", "apptainer"])
def test_container_deepfolder_pin_carries_checksums(
    language, runtime, language_runtime_images, monkeypatch
):
    images = language_runtime_images
    environment = _set_runtime(monkeypatch, runtime, images, language)
    member = Buffer(b"read-folder member bytes", "bytes")
    checksum = member.get_checksum()
    pins = {"data": {"member": checksum}}
    filesystem = {"data": {"celltype": "deepfolder", "filesystem": False}}
    code = (
        'result <- as.character(data[["member"]])'
        if language == "r"
        else 'result = data["member"].hex'
    )
    result = _call_handler(
        language,
        code,
        pins=pins,
        filesystem=filesystem,
        output="plain",
        environment=environment,
    )
    assert result == checksum.hex()


@pytest.mark.parametrize("language", ["r", "julia"])
def test_which_binary_available_only_in_environment(
    language, language_runtime_images, monkeypatch
):
    assert shutil.which(RUNTIME_ONLY_BINARY) is None
    environment = _set_runtime(
        monkeypatch, "docker", language_runtime_images, language
    )
    environment["which"] = [RUNTIME_ONLY_BINARY]
    expression = (
        f'Sys.which("{RUNTIME_ONLY_BINARY}")'
        if language == "r"
        else f'Sys.which("{RUNTIME_ONLY_BINARY}")'
    )
    result = _call_handler(language, _code(language, expression), environment=environment)
    assert result == f"/usr/local/bin/{RUNTIME_ONLY_BINARY}"


@pytest.mark.parametrize("language", ["r", "julia"])
def test_docker_and_conda_rejected_at_dispatch(language):
    with pytest.raises(ValueError, match="docker and conda_environment"):
        _call_handler(
            language,
            _code(language, '"should not run"'),
            environment={
                "docker": {"name": "phase3/example:local"},
                "conda_environment": sys.prefix,
            },
        )


@pytest.mark.parametrize("language", ["r", "julia"])
def test_runtime_none_never_runs_host_body(language, monkeypatch, tmp_path):
    monkeypatch.setenv("SEAMLESS_CONTAINER_RUNTIME", "none")
    marker = tmp_path / "host-body-ran"
    code = (
        f'result <- writeLines("ran", "{marker}")'
        if language == "r"
        else f'result = write("{marker}", "ran")'
    )
    with pytest.raises(RuntimeError, match="SEAMLESS_CONTAINER_RUNTIME"):
        _call_handler(
            language,
            code,
            environment={"docker": {"name": "phase3/example:local"}},
        )
    assert not marker.exists()


@pytest.mark.parametrize("language", ["r", "julia"])
def test_running_container_has_transformation_label(
    language, language_runtime_images, monkeypatch
):
    from seamless_config.select import select_execution

    images = language_runtime_images
    _set_runtime(monkeypatch, "docker", images)
    select_execution("process")
    tf = Transformer(language)
    tf.code = (
        'Sys.sleep(20)\nresult <- "done"'
        if language == "r"
        else 'sleep(20)\nresult = "done"'
    )
    tf.celltypes.result = "plain"
    tf.local = True
    tf.environment.set_docker({"name": images[language]})
    job = tf.transformation()
    assert job.construct() is not None, job.exception
    checksum = job.transformation_checksum.hex()
    job.start()
    try:
        deadline = time.monotonic() + 45
        container_id = None
        while time.monotonic() < deadline:
            result = subprocess.run(
                ["docker", "ps", "-q", "--filter", "label=seamless.tf=" + checksum],
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
            _run(["docker", "inspect", container_id], timeout=30)
        )[0]
        assert details["Config"]["Labels"]["seamless.tf"] == checksum
    finally:
        job.cancel()
        if "container_id" in locals() and container_id:
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                capture_output=True,
                timeout=15,
            )
