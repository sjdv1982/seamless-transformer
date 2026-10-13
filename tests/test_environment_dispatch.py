import os
import shlex
import subprocess
from types import SimpleNamespace

import pytest

from seamless import Buffer, Checksum
from seamless_transformer import Transformer
from seamless_transformer.run import run_transformation_dict


def _bash_transformer(tmp_path, *, docker=None, conda=None):
    marker = tmp_path / "body-executed"
    tf = Transformer("bash")
    tf.code = f"touch {shlex.quote(str(marker))}; printf 42 > RESULT"
    tf.celltypes.result = "text"
    if docker is not None:
        tf.environment.set_docker(docker)
    if conda is not None:
        tf.environment.set_conda_env(conda)
    return tf, marker


def _raw_bash(environment, marker):
    code = f"touch {shlex.quote(str(marker))}; printf 42 > RESULT"
    code_buffer = Buffer(code, "text")
    environment_buffer = Buffer(environment, "plain")
    code_buffer.tempref()
    environment_buffer.tempref()
    transformation = {
        "__language__": "bash",
        "code": ("text", None, code_buffer.get_checksum().hex()),
        "__output__": ("result", "mixed", None),
        "__meta__": {"process_lock": False},
    }
    checksum = Buffer(transformation, "plain").get_checksum()
    return run_transformation_dict(
        transformation,
        Checksum(checksum),
        {"__env__": environment_buffer.get_checksum().hex()},
        scratch=True,
        require_value=True,
    )


@pytest.mark.parametrize("execution", ["process", "spawn", "remote"])
def test_docker_and_conda_conflict_fails_before_execution_dispatch(
    execution, monkeypatch, tmp_path
):
    from seamless_transformer import transformation_cache

    tf, marker = _bash_transformer(
        tmp_path, docker={"name": "ubuntu:22.04"}, conda="seamless1"
    )
    handle = tf.build()
    assert handle.construct() is not None
    monkeypatch.setattr(transformation_cache, "get_execution", lambda: execution)
    try:
        handle.compute()
    except Exception as exc:
        message = str(exc)
    else:
        message = handle.exception or ""
    assert "docker" in message.lower() and "conda" in message.lower()
    assert not marker.exists()


@pytest.mark.parametrize(
    "spec,message",
    [
        ({"name": "ubuntu:22.04", "unknown_image_key": True}, "unknown"),
        ({"gpus": True}, "name"),
    ],
)
def test_invalid_docker_spec_builds_then_fails_at_dispatch(spec, message, tmp_path):
    tf, marker = _bash_transformer(tmp_path, docker=spec)
    handle = tf.build()
    assert handle.construct() is not None
    try:
        handle.compute()
    except Exception as exc:
        error = str(exc)
    else:
        error = handle.exception or ""
    assert message in error.lower()
    assert not marker.exists()


@pytest.mark.parametrize(
    "environment",
    [
        {"docker": {"name": "ubuntu:22.04"}, "conda_environment": "seamless1"},
        {"docker": {"name": "ubuntu:22.04", "unknown_image_key": True}},
    ],
)
def test_raw_transformation_environment_fails_before_bash_launch(environment, tmp_path):
    marker = tmp_path / "body-executed"
    with pytest.raises((ValueError, RuntimeError), match="docker|conda|unknown"):
        _raw_bash(environment, marker)
    assert not marker.exists()


def test_recorded_result_bypasses_new_environment_validation(tmp_path):
    marker = tmp_path / "body-executed"
    tf, marker = _bash_transformer(tmp_path)
    first = tf.build()
    first_checksum = first.compute()
    assert first_checksum is not None, first.exception
    assert Checksum(first_checksum).resolve().get_value("text") == "42"
    assert marker.exists()

    tf.environment.set_docker({"name": "ubuntu:22.04"})
    tf.environment.set_conda_env("seamless1")
    second = tf.build()
    assert second.construct() == first.construct()
    second_checksum = second.compute()
    assert second_checksum == first_checksum
    assert marker.exists()


def test_dask_sender_validates_only_before_a_new_submission(tmp_path):
    from seamless_dask.transformation_mixin import TransformationDaskMixin

    class FakeClient:
        def __init__(self, cached=None):
            self.cached = cached
            self.submissions = 0

        def get_fat_checksum_future(self, checksum):
            return checksum

        def _cached_transformation_for_submission(self, submission, member_id=None):
            return self.cached

        def submit_transformation(self, *args, **kwargs):
            self.submissions += 1
            raise AssertionError("Dask submission must follow dispatch validation")

    tf, marker = _bash_transformer(
        tmp_path, docker={"name": "ubuntu:22.04"}, conda="seamless1"
    )
    handle = tf.build()
    assert handle.construct() is not None
    client = FakeClient()
    with pytest.raises(RuntimeError, match="docker|conda"):
        TransformationDaskMixin._ensure_dask_futures(
            handle, client, require_value=False
        )
    assert client.submissions == 0
    assert not marker.exists()

    cached_futures = SimpleNamespace(base=None, thin=None, fat=None)
    cached_client = FakeClient(cached=cached_futures)
    assert TransformationDaskMixin._ensure_dask_futures(
        handle, cached_client, require_value=False
    ) is cached_futures
    assert cached_client.submissions == 0

    latched_handle = tf.build()
    latched_handle.construct()
    latched_handle._dask_futures = cached_futures
    latched_client = FakeClient()
    assert TransformationDaskMixin._ensure_dask_futures(
        latched_handle, latched_client, require_value=False
    ) is cached_futures
    assert latched_client.submissions == 0


def test_seamless_run_combined_environment_fails_at_dispatch(tmp_path):
    marker = tmp_path / "body-executed"
    env = os.environ.copy()
    env["SEAMLESS_CACHE"] = str(tmp_path / "cache")
    completed = subprocess.run(
        [
            "seamless-run",
            "--conda",
            "seamless1",
            "--docker",
            "ubuntu:22.04",
            "touch",
            str(marker),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode != 0
    assert "conda" in completed.stderr.lower() and "docker" in completed.stderr.lower()
    assert not marker.exists()
