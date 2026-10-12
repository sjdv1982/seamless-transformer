"""Only runtime selection uses fake executables; no container jobs are faked."""
import importlib
import socket
import tempfile
from pathlib import Path

import pytest


def select(monkeypatch, tmp_path, usable=(), explicit=None):
    for key in ('SEAMLESS_CONTAINER_RUNTIME', 'DOCKER_HOST', 'CONTAINER_HOST'):
        monkeypatch.delenv(key, raising=False)
    for runtime in ('docker', 'podman', 'apptainer'):
        executable = tmp_path / runtime
        executable.write_text('#!/bin/sh\n' +
            ('echo "{}"; exit 0\n' if runtime in usable else 'exit 1\n'))
        executable.chmod(0o755)
    monkeypatch.setenv('PATH', str(tmp_path))
    if explicit is not None:
        monkeypatch.setenv('SEAMLESS_CONTAINER_RUNTIME', explicit)
    import seamless_transformer.container as module
    return importlib.reload(module)


@pytest.mark.parametrize('runtime', ['docker', 'podman', 'apptainer'])
def test_explicit_runtime_selection(monkeypatch, tmp_path, runtime):
    module = select(monkeypatch, tmp_path, ('docker', 'podman', 'apptainer'), runtime)
    assert module.select_container_runtime('image') == runtime


@pytest.mark.parametrize('usable,expected', [
    (('docker', 'podman', 'apptainer'), 'docker'),
    (('podman', 'apptainer'), 'podman'), (('apptainer',), 'apptainer')])
def test_detection_order(monkeypatch, tmp_path, usable, expected):
    assert select(monkeypatch, tmp_path, usable).select_container_runtime('image') == expected


def test_detection_cached_per_process(monkeypatch, tmp_path):
    module = select(monkeypatch, tmp_path, ('docker', 'podman'))
    assert module.select_container_runtime('image') == 'docker'
    (tmp_path / 'docker').write_text('#!/bin/sh\nexit 1\n')
    assert module.select_container_runtime('other-image') == 'docker'


def test_none_refuses_installed_runtime(monkeypatch, tmp_path):
    module = select(monkeypatch, tmp_path, ('docker',), 'none')
    with pytest.raises(RuntimeError, match='does not allow|refus|disabled'):
        module.select_container_runtime('image')


def test_no_runtime_names_image_host_and_attempts(monkeypatch, tmp_path):
    module = select(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError) as error:
        module.select_container_runtime('unique-image')
    message = str(error.value)
    assert 'unique-image' in message and socket.gethostname() in message
    assert all(runtime in message for runtime in ('docker', 'podman', 'apptainer'))


@pytest.mark.parametrize('setting,runtime', [('DOCKER_HOST', 'docker'), ('CONTAINER_HOST', 'podman')])
def test_remote_daemon_refused(monkeypatch, tmp_path, setting, runtime):
    module = select(monkeypatch, tmp_path, (runtime,), runtime)
    monkeypatch.setenv(setting, 'tcp://remote.example:2375')
    with pytest.raises(RuntimeError, match='local|remote|unix'):
        module.select_container_runtime('image')


def test_refusal_precedes_job_materialization_and_never_runs_host(monkeypatch, tmp_path):
    from test_container_spec import raw_run
    select(monkeypatch, tmp_path, ('docker',), 'none')
    sentinel = tmp_path / 'host-executed'
    before = set(Path(tempfile.gettempdir()).glob('seamless-bash-transformer*'))
    with pytest.raises(RuntimeError, match='does not allow|refus|disabled'):
        raw_run({'docker': {'name': 'image'}}, 'touch ' + str(sentinel) + '; printf 42 > RESULT')
    assert not sentinel.exists()
    assert set(Path(tempfile.gettempdir()).glob('seamless-bash-transformer*')) == before
