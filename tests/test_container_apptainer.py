"""Mandatory real Apptainer contract tests, using digest-pinned SIF inputs."""
import importlib
import os
from pathlib import Path
import shlex
import subprocess
import uuid

import pytest

from seamless_transformer.container import build_container_command

UBUNTU = 'ubuntu@sha256:5ec03bb3441e8b0bf3b4f9cd4629a1ae763010dc3035bb8da3ae6cf026486401'
ALPINE = 'alpine@sha256:d9e853e87e55526f6b2917df91a2115c36dd7c696a35be12163d44e6e2a4b6bc'


@pytest.fixture(scope='session')
def sif_images(tmp_path_factory):
    directory = tmp_path_factory.mktemp('apptainer-images')
    # This phase requires a working runtime: missing executable is a failure.
    subprocess.run(['apptainer', '--version'], check=True, capture_output=True)
    for image in (UBUNTU, ALPINE):
        target = directory / (image + '.sif')
        target.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(['apptainer', 'pull', str(target), 'docker://' + image],
                                capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, result.stdout + result.stderr
        assert target.is_file()
    return directory


@pytest.fixture
def runtime_images(sif_images, monkeypatch):
    monkeypatch.setenv('SEAMLESS_CONTAINER_RUNTIME', 'apptainer')
    monkeypatch.setenv('SEAMLESS_CONTAINER_PULL', 'never')
    monkeypatch.setenv('SEAMLESS_APPTAINER_DIR', str(sif_images))
    monkeypatch.delenv('SEAMLESS_CONTAINER_ARGS', raising=False)
    return sif_images


def execute(body, image=UBUNTU, pins=None, filesystem=None, output='text', which=()):
    module = importlib.import_module('seamless_transformer.execute_bash')
    pins = dict(pins or {})
    pins['nonce'] = uuid.uuid4().hex
    return module.execute_bash('# ' + uuid.uuid4().hex + '\n' + body,
        list(pins), '', pins, filesystem or {}, output,
        docker_={'name': image}, which_=which)


def command(image, job, reads=(), **spec):
    return build_container_command('apptainer', {'name': image, **spec},
        str(job), list(map(str, reads)), tf_checksum='a' * 64, process_id=os.getpid())


@pytest.mark.parametrize('gpus', [False, True])
def test_launcher_arguments(runtime_images, tmp_path, monkeypatch, gpus):
    monkeypatch.setenv('SEAMLESS_CONTAINER_ARGS', '--hostname "site host"')
    job = tmp_path / 'job with spaces'
    read = tmp_path / 'read buffers'
    args = command(UBUNTU, job, [read], gpus=gpus, shm_size='1g')
    assert args[:2] == ['apptainer', 'exec']
    assert '--containall' in args
    assert args[args.index('--cwd') + 1] == '/seamless-job'
    binds = [args[i + 1] for i, value in enumerate(args) if value == '--bind']
    assert f'{job}:/seamless-job' in binds
    assert f'{read}:{read}:ro' in binds
    assert ('--nv' in args) is gpus
    for flag in ('--shm-size', '--user', '--userns=keep-id', '--no-init', '--no-umask', '--no-eval', '--env-file'):
        assert flag not in args
    sif = str(runtime_images / (UBUNTU + '.sif'))
    index = args.index(sif)
    assert args[index - 2:index] == ['--hostname', 'site host']
    assert args[index + 1:] == ['bash', 'transform-inner.sh']


def test_literal_nested_sif_resolution(runtime_images, tmp_path):
    name = 'registry.example/team/tool:1.19'
    target = runtime_images / (name + '.sif')
    target.parent.mkdir(parents=True)
    target.symlink_to(runtime_images / (UBUNTU + '.sif'))
    assert str(target) in command(name, tmp_path)
    assert execute('printf literal > RESULT', name) == 'literal'


def test_missing_sif_prints_exact_pull_command(runtime_images, tmp_path):
    name = 'registry.example/team/absent:1.19'
    target = runtime_images / (name + '.sif')
    with pytest.raises(Exception) as error:
        command(name, tmp_path)
    expected = shlex.join(['apptainer', 'pull', str(target), 'docker://' + name])
    assert expected in str(error.value)


def test_legacy_directory_and_simg_are_not_read(runtime_images, tmp_path, monkeypatch):
    name = 'legacy:1'
    legacy = tmp_path / 'legacy'
    legacy.mkdir()
    (legacy / (name + '.sif')).symlink_to(runtime_images / (UBUNTU + '.sif'))
    (runtime_images / (name + '.simg')).symlink_to(runtime_images / (UBUNTU + '.sif'))
    monkeypatch.setenv('SEAMLESS_SINGULARITY_DIR', str(legacy))
    with pytest.raises(Exception) as error:
        command(name, tmp_path)
    assert str(runtime_images / (name + '.sif')) in str(error.value)


def test_missing_pull_policy_uses_runtime_cache_reference(runtime_images, tmp_path, monkeypatch):
    monkeypatch.setenv('SEAMLESS_CONTAINER_PULL', 'missing')
    name = 'registry.example/team/not-present:1'
    args = command(name, tmp_path)
    assert args[-3:] == ['docker://' + name, 'bash', 'transform-inner.sh']
    # A local SIF continues to take precedence even under missing.
    assert str(runtime_images / (UBUNTU + '.sif')) in command(UBUNTU, tmp_path)


def test_string_file_and_directory_pins_and_result_file(runtime_images):
    assert execute('printf "%s|" "$text" > RESULT; cat data.txt folder/member >> RESULT',
        pins={'text': 'hello world', 'data.txt': b'file\n', 'folder': {'member': 'directory\n'}},
        filesystem={'folder': {'mode': 'directory', 'filesystem': False}}) == 'hello world|file\ndirectory\n'


def test_result_directory(runtime_images):
    assert execute('mkdir -p RESULT/sub; printf alpha > RESULT/a; printf beta > RESULT/sub/b',
        output='folder') == {'a': b'alpha', 'sub/b': b'beta'}


def test_filesystem_symlink_resolves_and_read_folder_is_read_only(runtime_images, tmp_path):
    from seamless_remote import buffer_remote
    folder = tmp_path / 'buffers'
    folder.mkdir()
    data = folder / 'input.txt'
    data.write_text('filesystem input\n')
    name = 'phase4-' + uuid.uuid4().hex
    buffer_remote.define_extern_client(name, 'bufferfolder', directory=str(folder))
    client = buffer_remote._extern_clients[name]
    buffer_remote._read_folders_clients.append(client)
    try:
        result = execute('cat "$data" > RESULT; test -L data; '
            'if printf bad > "$(dirname "$data")/forbidden"; then exit 9; fi',
            pins={'data': str(data)}, filesystem={'data': {'mode': 'file', 'filesystem': True}})
        assert result == 'filesystem input\n'
        assert not (folder / 'forbidden').exists()
    finally:
        buffer_remote._read_folders_clients.remove(client)
        del buffer_remote._extern_clients[name]


def test_home_cwd_and_host_ownership(runtime_images):
    result = execute('test "$HOME" = /seamless-job; test "$PWD" = /seamless-job; '
        'printf writable > "$HOME/home-file"; '
        'printf "%s:%s|" "$(id -u)" "$(id -g)" > RESULT; '
        'cat "$HOME/home-file" >> RESULT; stat -c "|%u:%g" home-file >> RESULT')
    owner = f'{os.getuid()}:{os.getgid()}'
    assert result == f'{owner}|writable|{owner}\n'


def test_host_environment_and_notify_do_not_reach_script(runtime_images, monkeypatch):
    monkeypatch.setenv('SEAMLESS_PROCESS_LOCK_NOTIFY', '/host/notify')
    monkeypatch.setenv('SEAMLESS_PHASE4_HOST_SECRET', 'host-only')
    assert execute('test -z "${SEAMLESS_PROCESS_LOCK_NOTIFY+x}"; '
        'test -z "${SEAMLESS_PHASE4_HOST_SECRET+x}"; printf isolated > RESULT') == 'isolated'


def test_image_without_bash_is_environment_entry_failure(runtime_images):
    with pytest.raises(Exception) as error:
        execute('printf should-not-run > RESULT', ALPINE)
    message = str(error.value)
    assert 'not entered' in message.lower()
    assert 'apptainer' in message and ALPINE in message
    assert 'Return code: 255' in message
    assert 'bash' in message.lower()


def test_required_binaries_checked_inside_image(runtime_images):
    assert execute('printf checked > RESULT', which=['bash']) == 'checked'
    with pytest.raises(Exception) as error:
        execute('printf should-not-run > RESULT', which=['seamless-absent-' + uuid.uuid4().hex])
    assert 'not entered' not in str(error.value).lower()
    assert 'Required environment binaries are unavailable' in str(error.value)


@pytest.mark.parametrize('pull', ['never', 'missing'])
def test_unset_sif_directory(runtime_images, tmp_path, monkeypatch, pull):
    monkeypatch.delenv('SEAMLESS_APPTAINER_DIR')
    monkeypatch.setenv('SEAMLESS_CONTAINER_PULL', pull)
    if pull == 'missing':
        assert command(UBUNTU, tmp_path)[-3:] == ['docker://' + UBUNTU, 'bash', 'transform-inner.sh']
    else:
        with pytest.raises(Exception) as error:
            command(UBUNTU, tmp_path)
        assert 'SEAMLESS_APPTAINER_DIR' in str(error.value)


def test_materialized_launcher_runs_after_move_to_path_with_spaces(runtime_images, tmp_path):
    from seamless_transformer.execute_bash import write_bash_job
    original = tmp_path / 'original'
    original.mkdir()
    write_bash_job('printf "%s|%s" "$HOME" "$PWD" > RESULT; '
        'printf writable > "$HOME/home-file"', [], '', {}, {},
        directory=str(original), docker_={'name': UBUNTU}, container_runtime_='apptainer')
    assert '__SEAMLESS_JOB_DIR__' not in (original / 'transform.sh').read_text()
    moved = tmp_path / 'moved job with spaces'
    original.rename(moved)
    completed = subprocess.run(['bash', 'transform.sh'], cwd=moved,
        capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (moved / 'RESULT').read_text() == '/seamless-job|/seamless-job'
    assert (moved / 'home-file').read_text() == 'writable'
    assert (moved / 'RESULT').stat().st_uid == os.getuid()
    assert (moved / 'RESULT').stat().st_gid == os.getgid()
