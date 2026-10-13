"""Real Unix-face environment execution and command-word upload contracts."""
import os
from pathlib import Path
import shlex
import subprocess
import uuid

import pytest


# Same pinned base as test_container_docker.py: no mutable tag is executed.
UBUNTU = 'ubuntu@sha256:5ec03bb3441e8b0bf3b4f9cd4629a1ae763010dc3035bb8da3ae6cf026486401'


@pytest.fixture(scope='module')
def local_services(tmp_path_factory):
    """Own a cold cache and launcher namespace for this file's service lifetime."""
    root = tmp_path_factory.mktemp('unix-services')
    home = root / 'home'
    home.mkdir()
    # Default launcher paths agree between the controller and state helpers;
    # their explicit directory override currently has different semantics.
    settings = {'SEAMLESS_CACHE': str(root / 'cache'), 'HOME': str(home),
                'XDG_CONFIG_HOME': str(root / 'config'),
                'XDG_CACHE_HOME': str(root / 'xdg-cache'),
                'XDG_DATA_HOME': str(root / 'data'),
                'REMOTE_HTTP_LAUNCHER_DIR': None}
    yield settings
    # Only stop services registered in this fixture's private namespace.
    keys = [path.stem for path in (home / '.remote-http-launcher' / 'server').glob('*.json')]
    if keys:
        env = os.environ.copy()
        for name, value in settings.items():
            if value is None:
                env.pop(name, None)
            else:
                env[name] = value
        for command in ('rhl-stop', 'rhl-rm'):
            result = subprocess.run([command, *keys], env=env, capture_output=True,
                                    text=True, timeout=60)
            assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(autouse=True)
def isolated_service_environment(monkeypatch, local_services):
    for name, value in local_services.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


def engine(*args):
    result = subprocess.run(['docker', *args], capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


@pytest.fixture(scope='module')
def environments(tmp_path_factory):
    """Require Docker and conda, with a tool absent from the submitting PATH."""
    engine('info')
    inspected = subprocess.run(['docker', 'image', 'inspect', UBUNTU], capture_output=True)
    if inspected.returncode:
        engine('pull', UBUNTU)
    root = tmp_path_factory.mktemp('unix-environments')
    tool = 'toolonlyinenvironment_' + uuid.uuid4().hex
    body = '#!/bin/bash\nprintf "ENVIRONMENT:%s\\n" "$1"\n'
    context = root / 'image'
    context.mkdir()
    (context / tool).write_text(body)
    (context / 'Dockerfile').write_text(
        f'FROM {UBUNTU}\nCOPY {tool} /usr/local/bin/{tool}\n'
        f'RUN chmod +x /usr/local/bin/{tool}\n')
    tag = 'seamless-phase6-cli:' + uuid.uuid4().hex
    engine('build', '-t', tag, str(context))
    image = engine('image', 'inspect', '--format', '{{.Id}}', tag)
    prefix = root / 'conda-prefix'
    created = subprocess.run(['conda', 'create', '--offline', '-y', '-p', str(prefix)],
                             capture_output=True, text=True, timeout=180)
    assert created.returncode == 0, created.stdout + created.stderr
    bindir = prefix / 'bin'
    bindir.mkdir(exist_ok=True)
    (bindir / tool).write_text(body)
    (bindir / tool).chmod(0o755)
    yield {'docker': ['--docker', image], 'conda': ['--conda', str(prefix)]}, tool
    engine('image', 'rm', tag)


def cli(tmp_path, flags, words, *, env=None, success=True, secondary=False):
    run_env = os.environ.copy()
    run_env['SEAMLESS_CONTAINER_RUNTIME'] = 'docker'
    # The module fixture keeps cache and service identity stable across calls.
    if env:
        run_env.update(env)
    command = shlex.join(words)
    if secondary:
        command = 'echo ready; ' + command
    result = subprocess.run(['seamless-run', '--local', *flags, command], cwd=tmp_path,
                            env=run_env, capture_output=True, text=True, timeout=120)
    output = result.stdout + result.stderr
    if success:
        assert result.returncode == 0, output
    else:
        assert result.returncode != 0, output
    return output


@pytest.mark.parametrize('kind', ['docker', 'conda'])
def test_environment_only_tool_runs_without_host_chmod(tmp_path, environments, kind):
    flags, tool = environments
    nonce = uuid.uuid4().hex
    output = cli(tmp_path, flags[kind], [tool, nonce])
    assert 'ENVIRONMENT:' + nonce in output
    assert 'chmod:' not in output


@pytest.mark.parametrize('kind', ['docker', 'conda'])
@pytest.mark.parametrize('secondary', [False, True])
def test_host_path_tool_outside_bin_is_not_uploaded(tmp_path, environments, kind, secondary):
    flags, tool = environments
    outside = tmp_path / 'personal-tools'
    outside.mkdir()
    (outside / tool).write_text('#!/bin/bash\nprintf "HOST_POISON\\n"\nexit 73\n')
    (outside / tool).chmod(0o755)
    nonce = uuid.uuid4().hex
    output = cli(tmp_path, flags[kind], [tool, nonce],
                 # Conda preserves PATH entries preceding its active prefix.
                 # Keep the host-only tool searchable while allowing conda's
                 # activated bin directory to take its normal precedence.
                 env={'PATH': os.environ['PATH'] + os.pathsep + str(outside)},
                 secondary=secondary)
    assert 'ENVIRONMENT:' + nonce in output
    assert 'HOST_POISON' not in output


@pytest.mark.parametrize('kind', ['docker', 'conda', 'host'])
@pytest.mark.parametrize('name', ['analyze.sh', 'script with spaces.sh'])
@pytest.mark.parametrize('secondary', [False, True])
def test_local_nonexecutable_script_is_uploaded_and_chmodded(tmp_path, environments, kind, name, secondary):
    flags, _ = environments
    script = tmp_path / name
    script.write_text('#!/bin/bash\nprintf "LOCAL_SCRIPT:%s\\n" "$1"\n')
    script.chmod(0o644)
    nonce = uuid.uuid4().hex
    output = cli(tmp_path, [] if kind == 'host' else flags[kind], ['./' + name, nonce],
                 secondary=secondary)
    assert 'LOCAL_SCRIPT:' + nonce in output
    assert script.stat().st_mode & 0o111 == 0  # chmod applies to uploaded copy.


@pytest.mark.parametrize('kind', ['docker', 'conda', 'host'])
def test_missing_tool_fails_at_execution_not_chmod(tmp_path, environments, kind):
    flags, _ = environments
    missing = 'missingtool_' + uuid.uuid4().hex
    output = cli(tmp_path, [] if kind == 'host' else flags[kind],
                 [missing, uuid.uuid4().hex], success=False)
    assert missing in output
    assert 'chmod:' not in output
    assert 'command not found' in output or 'not found' in output


@pytest.mark.parametrize('kind', ['docker', 'conda', 'host'])
def test_quoted_home_script_is_uploaded_and_chmodded(tmp_path, environments, kind):
    flags, _ = environments
    name = 'home-script-' + uuid.uuid4().hex + '.sh'
    script = Path(os.environ['HOME']) / name
    script.write_text('#!/bin/bash\nprintf "HOME_SCRIPT:%s\\n" "$1"\n')
    script.chmod(0o644)
    nonce = uuid.uuid4().hex
    # shlex.join quotes the tilde, so bash itself cannot expand it.
    output = cli(tmp_path, [] if kind == 'host' else flags[kind], ['~/' + name, nonce])
    assert 'HOME_SCRIPT:' + nonce in output
    assert script.stat().st_mode & 0o111 == 0


@pytest.mark.parametrize('kind', ['docker', 'conda', 'host'])
def test_executable_preparation_preserves_same_named_variable_inputs(tmp_path, environments, kind):
    flags, _ = environments
    script = tmp_path / 'inspect-pins.sh'
    script.write_text('#!/bin/bash\nprintf "PINS:%s:%s\\n" '
                      '"$seamless_executable" "$seamless_executable_copy"\n')
    script.chmod(0o644)
    first = 'first-' + uuid.uuid4().hex
    second = 'second-' + uuid.uuid4().hex
    options = [] if kind == 'host' else flags[kind]
    output = cli(tmp_path, [*options, '--var', 'seamless_executable=' + first,
                           '--var', 'seamless_executable_copy=' + second],
                 ['./inspect-pins.sh'])
    assert 'PINS:' + first + ':' + second in output
    assert script.stat().st_mode & 0o111 == 0



def test_combined_docker_and_conda_fails_at_dispatch(tmp_path):
    marker = tmp_path / 'ran-body'
    output = cli(tmp_path, ['--docker', 'phase2-unused-image', '--conda', 'phase2-unused-env'],
                 ['sh', '-c', 'touch "$1"', 'sh', str(marker)], success=False)
    assert 'cannot be used together' in output.lower()
    assert not marker.exists()

