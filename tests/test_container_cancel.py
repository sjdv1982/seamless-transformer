"""Phase 5: cancellation kills real bash jobs on local and jobserver backends.

No runtime is mocked or skipped. The small registry-only test below is separate
from the runtime tests so that callback scoping has a deterministic assertion.
"""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import uuid

import pytest

UBUNTU = 'ubuntu@sha256:5ec03bb3441e8b0bf3b4f9cd4629a1ae763010dc3035bb8da3ae6cf026486401'

_SCRIPT = r'''
import json, os, pathlib, signal, subprocess, sys, time, uuid
import seamless
from seamless_transformer import Transformer
from seamless_config.select import select_execution
from seamless_transformer.transformation_cache import get_transformation_cache
mode, runtime, action, image = sys.argv[1:]
if mode != 'jobserver':
    select_execution(mode)
    if mode == 'spawn':
        from seamless_transformer import spawn
        spawn(1)
else:
    import seamless.config
    seamless.config.init()
token = 'seamless-cancel-' + uuid.uuid4().hex
tf = Transformer('bash')
tf.code = '# ' + uuid.uuid4().hex + '\nexec -a ' + token + ' sleep 120'
tf.celltypes.result = 'text'
if runtime in ('docker', 'podman', 'apptainer'):
    tf.environment.set_docker({'name': image})
elif runtime == 'conda':
    tf.environment.set_conda_env(sys.prefix)
job = tf.transformation()
job.construct()
checksum = job.transformation_checksum.hex()

def engine(*args):
    r = subprocess.run([runtime, *args], capture_output=True, text=True, timeout=10)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout.strip()

def containers():
    # Include stopped containers: teardown must remove them, not only stop them.
    return engine('ps', '-aq', '--filter', 'label=seamless.tf=' + checksum)

def token_pids():
    result = []
    for entry in pathlib.Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / 'cmdline').read_bytes().split(b'\0')
            if argv and argv[0] == token.encode():
                result.append(int(entry.name))
        except (OSError, ProcessLookupError):
            pass
    return result

def group_live(pgid):
    # A zombie has exited; its eventual reaping is the host init's responsibility.
    rows = subprocess.check_output(['ps', '-eo', 'pgid=,stat='], text=True)
    return any(int(fields[0]) == pgid and not fields[1].startswith('Z')
               for row in rows.splitlines() if len(fields := row.split()) == 2)

def wait(predicate, timeout, message):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        if job.exception is not None:
            raise AssertionError((message, job.exception, job.logs))
        time.sleep(.05)
    raise AssertionError((message, job.status, job.exception, job.logs))

pgid = None
container = None
try:
    job.start()
    if runtime in ('docker', 'podman'):
        container = wait(containers, 45, 'container never started').splitlines()[0]
        details = json.loads(engine('inspect', container))[0]
        worker_pid = int(details['Config']['Labels']['seamless.pid'])
        # Wait for the inner body too; a container existing is not execution proof.
        wait(lambda: token in engine('top', container, '-eo', 'pid,args'), 15, 'sleep never started')
    else:
        pid = wait(token_pids, 45, 'sleep never started')[0]
        pgid = os.getpgid(pid)
    started = time.monotonic()
    if action == 'sigterm':
        assert mode == 'spawn' and runtime in ('docker', 'podman')
        assert worker_pid != os.getpid(), 'spawn must execute in a worker child'
        os.kill(worker_pid, signal.SIGTERM)
    elif action == 'sigkill':
        assert mode == 'spawn' and runtime in ('docker', 'podman')
        assert worker_pid != os.getpid(), 'spawn must execute in a worker child'
        os.kill(worker_pid, signal.SIGKILL)
        get_transformation_cache().cancel_by_checksum(checksum)  # PID-label backstop.
    else:
        # Handle.cancel() softcancels participation; the exit criterion is hard.
        get_transformation_cache().cancel_by_checksum(checksum)
    end = started + 5
    while time.monotonic() < end:
        gone = not containers() if container else not group_live(pgid)
        if gone:
            break
        time.sleep(.05)
    else:
        raise AssertionError(('cleanup exceeded 5 seconds', mode, runtime, action, checksum, pgid))
    print('CANCEL_OK', mode, runtime, action, flush=True)
finally:
    # Even a failed assertion must not leave a 120-second job behind.
    if container:
        subprocess.run([runtime, 'rm', '-f', *(['-t', '0'] if runtime == 'podman' else []), container],
                       capture_output=True, timeout=15)
    if pgid and group_live(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    job.cancel()
    seamless.close()
'''


@pytest.fixture(scope='session')
def apptainer_images(tmp_path_factory):
    directory = tmp_path_factory.mktemp('cancel-apptainer')
    target = directory / (UBUNTU + '.sif')
    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(['apptainer', 'pull', str(target), 'docker://' + UBUNTU],
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    return directory


def _runtime_env(runtime, request):
    env = os.environ.copy()
    env['SEAMLESS_CONTAINER_PULL'] = 'never'
    env.pop('SEAMLESS_CONTAINER_ARGS', None)
    if runtime in ('docker', 'podman', 'apptainer'):
        env['SEAMLESS_CONTAINER_RUNTIME'] = runtime
    if runtime == 'apptainer':
        env['SEAMLESS_APPTAINER_DIR'] = str(request.getfixturevalue('apptainer_images'))
    if runtime in ('docker', 'podman'):
        result = subprocess.run([runtime, 'image', 'inspect', UBUNTU], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    return env


def _run_case(mode, runtime, action, request, tmp_path):
    env = _runtime_env(runtime, request)
    command = [sys.executable, '-c', _SCRIPT, mode, runtime, action, UBUNTU]
    if mode != 'jobserver':
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'CANCEL_OK' in result.stdout
        return
    helper = Path(__file__).parent / 'cancellation' / '_harness.py'
    spec = importlib.util.spec_from_file_location('container_cancel_harness', helper)
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    home, work = tmp_path / 'home', tmp_path / 'work'
    home.mkdir()
    work.mkdir()
    name = 'cancel-' + uuid.uuid4().hex
    harness._write_cluster_config(home, cluster_name=name, conda_env=harness.current_conda_env(),
                                  remote_kind='jobserver', port_base=62000)
    harness._write_test_config(work, project=name, cluster_name=name, remote_kind='jobserver')
    # Podman rootless image storage must retain the operator's location even
    # though the cluster harness isolates Seamless configuration under HOME.
    env.setdefault('XDG_DATA_HOME', str(Path(env['HOME']) / '.local/share'))
    env.setdefault('XDG_CONFIG_HOME', str(Path(env['HOME']) / '.config'))
    env.update(HOME=str(home), RHL_FALLBACK_CONDA_SOURCE=str(Path(sys.prefix).parents[1] / 'etc/profile.d/conda.sh'))
    cluster = harness.Cluster(workdir=work, env=env, project=name, cluster_name=name)
    try:
        warmup = cluster.run_command(['seamless-run', '-q', '-c', 'echo warmup'], timeout=120)
        assert warmup.returncode == 0, warmup.stdout + warmup.stderr
        result = cluster.run_command(command, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'CANCEL_OK' in result.stdout
    finally:
        for command in ('seamless-service-stop', 'seamless-service-rm'):
            try:
                cluster.run_command([command, '--cluster', name, '--project', name], timeout=30)
            except Exception:
                continue


@pytest.mark.parametrize('mode', ['process', 'spawn', 'jobserver'])
@pytest.mark.parametrize('runtime', ['plain', 'conda', 'docker', 'podman', 'apptainer'])
def test_hard_cancel_real_sleep(mode, runtime, request, tmp_path):
    _run_case(mode, runtime, 'cancel', request, tmp_path)


@pytest.mark.parametrize('runtime', ['docker', 'podman'])
@pytest.mark.parametrize('action', ['sigterm', 'sigkill'])
def test_worker_death_removes_real_container(runtime, action, request, tmp_path):
    _run_case('spawn', runtime, action, request, tmp_path)


def test_registry_cleanup_scoped_and_unregisterable():
    from seamless_transformer.bash_cancellation import register_cleanup, run_cleanups
    calls = []
    a, b = uuid.uuid4().hex, uuid.uuid4().hex
    unregister_a = register_cleanup(a, lambda: calls.append('a'))
    unregister_b = register_cleanup(b, lambda: calls.append('b'))
    try:
        run_cleanups(a)
        assert calls == ['a']
        unregister_a()
        unregister_b()
        run_cleanups(b)
        assert calls == ['a']
    finally:
        unregister_a()
        unregister_b()


@pytest.mark.parametrize('fail', [False, True])
def test_executor_finally_unregisters_and_kills_background_job(fail, monkeypatch, tmp_path):
    import importlib
    import time
    module = importlib.import_module('seamless_transformer.execute_bash')
    registry = importlib.import_module('seamless_transformer.bash_cancellation')
    calls = []
    callbacks = []
    original = registry.register_cleanup

    def register(checksum, callback):
        calls.append('register')
        callbacks.append(callback)
        remove = original(checksum, callback)

        def unregister():
            calls.append('unregister')
            remove()
        return unregister

    # Only observe the registry boundary; bash execution and process killing are real.
    monkeypatch.setattr(module, 'register_cleanup', register)
    checksum = uuid.uuid4().hex + uuid.uuid4().hex
    marker = tmp_path / 'background.pid'
    body = 'sleep 120 >/dev/null 2>&1 & echo $! > ' + str(marker)
    body += '; exit 9' if fail else '; printf done > RESULT'
    try:
        if fail:
            with pytest.raises(Exception):
                module.execute_bash(body, [], '', {}, {}, 'text', tf_checksum_=checksum)
        else:
            assert module.execute_bash(body, [], '', {}, {}, 'text', tf_checksum_=checksum) == 'done'
        assert calls == ['register', 'unregister']
        assert marker.exists()
        pid = int(marker.read_text())
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                state = Path('/proc', str(pid), 'stat').read_text().split(') ', 1)[1].split()[0]
            except FileNotFoundError:
                break
            if state == 'Z':
                break
            time.sleep(.05)
        else:
            pytest.fail('background bash child survived executor finally cleanup')
    finally:
        for callback in callbacks:
            callback()
        registry.run_cleanups(checksum)
