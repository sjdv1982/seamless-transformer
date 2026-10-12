"""Mandatory real docker and podman contract tests, with digest-pinned inputs."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import pytest

from seamless_transformer.container import build_container_command

UBUNTU = 'ubuntu@sha256:5ec03bb3441e8b0bf3b4f9cd4629a1ae763010dc3035bb8da3ae6cf026486401'
ALPINE = 'alpine@sha256:d9e853e87e55526f6b2917df91a2115c36dd7c696a35be12163d44e6e2a4b6bc'


def engine(runtime, *args):
    result = subprocess.run([runtime, *args], capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope='session', params=['docker', 'podman'])
def images(request, tmp_path_factory):
    runtime = request.param
    engine(runtime, 'info')  # Missing/broken runtime is a failure, never a skip.
    for image in (UBUNTU, ALPINE):
        inspected = subprocess.run([runtime, 'image', 'inspect', image], capture_output=True)
        if inspected.returncode:
            engine(runtime, 'pull', image)
    context = tmp_path_factory.mktemp('entrypoint-' + runtime)
    (context / 'Dockerfile').write_text('FROM ' + UBUNTU + '\nENTRYPOINT ["/bin/false"]\n')
    tag = 'seamless-phase3-entrypoint:' + uuid.uuid4().hex
    engine(runtime, 'build', '-t', tag, str(context))
    built = json.loads(engine(runtime, 'image', 'inspect', tag))[0]['Id']
    yield runtime, {'bash': UBUNTU, 'no_bash': ALPINE, 'entrypoint': built}
    engine(runtime, 'image', 'rm', tag)


@pytest.fixture
def runtime_images(images, monkeypatch):
    runtime, available = images
    monkeypatch.setenv('SEAMLESS_CONTAINER_RUNTIME', runtime)
    monkeypatch.setenv('SEAMLESS_CONTAINER_PULL', 'never')
    monkeypatch.delenv('SEAMLESS_CONTAINER_ARGS', raising=False)
    return runtime, available


def execute(body, image, pins=None, filesystem=None, output='text', which=()):
    module = importlib.import_module('seamless_transformer.execute_bash')
    pins = dict(pins or {})
    pins['nonce'] = uuid.uuid4().hex
    return module.execute_bash('# ' + uuid.uuid4().hex + '\n' + body,
        list(pins), '', pins, filesystem or {}, output,
        docker_={'name': image}, which_=which)


@pytest.mark.parametrize('pull', ['never', 'missing'])
def test_launcher_arguments(runtime_images, tmp_path, monkeypatch, pull):
    runtime, available = runtime_images
    monkeypatch.setenv('SEAMLESS_CONTAINER_PULL', pull)
    monkeypatch.setenv('SEAMLESS_CONTAINER_ARGS', '--network none --security-opt "label=disable"')
    job = tmp_path / 'job with spaces'
    read = tmp_path / 'buffers'
    args = build_container_command(runtime,
        {'name': available['bash'], 'gpus': True, 'shm_size': '1g'},
        str(job), [str(read)], tf_checksum='a' * 64, process_id=os.getpid())
    assert args[:2] == [runtime, 'run']
    assert '--rm' in args and '--init' in args
    assert args[args.index('--entrypoint') + 1] == ''
    assert '--pull=' + pull in args
    assert args[args.index('-w') + 1] == '/seamless-job'
    mounts = [args[i + 1] for i, value in enumerate(args) if value == '--mount']
    assert f'type=bind,src={job},dst=/seamless-job' in mounts
    assert f'type=bind,src={read},dst={read},readonly' in mounts
    assert args[args.index('--shm-size') + 1] == '1g'
    if runtime == 'docker':
        assert args[args.index('--gpus') + 1] == 'all'
    else:
        assert args[args.index('--device') + 1] == 'nvidia.com/gpu=all'
    image_index = args.index(available['bash'])
    assert args[image_index - 4:image_index] == ['--network', 'none', '--security-opt', 'label=disable']
    assert args[image_index + 1:] == ['bash', 'transform-inner.sh']
    assert 'seamless.tf=' + 'a' * 64 in args
    assert 'seamless.pid=' + str(os.getpid()) in args
    info = json.loads(engine(runtime, 'info', '--format', '{{json .}}'))
    if runtime == 'podman':
        rootless = info['host']['security']['rootless']
    else:
        rootless = any('rootless' in item for item in info.get('SecurityOptions', []))
    if rootless and runtime == 'podman':
        assert '--userns=keep-id' in args and '--user' not in args
    elif rootless:
        assert '--user' not in args
    else:
        assert args[args.index('--user') + 1] == f'{os.getuid()}:{os.getgid()}'


def test_comma_mount_paths_rejected(runtime_images, tmp_path):
    runtime, available = runtime_images
    with pytest.raises(ValueError, match='comma|,'):
        build_container_command(runtime, {'name': available['bash']}, str(tmp_path / 'a,b'), [])
    with pytest.raises(ValueError, match='comma|,'):
        build_container_command(runtime, {'name': available['bash']}, str(tmp_path), [str(tmp_path / 'a,b')])


def test_string_file_and_directory_pins_and_result_file(runtime_images):
    _, available = runtime_images
    result = execute('printf "%s|" "$text" > RESULT; cat data.txt folder/member >> RESULT',
        available['bash'], {'text': 'hello world', 'data.txt': b'file\n', 'folder': {'member': 'directory\n'}},
        {'folder': {'mode': 'directory', 'filesystem': False}})
    assert result == 'hello world|file\ndirectory\n'


def test_result_directory(runtime_images):
    _, available = runtime_images
    assert execute('mkdir -p RESULT/sub; printf alpha > RESULT/a; printf beta > RESULT/sub/b',
        available['bash'], output='folder') == {'a': b'alpha', 'sub/b': b'beta'}


def test_filesystem_symlink_resolves_and_read_folder_is_read_only(runtime_images, tmp_path):
    from seamless_remote import buffer_remote
    _, available = runtime_images
    folder = tmp_path / 'buffers'
    folder.mkdir()
    data = folder / 'input.txt'
    data.write_text('filesystem input\n')
    name = 'phase3-' + uuid.uuid4().hex
    buffer_remote.define_extern_client(name, 'bufferfolder', directory=str(folder))
    client = buffer_remote._extern_clients[name]
    buffer_remote._read_folders_clients.append(client)
    try:
        result = execute('cat "$data" > RESULT; test -L data; '
            'if printf bad > "$(dirname "$data")/forbidden"; then exit 9; fi',
            available['bash'], {'data': str(data)}, {'data': {'mode': 'file', 'filesystem': True}})
        assert result == 'filesystem input\n'
        assert not (folder / 'forbidden').exists()
    finally:
        buffer_remote._read_folders_clients.remove(client)
        del buffer_remote._extern_clients[name]


def test_job_directory_and_ownership(runtime_images):
    _, available = runtime_images
    result = execute('printf file > owned; printf "%s|%s|%s" "$PWD" "$(id -u)" "$(stat -c %u owned)" > RESULT', available['bash'])
    assert result == f'/seamless-job|{os.getuid()}|{os.getuid()}'


def test_image_entrypoint_is_bypassed(runtime_images):
    _, available = runtime_images
    assert execute('printf entrypoint-bypassed > RESULT', available['entrypoint']) == 'entrypoint-bypassed'


@pytest.mark.parametrize('kind,code', [('absent', 125), ('no_bash', 127)])
def test_environment_entry_failures(runtime_images, kind, code, capsys):
    runtime, available = runtime_images
    image = 'seamless-absent-' + uuid.uuid4().hex if kind == 'absent' else available[kind]
    with pytest.raises(Exception) as error:
        execute('printf should-not-run > RESULT', image)
    message = str(error.value)
    assert 'not entered' in message.lower()
    assert runtime in message and image in message
    expected_code = 1 if kind == 'no_bash' and runtime == 'podman' else code
    assert f'Return code: {expected_code}' in message
    assert 'bash' in message.lower()
    if kind == 'no_bash' and runtime == 'podman':
        stderr = capsys.readouterr().err
        assert 'failed to exec pid1' in stderr and 'No such file or directory' in stderr
    if kind == 'absent':
        assert 'runtime' in message.lower() or 'image' in message.lower()


def test_host_environment_and_notify_do_not_reach_script(runtime_images, monkeypatch):
    _, available = runtime_images
    monkeypatch.setenv('SEAMLESS_PROCESS_LOCK_NOTIFY', '/host/notify')
    monkeypatch.setenv('SEAMLESS_PHASE3_HOST_SECRET', 'host-only')
    assert execute('test -z "${SEAMLESS_PROCESS_LOCK_NOTIFY+x}"; '
        'test -z "${SEAMLESS_PHASE3_HOST_SECRET+x}"; printf isolated > RESULT',
        available['bash']) == 'isolated'


def test_required_binaries_checked_inside_image(runtime_images):
    _, available = runtime_images
    assert execute('printf present > RESULT', available['bash'], which=['bash', 'cat']) == 'present'
    missing = 'missing_binary_' + uuid.uuid4().hex
    with pytest.raises(Exception, match='Required environment binaries are unavailable'):
        execute('printf should-not-run > RESULT', available['bash'], which=[missing])


@pytest.mark.parametrize('mode', ['process', 'spawn'])
def test_public_transformer_and_separate_logs(runtime_images, mode):
    runtime, available = runtime_images
    script = '''
import uuid
import seamless
from seamless_config.select import select_execution
from seamless_transformer import Transformer
select_execution(MODE)
tf = Transformer('bash')
tf.code = '# ' + uuid.uuid4().hex + '\\nprintf "docker-stdout\\n"; printf "docker-stderr\\n" >&2; printf result > RESULT'
tf.celltypes.result = 'text'
tf.environment.set_docker({'name': IMAGE})
job = tf.transformation()
try:
    job.compute()
    assert job.value == 'result', (job.value, job.exception)
    log = job.logs
    assert isinstance(log, str), log
    separator = '*' * 49
    stdout_block = separator + '\\n* Standard output\\n' + separator + '\\ndocker-stdout\\n\\n'
    stderr_block = separator + '\\n* Standard error\\n' + separator + '\\ndocker-stderr\\n\\n'
    assert log.startswith(stdout_block + stderr_block), log
    assert 'Execution time:' in log and not log.endswith('\\n'), log
finally:
    seamless.close()
'''.replace('MODE', repr(mode)).replace('IMAGE', repr(available['bash']))
    completed = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_raw_dict_tf_dunder_enters_container(runtime_images):
    from seamless import Buffer
    from seamless_config.select import select_execution
    from seamless_transformer.transformation_class import transformation_from_dict
    _, available = runtime_images
    select_execution('process')
    code = Buffer('# ' + uuid.uuid4().hex + '\nprintf "%s" "$PWD" > RESULT', 'text')
    environment = Buffer({'docker': {'name': available['bash']}}, 'plain')
    code.tempref()
    environment.tempref()
    payload = {'__language__': 'bash', 'code': ('text', None, code.get_checksum().hex()),
               '__output__': ('result', 'text', None)}
    job = transformation_from_dict(payload, tf_dunder={'__env__': environment.get_checksum().hex()})
    job.compute()
    assert job.value == '/seamless-job', (job.value, job.exception)


def test_materialized_launcher_runs_after_move_and_files_have_host_owner(runtime_images, tmp_path):
    from seamless_transformer.execute_bash import write_bash_job
    runtime, available = runtime_images
    original = tmp_path / 'original'
    original.mkdir()
    write_bash_job('printf owned > RESULT; printf "%s" "$PWD" > location', [], '', {}, {},
        directory=str(original), docker_={'name': available['bash']}, container_runtime_=runtime)
    moved = tmp_path / 'moved'
    original.rename(moved)
    completed = subprocess.run(['bash', 'transform.sh'], cwd=moved,
        capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (moved / 'RESULT').read_text() == 'owned'
    assert (moved / 'location').read_text() == '/seamless-job'
    assert (moved / 'RESULT').stat().st_uid == os.getuid()
    assert (moved / 'RESULT').stat().st_gid == os.getgid()


def test_portable_launcher_rejects_actual_comma_path(runtime_images, tmp_path):
    runtime, available = runtime_images
    with pytest.raises(ValueError, match='comma|,'):
        build_container_command(runtime, {'name': available['bash']},
            str(tmp_path / 'job,comma'), [], portable_job_directory=True)


def test_raw_write_remote_job_rejects_comma_path(runtime_images, tmp_path):
    from seamless import Buffer
    from seamless_transformer.run import run_transformation_dict
    _, available = runtime_images
    directory = tmp_path / 'raw,job'
    code = Buffer('# ' + uuid.uuid4().hex + '\nprintf should-not-run > RESULT', 'text')
    environment = Buffer({'docker': {'name': available['bash']}}, 'plain')
    code.tempref()
    environment.tempref()
    payload = {'__language__': 'bash', 'code': ('text', None, code.get_checksum().hex()),
               '__output__': ('result', 'text', None)}
    with pytest.raises(ValueError, match='comma|,'):
        run_transformation_dict(payload, Buffer(payload, 'plain').get_checksum(),
            {'__env__': environment.get_checksum().hex(),
             '__meta__': {'write_remote_job': str(directory)}}, scratch=True, require_value=True)
    assert not (directory / 'RESULT').exists()


@pytest.mark.parametrize('failure', ['script', 'startup'])
def test_public_failure_has_separate_stream_trailers(runtime_images, failure):
    runtime, available = runtime_images
    image = available['bash'] if failure == 'script' else available['no_bash']
    code = '# ' + uuid.uuid4().hex + '\nprintf "failure-stdout\\n"; printf "failure-stderr\\n" >&2; exit 7'
    script = f'''
import seamless
from seamless_config.select import select_execution
from seamless_transformer import Transformer
select_execution('process')
tf = Transformer('bash')
tf.code = {code!r}
tf.celltypes.result = 'text'
tf.environment.set_docker({{'name': {image!r}}})
job = tf.transformation()
try:
    job.compute()
    assert job.exception, 'failure must be recorded'
    log = job.logs
    assert log == job.exception and isinstance(log, str), log
    assert '* Standard error' in log, log
    if {failure!r} == 'script':
        separator = '*' * 49
        assert separator + '\\n* Standard output\\n' + separator + '\\nfailure-stdout\\n' in log, log
        assert separator + '\\n* Standard error\\n' + separator + '\\nfailure-stderr\\n' in log, log
        assert log.index('failure-stderr', log.index('* Standard error')) > log.index('* Standard error'), log
    else:
        assert 'not entered' in log.lower() and {runtime!r} in log and {image!r} in log, log
        assert 'No such file or directory' in log or 'executable file not found' in log, log
    assert 'Execution time:' not in log, log
finally:
    seamless.close()
'''
    completed = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_container_streams_arrive_before_job_finishes(runtime_images, tmp_path, monkeypatch):
    import io
    import threading
    import time
    from seamless_remote import buffer_remote
    _, available = runtime_images
    directory = tmp_path / 'stream-gate'
    directory.mkdir()
    name = 'stream-' + uuid.uuid4().hex
    buffer_remote.define_extern_client(name, 'bufferfolder', directory=str(directory))
    client = buffer_remote._extern_clients[name]
    buffer_remote._read_folders_clients.append(client)
    started = threading.Event()

    class Stream(io.StringIO):
        def write(self, text):
            result = super().write(text)
            if 'CONTAINER_LIVE_STDOUT' in self.getvalue():
                started.set()
            return result

    stdout, stderr = Stream(), io.StringIO()
    monkeypatch.setattr(sys, 'stdout', stdout)
    monkeypatch.setattr(sys, 'stderr', stderr)
    results, failures = [], []

    def run():
        try:
            results.append(execute('echo CONTAINER_LIVE_STDOUT; echo CONTAINER_LIVE_STDERR >&2; '
                'while ! test -f "$gate/release"; do sleep .02; done; printf done > RESULT',
                available['bash'], {'gate': str(directory)},
                {'gate': {'filesystem': True, 'mode': 'directory'}}))
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert started.wait(15), stdout.getvalue() + stderr.getvalue()
        deadline = time.monotonic() + 5
        while 'CONTAINER_LIVE_STDERR' not in stderr.getvalue() and time.monotonic() < deadline:
            time.sleep(.01)
        assert 'CONTAINER_LIVE_STDERR' in stderr.getvalue()
        assert thread.is_alive(), failures
    finally:
        (directory / 'release').touch()
        thread.join(20)
        buffer_remote._read_folders_clients.remove(client)
        del buffer_remote._extern_clients[name]
    assert not thread.is_alive()
    assert not failures, failures
    assert results == ['done']
