"""Phase 2: real conda wrapping; fresh identities prevent environment cache hits."""
import importlib
import io
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
import uuid

import pytest


def _code(body):
    return '# nonce: ' + uuid.uuid4().hex + '\n' + body


def _execute(body, environment='seamless1'):
    module = importlib.import_module('seamless_transformer.execute_bash')
    nonce = uuid.uuid4().hex
    return module.execute_bash(_code(body), ['nonce'], environment, {'nonce': nonce}, {}, 'text')


def _isolated(mode):
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).parent) + os.pathsep + env.get('PYTHONPATH', '')
    result = subprocess.run([sys.executable, '-c',
        f"import test_bash_conda as m; m._public_run({mode!r})"],
        env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'CONDA_OK' in result.stdout


def _public_run(mode):
    import seamless
    from seamless_config.select import select_execution
    from seamless_transformer import Transformer
    select_execution(mode)
    tf = Transformer('bash', direct=True)
    tf.code = _code('printf "%s" "$CONDA_PREFIX" > RESULT')
    tf.environment.set_conda_env('seamless1')
    tf.celltypes.result = 'text'
    try:
        assert Path(tf()).resolve() == Path(sys.prefix).resolve()
        print('CONDA_OK')
    finally:
        seamless.close()


@pytest.mark.parametrize('mode', ['process', 'spawn'])
def test_named_conda_environment_on_real_backend(mode):
    _isolated(mode)


def test_prefix_environment():
    assert Path(_execute('printf "%s" "$CONDA_PREFIX" > RESULT', sys.prefix)).resolve() == Path(sys.prefix).resolve()


@pytest.mark.parametrize('exitcode', [1, 7, 127])
def test_script_failure_is_not_environment_entry_failure(exitcode):
    with pytest.raises(Exception) as error:
        _execute(f'echo BODY_ENTERED >&2; exit {exitcode}')
    message = str(error.value)
    assert 'Bash transformer exception' in message
    assert str(exitcode) in message
    assert 'seamless1' in message
    assert 'not entered' not in message.lower()


def test_missing_environment_is_not_entered():
    name = 'seamless_missing_' + uuid.uuid4().hex
    with pytest.raises(Exception) as error:
        _execute('printf should_not_run > RESULT', name)
    message = str(error.value)
    assert 'not entered' in message.lower()
    assert name in message
    assert 'conda' in message.lower()
    assert 'transform-inner.sh' in message
    assert '1' in message


def test_stdout_and_stderr_arrive_before_exit(monkeypatch, tmp_path):
    release = tmp_path / 'release'
    started = threading.Event()
    class Stream(io.StringIO):
        def write(self, text):
            result = super().write(text)
            if 'CONDA_LIVE_STDOUT' in self.getvalue():
                started.set()
            return result
    stdout, stderr = Stream(), io.StringIO()
    monkeypatch.setattr(sys, 'stdout', stdout)
    monkeypatch.setattr(sys, 'stderr', stderr)
    results, errors = [], []
    def run():
        try:
            results.append(_execute('echo CONDA_LIVE_STDOUT; echo CONDA_LIVE_STDERR >&2; '
                f'while ! test -f {shlex.quote(str(release))}; do sleep .02; done; printf done > RESULT'))
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert started.wait(15), stdout.getvalue() + stderr.getvalue()
        deadline = time.monotonic() + 5
        while 'CONDA_LIVE_STDERR' not in stderr.getvalue() and time.monotonic() < deadline:
            time.sleep(.01)
        assert 'CONDA_LIVE_STDERR' in stderr.getvalue()
        assert thread.is_alive()
    finally:
        release.touch()
        thread.join(20)
    assert not thread.is_alive()
    assert not errors, errors
    assert results == ['done']


def test_which_checked_after_activation(monkeypatch):
    from seamless_config.select import select_execution
    from seamless_transformer import Transformer
    select_execution('process')
    # dask is provided by seamless1, not the deliberately restricted host PATH.
    assert (Path(sys.prefix) / 'bin/dask').exists()
    monkeypatch.setenv('PATH', '/usr/bin:/bin')
    tf = Transformer('bash', direct=True)
    tf.code = _code('command -v dask > RESULT')
    tf.environment.set_conda_env('seamless1')
    tf.environment.set_which(['dask'])
    tf.celltypes.result = 'text'
    assert Path(tf().strip()).parent == Path(sys.prefix) / 'bin'


def test_missing_required_binary_is_classified_inside_environment():
    from seamless_config.select import select_execution
    from seamless_transformer import Transformer
    select_execution('process')
    missing = 'seamless_missing_binary_' + uuid.uuid4().hex
    tf = Transformer('bash', direct=True)
    tf.code = _code('printf should_not_run > RESULT')
    tf.environment.set_conda_env('seamless1')
    tf.environment.set_which([missing])
    with pytest.raises(Exception) as error:
        tf()
    message = str(error.value)
    assert 'Required environment binaries are unavailable' in message
    assert missing in message
    assert 'seamless1' in message


def test_which_check_preserves_pins_named_after_its_variables():
    module = importlib.import_module('seamless_transformer.execute_bash')
    result = module.execute_bash(
        'printf "%s|%s" "$binary" "$missing_binaries" > RESULT',
        ['binary', 'missing_binaries'],
        None,
        {'binary': 'user-binary', 'missing_binaries': 'user-count'},
        {},
        'text',
        which_=['bash'],
    )
    assert result == 'user-binary|user-count'


@pytest.mark.parametrize('environment, selector', [('seamless1', '-n'), (sys.prefix, '-p')])
def test_launcher_text_and_inner_script(tmp_path, environment, selector):
    module = importlib.import_module('seamless_transformer.execute_bash')
    module.write_bash_job(_code('printf ok > RESULT'), ['nonce'], environment,
        {'nonce': uuid.uuid4().hex}, {}, directory=str(tmp_path))
    launcher = (tmp_path / 'transform.sh').read_text()
    lines = [line for line in launcher.splitlines() if line and not line.startswith('#')]
    assert len(lines) == 1
    argv = shlex.split(lines[0])
    assert argv[0] == 'exec'
    assert argv[2:] == ['run', selector, environment, '--no-capture-output', 'bash', 'transform-inner.sh']
    inner = (tmp_path / 'transform-inner.sh').read_text()
    assert 'export nonce=' in inner
    assert '.seamless-started' in inner
    assert 'conda activate' not in inner
    assert 'conda.sh' not in inner


def test_write_remote_job_runs_after_directory_is_moved(tmp_path):
    module = importlib.import_module('seamless_transformer.run')
    original = tmp_path / 'original'
    moved = tmp_path / 'moved job'
    nonce = uuid.uuid4().hex
    module._write_remote_bash_job(str(original), _code('printf "%s:%s" "$nonce" "$CONDA_PREFIX" > RESULT'),
        ['nonce'], 'seamless1', {'nonce': nonce}, {})
    assert not (original / 'RESULT').exists()
    original.rename(moved)
    result = subprocess.run(['bash', 'transform.sh'], cwd=moved, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (moved / 'RESULT').read_text() == nonce + ':' + sys.prefix
    assert (moved / '.seamless-started').exists()


def test_notify_variable_is_inherited():
    from seamless_transformer.process_lock import scope, current_holder
    with scope(True):
        holder = current_holder()
        assert holder.held
        # The real lock owns this FIFO. Conda must preserve the notification
        # endpoint supplied by execute_bash, rather than a stale host variable.
        assert _execute('test -p "$SEAMLESS_PROCESS_LOCK_NOTIFY"; '
            'printf x > "$SEAMLESS_PROCESS_LOCK_NOTIFY"; printf done > RESULT') == 'done'
        assert not holder.held
