"""Opt-in Phase 6 real-backend matrix; never collected by pytest.

Run with conda run --live-stream -n seamless1 python <this file> --output results.jsonl.
Each cell has a separate process group, three uncached transformations, and a
hard deadline. Diagnostic wrappers delegate unchanged to the shipped handlers.
"""
import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


SNIPPETS = {
    "julia": {
        "compute": "result = value * value + 1",
        "sleep": "sleep(0.2); result = value * value + 1",
        "println": 'println("matrix println ", value); result = value * value + 1',
        "async": "result = fetch(@async (sleep(0.1); value * value + 1))",
        "threads_spawn": "result = fetch(Threads.@spawn (sleep(0.1); value * value + 1))",
        "subprocess": 'run(`sh -c "exit 0"`); result = value * value + 1',
        "allocation_gc": "for i in 1:20; zeros(100000); GC.gc(); end; result = value * value + 1",
        "long10": "deadline = time() + 10; while time() < deadline; sqrt(Float64(value)); end; result = value * value + 1",
    },
    "r": {
        "compute": "result <- value * value + 1",
        "sleep": "Sys.sleep(0.2); result <- value * value + 1",
        "println": 'cat("matrix println", value, "\\n"); result <- value * value + 1',
        # Base R has no Julia task scheduler: evaluate a delayed promise, and
        # exercise a fork job via the installed standard parallel package.
        "async": "delayedAssign('job', { Sys.sleep(0.1); value * value + 1 }); result <- job",
        "threads_spawn": "job <- parallel::mcparallel(value * value + 1); result <- parallel::mccollect(job)[[1]]",
        "subprocess": "stopifnot(system2('sh', c('-c', shQuote('exit 0'))) == 0); result <- value * value + 1",
        "allocation_gc": "for (i in seq_len(20)) { numeric(100000); gc() }; result <- value * value + 1",
        "long10": "deadline <- Sys.time() + 10; while (Sys.time() < deadline) sqrt(value); result <- value * value + 1",
    },
}


def observing_handler(code, pins, filesystem, output_name, output_celltype,
                      metavars, environment, process_lock):
    """Self-contained shipped source: no closures or diagnostic result changes."""
    import json
    import os
    import threading
    import time
    from seamless_transformer.languages.handlers.julia import julia
    from seamless_transformer.languages.handlers.r import r

    def emit(phase, **extra):
        record = dict(phase=phase, pid=os.getpid(), native_id=threading.get_native_id(),
                      ident=threading.get_ident(), time=time.time(), value=pins['value'],
                      process_lock=process_lock,
                      threads=[dict(name=t.name, ident=t.ident, native_id=t.native_id,
                                    alive=t.is_alive()) for t in threading.enumerate()])
        record.update(extra)
        data = (json.dumps(record) + '\n').encode()
        fd = os.open(os.environ['SEAMLESS_MATRIX_EVENTS'], os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)
        print('MATRIX_BODY ' + json.dumps(record), flush=True)

    emit('body_start')
    assert process_lock is True
    handler = julia if metavars['BACKEND'] == 'juliacall' else r
    try:
        result = handler(code=code, pins=pins, filesystem=filesystem,
                         output_name=output_name, output_celltype=output_celltype,
                         metavars=metavars, environment=environment, process_lock=process_lock)
    except BaseException as exc:
        emit('body_error', error=repr(exc))
        raise
    emit('body_complete', result=result,
         signals=os.environ.get('PYTHON_JULIACALL_HANDLE_SIGNALS'))
    return result


def event(phase, **extra):
    import threading
    record = dict(phase=phase, pid=os.getpid(), time=time.time(),
                  threads=[dict(name=t.name, ident=t.ident, native_id=t.native_id,
                                alive=t.is_alive()) for t in threading.enumerate()])
    record.update(extra)
    with open(os.environ['SEAMLESS_MATRIX_EVENTS'], 'a') as stream:
        stream.write(json.dumps(record) + '\n')
    print('MATRIX_DRIVER ' + json.dumps(record), flush=True)


def child(cell):
    import seamless
    import numpy as np
    from seamless.transformer import Transformer
    from seamless_transformer import spawn
    from seamless_transformer.languages import register_language_handler

    event('driver_start', cell=cell)
    backend = 'juliacall' if cell['language'] == 'julia' else 'ryp'
    register_language_handler(cell['language'], observing_handler, metavars={'BACKEND': backend})
    if cell['mode'] == 'spawn':
        spawn(1)
    if cell['language'] == 'julia':
        runtime_code = r'''open(ENV["SEAMLESS_MATRIX_EVENTS"], "a") do io
    println(io, "{\"phase\":\"runtime_version\",\"language\":\"julia\",\"version\":\"", VERSION, "\",\"pid\":", getpid(), "}")
end
'''
    else:
        runtime_code = r'''matrix_connection <- file(Sys.getenv("SEAMLESS_MATRIX_EVENTS"), "a")
writeLines(sprintf('{"phase":"runtime_version","language":"r","version":"%s","pid":%d}', as.character(getRversion()), Sys.getpid()), matrix_connection)
close(matrix_connection)
value <- as.double(value)
'''
    for repetition in range(3):
        value = cell['salt'] + repetition + 1
        tf = Transformer(cell['language'])
        # Compose text locally: tf.code reads back as a Buffer, not a string.
        code = runtime_code + SNIPPETS[cell['language']][cell['snippet']]
        if cell['snippet'] == 'allocation_gc':
            tf.celltypes.array = 'binary'
            tf.pins.array = np.full(100000, value, dtype=np.float64)
            code += ('; @assert sum(array) == 100000 * value' if cell['language'] == 'julia'
                     else '; stopifnot(sum(array) == 100000 * value)')
        tf.code = code
        tf.metavars.BACKEND = backend
        tf.celltypes.value = 'plain'
        tf.pins.value = value
        tf.celltypes.result = 'plain'
        tf.process_lock = True
        tf.local = cell['mode'] == 'inprocess'
        result = tf.build().run()
        assert result == value * value + 1, (value, result)
        event('result_verified', value=value, result=result, repetition=repetition)
    event('computations_complete')
    seamless.close()
    event('close_complete')
    # Natural Python shutdown is part of the experiment. Do not use os._exit.


def matrix():
    for language in SNIPPETS:
        for helper in ('1', '0'):
            for mode in ('inprocess', 'spawn'):
                for signals in (('no', 'yes') if language == 'julia' and helper == '0' else (None,)):
                    group = ('inline-signals-' + signals if signals else
                             'helper-on-default-signals' if language == 'julia' else
                             'ryp-helper-env-' + helper)
                    for snippet in SNIPPETS[language]:
                        cell_id = '/'.join((language, helper, mode, signals or 'default', snippet))
                        yield dict(language=language, helper=helper, mode=mode,
                                   signals=signals, snippet=snippet, group=group, cell_id=cell_id)


def validate(events, cell):
    starts = [e for e in events if e['phase'] == 'body_start']
    ends = [e for e in events if e['phase'] == 'body_complete']
    verified = [e for e in events if e['phase'] == 'result_verified']
    versions = [e for e in events if e['phase'] == 'runtime_version']
    problems = []
    if len(starts) != 3 or len(ends) != 3 or len(verified) != 3:
        problems.append('expected three real bodies and three verified results')
    if len(versions) != 3 or any(not e.get('version') or e['language'] != cell['language'] for e in versions):
        problems.append('expected three actual native runtime version observations')
    if any(e.get('process_lock') is not True for e in starts):
        problems.append('process lock absent')
    body_pids = sorted({e['pid'] for e in starts})
    driver_pids = {e['pid'] for e in events if e['phase'] == 'driver_start'}
    if len(body_pids) != 1:
        problems.append('body PID changed (worker restart or missing body)')
    if body_pids and ((body_pids[0] not in driver_pids) != (cell['mode'] == 'spawn')):
        problems.append('execution mode PID mismatch')
    if not any(e['phase'] == 'close_complete' for e in events):
        problems.append('close did not finish')
    return problems


def run_cell(cell, timeout):
    with tempfile.TemporaryDirectory(prefix='seamless-matrix-') as temp:
        events_path = Path(temp) / 'events.jsonl'
        env = os.environ.copy()
        env['SEAMLESS_INTERPRETER_THREAD'] = cell['helper']
        env['SEAMLESS_MATRIX_EVENTS'] = str(events_path)
        env.pop('PYTHON_JULIACALL_HANDLE_SIGNALS', None)
        if cell['signals']:
            env['PYTHON_JULIACALL_HANDLE_SIGNALS'] = cell['signals']
        started = time.monotonic()
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--child', json.dumps(cell)],
                                env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, start_new_session=True)
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate(timeout=10)
        events = [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []
        problems = validate(events, cell)
        completed = any(e['phase'] == 'computations_complete' for e in events)
        status = ('hang(timeout)' if timed_out else 'crash' if proc.returncode < 0 else
                  'nonzero_error' if proc.returncode != 0 or problems else 'pass')
        pids = list(dict.fromkeys(e['pid'] for e in events if e['phase'] == 'body_start'))
        return dict(cell=cell, status=status, timeout_phase=('shutdown' if completed else 'body_or_initialization') if timed_out else None,
                    timeout_seconds=timeout, elapsed_seconds=time.monotonic()-started,
                    returncode=proc.returncode, body_pids=pids, worker_restarts=max(0, len(pids)-1),
                    validation_errors=problems, stdout=stdout, stderr=stderr, events=events,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())


def main():
    import hashlib

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='interpreter-thread-matrix.jsonl')
    parser.add_argument('--child')
    parser.add_argument('--language', choices=('julia', 'r'))
    parser.add_argument('--snippet', choices=tuple(SNIPPETS['julia']))
    parser.add_argument('--mode', choices=('inprocess', 'spawn'))
    parser.add_argument('--helper', choices=('0', '1'))
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    if args.child:
        child(json.loads(args.child))
        return
    cells = [c for c in matrix() if all(getattr(args, key) is None or getattr(args, key) == c[key]
                                       for key in ('language', 'snippet', 'mode', 'helper'))]
    if args.list:
        for c in cells:
            print(json.dumps(c))
        return
    root = Path(__file__).resolve().parents[3]
    repos = ('seamless-core', 'seamless-transformer', 'seamless-remote', 'seamless-config', 'seamless-dask', 'seamless-signature')
    os.environ['PYTHONPATH'] = os.pathsep.join(str(root / r) for r in repos) + os.pathsep + os.environ.get('PYTHONPATH', '')
    versions = {'python': sys.version}
    for package in ('juliacall', 'ryp', 'numpy'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    environment = {k: os.environ.get(k) for k in ('CONDA_PREFIX', 'CONDA_DEFAULT_ENV', 'JULIA_PROJECT', 'PYTHONPATH', 'PATH', 'PYTHON_JULIACALL_THREADS')}
    source_paths = [Path(__file__).resolve()] + [root / 'seamless-transformer/seamless_transformer/languages/handlers' / (name + '.py') for name in ('julia', 'r')]
    configuration = dict(schema=2, repetitions=3, normal_timeout=45, long_timeout=90,
                         cells=list(matrix()), versions=versions, environment=environment,
                         source_hashes={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
    fingerprint = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    completed = set()
    output = Path(args.output)
    if output.exists():
        data = output.read_bytes()
        offset = 0
        lines = data.splitlines(keepends=True)
        for index, line in enumerate(lines):
            try:
                record = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if index != len(lines) - 1:
                    raise ValueError('Malformed interior JSONL record; use a new output file')
                # An interrupted final write is not a finished cell.
                with output.open('r+b') as recovery:
                    recovery.truncate(offset)
                break
            if record.get('configuration_fingerprint') != fingerprint:
                raise ValueError('Configuration fingerprint changed; use a new output file')
            if record.get('record_complete') is True:
                completed.add(record['cell']['cell_id'])
            offset += len(line)
        if data and lines and not lines[-1].endswith(b'\n') and offset == len(data):
            with output.open('ab') as recovery:
                recovery.write(b'\n')
    salt = int(time.time_ns() % 10000000)
    with output.open('a', buffering=1) as stream:
        for index, cell in enumerate(cells):
            if cell['cell_id'] in completed:
                print(json.dumps(dict(cell_id=cell['cell_id'], resumed_skip=True)), flush=True)
                continue
            cell['salt'] = salt + index * 10
            result = run_cell(cell, 90 if cell['snippet'] == 'long10' else 45)
            result['versions'] = versions
            result['environment'] = dict(environment, SEAMLESS_INTERPRETER_THREAD=cell['helper'],
                                         PYTHON_JULIACALL_HANDLE_SIGNALS=cell['signals'])
            result['configuration_fingerprint'] = fingerprint
            result['configuration'] = configuration
            result['record_complete'] = True
            stream.write(json.dumps(result) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            print(json.dumps(dict(cell=cell, status=result['status'], timeout_phase=result['timeout_phase'])), flush=True)


if __name__ == '__main__':
    main()
