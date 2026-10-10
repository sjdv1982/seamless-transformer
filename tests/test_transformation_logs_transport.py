"""Phase C: capture/transport/store, before the public formatter surface."""
import asyncio
import sys
import threading
from types import SimpleNamespace

import pytest
from seamless import Checksum
from seamless_transformer import worker
import seamless_transformer.transformation_cache as tc


def _execute_child_request(payload):
    # Install during the call: pytest replaces its capture streams between phases.
    from seamless_transformer.output_routing import install_output_routing
    original_stdout, original_stderr = sys.stdout, sys.stderr
    original_thread_start = threading.Thread.start
    try:
        install_output_routing()
        return worker._execute_transformation_request(payload)
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
        threading.Thread.start = original_thread_start


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('want_logs', [False, True])
def test_worker_success_opt_in_and_independent_truncation(monkeypatch, streaming, want_logs):
    result = Checksum('1' * 64)
    stdout = 'α' * 4960 + 'DROP' * 30 + '尾' * 5000
    stderr = 'short error\n'
    def execute(payload, formatter):
        sys.stdout.write(stdout)
        sys.stderr.write(stderr)
        return result
    monkeypatch.setattr(worker, '_execute_transformation_impl', execute)
    reply = _execute_child_request({'want_logs': want_logs, 'streaming': streaming})
    if not want_logs:
        assert reply == result
    else:
        checksum, logs = reply
        assert checksum == result
        assert logs == {'stdout': 'α' * 4960 + '\n...(skipped 120 characters)...\n' + '尾' * 5000, 'stderr': stderr}


@pytest.mark.parametrize('length', [9999, 10000, 10001])
def test_capture_truncation_boundary(monkeypatch, length):
    def execute(payload, formatter):
        sys.stdout.write('x' * length)
        return Checksum('2' * 64)
    monkeypatch.setattr(worker, '_execute_transformation_impl', execute)
    _, logs = _execute_child_request({'want_logs': True})
    expected = 'x' * length if length <= 10000 else 'x' * 4960 + '\n...(skipped 41 characters)...\n' + 'x' * 5000
    assert logs == {'stdout': expected, 'stderr': ''}


@pytest.mark.parametrize('streaming', [False, True])
def test_failure_reply_unchanged_and_full_output(monkeypatch, streaming):
    output = 'FAIL_OUTPUT' * 1100
    def execute(payload, formatter):
        sys.stdout.write(output)
        sys.stderr.write('failure error\n')
        return 'original failure'
    monkeypatch.setattr(worker, '_execute_transformation_impl', execute)
    bare = _execute_child_request({'streaming': streaming})
    requested = _execute_child_request({'want_logs': True, 'streaming': streaming})
    assert requested == bare
    assert isinstance(requested, str)
    assert output in requested
    assert 'failure error\n' in requested
    assert 'Execution time' not in requested
    assert 'skipped' not in requested


@pytest.mark.parametrize('want_logs', [False, True])
def test_manager_sets_opt_in_and_unwraps_reply(monkeypatch, want_logs):
    result = Checksum('3' * 64)
    logs = {'stdout': 'hello', 'stderr': 'warning'}
    payloads = []
    async def request(method, payload):
        payloads.append(payload)
        return (result, logs) if payload.get('want_logs') else result
    handle = SimpleNamespace(name='fake', wait_until_ready=lambda: asyncio.sleep(0), request=request)
    manager = worker._WorkerManager.__new__(worker._WorkerManager)
    manager._handles = [handle]
    manager._load = {}
    manager._limits = {}
    manager._delegate_owner_by_handle = {}
    manager._active_handles = {}
    manager._active_dispatches = {}
    manager._active_dispatch_lock = threading.RLock()
    manager._cancelled_checksums = set()
    monkeypatch.setattr(manager, '_prefetch_transformation_assets', lambda *a: asyncio.sleep(0))
    reply = asyncio.run(manager._dispatch({}, result, {}, False, want_logs=want_logs))
    assert bool(payloads[0].get('want_logs')) is want_logs
    assert reply == ((result, logs) if want_logs else result)
    assert manager._load['fake'] == 0


@pytest.fixture
def isolated_cache(monkeypatch):
    cache = tc.TransformationCache()
    monkeypatch.setattr(tc, 'get_execution', lambda: 'spawn')
    monkeypatch.setattr(tc.worker, 'has_spawned', lambda: True)
    monkeypatch.setattr(tc, 'is_worker', lambda: False)
    monkeypatch.setattr(tc, 'database_remote', None)
    monkeypatch.setattr(tc, 'buffer_remote', None)
    monkeypatch.setattr(tc, '_await_buffer_writer', lambda *a: asyncio.sleep(0))
    monkeypatch.setattr(Checksum, 'tempref', lambda *a, **k: None)
    monkeypatch.setattr(Checksum, 'mark_scratch', lambda *a, **k: None)
    return cache


def _uncached(cache, checksum):
    return cache._run_uncached({'__language__': 'python', '__output__': ('result', 'mixed', None)}, tf_checksum=checksum, tf_dunder={}, scratch=True, require_value=False, force_local=False, store_execution_record=False, strict_dunder=False, record_mode=False)


def test_spawn_store_cleared_in_flight_replaced_on_success(monkeypatch, isolated_cache):
    cache = isolated_cache
    checksum = Checksum('4' * 64)
    cache.set_transformation_log(checksum, {'stdout': 'old', 'stderr': 'old error', 'wall_time_seconds': 99})
    async def dispatch(*a, **kwargs):
        assert kwargs['with_logs'] is True
        assert cache.get_transformation_log(checksum) is None
        return Checksum('5' * 64), {'stdout': 'new', 'stderr': 'new error'}
    monkeypatch.setattr(tc.worker, 'dispatch_to_workers', dispatch)
    clock = iter([10.0, 12.375])
    monkeypatch.setattr(tc.time, 'perf_counter', lambda: next(clock))
    assert asyncio.run(_uncached(cache, checksum)) == Checksum('5' * 64)
    assert cache.get_transformation_log(checksum) == {'stdout': 'new', 'stderr': 'new error', 'wall_time_seconds': 2.375}


@pytest.mark.parametrize('failure', ['failed with output', 'Transformation was canceled'])
def test_failed_or_canceled_dispatch_leaves_no_stored_log(monkeypatch, isolated_cache, failure):
    cache = isolated_cache
    checksum = Checksum('6' * 64)
    cache.set_transformation_log(checksum, {'stdout': 'old', 'stderr': '', 'wall_time_seconds': 1})
    async def dispatch(*a, **k):
        assert cache.get_transformation_log(checksum) is None
        return failure, None
    monkeypatch.setattr(tc.worker, 'dispatch_to_workers', dispatch)
    with pytest.raises(Exception):
        asyncio.run(_uncached(cache, checksum))
    assert cache.get_transformation_log(checksum) is None


def test_store_replace_clear_and_checksum_normalization(isolated_cache):
    cache = isolated_cache
    checksum = Checksum('7' * 64)
    assert cache.get_transformation_log(checksum) is None
    cache.set_transformation_log(checksum, {'stdout': 'first', 'stderr': '', 'wall_time_seconds': 1})
    cache.set_transformation_log(Checksum(checksum.hex()), {'stdout': '', 'stderr': 'second', 'wall_time_seconds': 2})
    assert cache.get_transformation_log(checksum) == {'stdout': '', 'stderr': 'second', 'wall_time_seconds': 2}
    cache.clear_transformation_log(checksum)
    assert cache.get_transformation_log(checksum) is None


def test_log_store_evicts_least_recently_read_entry():
    cache = tc.TransformationCache(log_cache_size=2)
    a, b, c = (Checksum(ch * 64) for ch in 'abc')
    entry = {'stdout': 'output', 'stderr': '', 'wall_time_seconds': 1}
    cache.set_transformation_log(a, entry)
    cache.set_transformation_log(b, entry)
    assert cache.get_transformation_log(a) == entry
    cache.set_transformation_log(c, entry)
    assert cache.get_transformation_log(b) is None
    assert cache.get_transformation_log(a) == entry
    assert cache.get_transformation_log(c) == entry


@pytest.mark.parametrize('include_logs', [True, False])
def test_jobserver_store_uses_server_runtime_and_missing_logs_fallback(monkeypatch, isolated_cache, include_logs):
    cache = isolated_cache
    checksum = Checksum('c' * 64)
    async def remote_run(*a, **k):
        response = {'result_checksum': Checksum('d' * 64), 'record_runtime': {'wall_time_seconds': 8.125}}
        if include_logs:
            response['logs'] = {'stdout': 'server output', 'stderr': 'server error'}
        return response
    monkeypatch.setattr(tc, 'get_execution', lambda: 'remote')
    monkeypatch.setattr(tc, 'jobserver_remote', SimpleNamespace(run_transformation=remote_run))
    monkeypatch.setattr(tc, 'database_remote', SimpleNamespace(has_write_server=lambda: True))
    monkeypatch.setattr(tc, 'buffer_remote', SimpleNamespace(has_write_server=lambda: True))
    monkeypatch.setattr(cache, '_record_transformation_result', lambda *a, **k: asyncio.sleep(0))
    from seamless.caching import buffer_writer
    monkeypatch.setattr(buffer_writer, 'flush', lambda: None)
    asyncio.run(_uncached(cache, checksum))
    assert cache.get_transformation_log(checksum) == {'stdout': 'server output' if include_logs else '', 'stderr': 'server error' if include_logs else '', 'wall_time_seconds': 8.125}
