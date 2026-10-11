"""Phase F: nested handles, isolated child stores, and delegation protocol."""
import asyncio
import concurrent.futures
import sys
import threading
from types import SimpleNamespace

import pytest
from seamless import Checksum
from seamless_transformer import delayed, worker
import seamless_transformer.transformation_cache as tc
from seamless_transformer.transformation_class import Transformation

TF = Checksum('11' * 32)
RESULT = Checksum('22' * 32)
RAW = {'stdout': 'nested-output\n', 'stderr': 'nested-error\n', 'wall_time_seconds': 7.25}


def echo_poll_request_ids(request):
    """Protocol fixtures echo only request nonces actually present in the poll."""
    async def wrapped(kind, payload):
        response = await request(kind, payload)
        if kind != 'delegate_transformation_poll' or not isinstance(response, dict):
            return response
        request_ids = payload.get('delegate_request_ids', {})
        echoed = {}
        for token, reply in response.items():
            if token not in request_ids and 'delegate_request_id' not in reply:
                continue
            reply = dict(reply)
            if token in request_ids:
                reply.setdefault('delegate_request_id', request_ids[token])
            echoed[token] = reply
        return echoed
    return wrapped


@pytest.fixture
def cache(monkeypatch):
    value = tc.TransformationCache()
    monkeypatch.setattr(tc, 'get_transformation_cache', lambda: value)
    return value


def passive_handle(checksum):
    def forbidden(*args, **kwargs):
        raise AssertionError('logs must be passive')
    handle = Transformation('mixed', forbidden, forbidden, forbidden, forbidden)
    handle._transformation_checksum = checksum
    return handle


@pytest.fixture
def routing():
    from contextlib import contextmanager
    from seamless_transformer.output_routing import install_output_routing

    @contextmanager
    def during_call():
        # Pytest replaces capture streams between fixture setup and test calls.
        previous = sys.stdout, sys.stderr, threading.Thread.start
        install_output_routing()
        try:
            yield
        finally:
            sys.stdout, sys.stderr, threading.Thread.start = previous

    return during_call


def test_local_nested_capture_is_owned_by_inner_handle(cache, routing, monkeypatch):
    with routing():
        from seamless_transformer.output_routing import output_route
        import io
        outer_stdout, outer_stderr = io.StringIO(), io.StringIO()
        cache.set_transformation_log(TF, RAW)

        def execute(payload, formatter):
            assert cache.get_transformation_log(TF) is None
            print('inner-only')
            print('inner-error', file=sys.stderr)
            return RESULT

        monkeypatch.setattr(worker, '_execute_transformation_impl', execute)
        with output_route(stdout=outer_stdout, stderr=outer_stderr):
            print('outer-before')
            result = worker._execute_transformation({'tf_checksum': TF})
            print('outer-after')
        assert result == RESULT
        inner = passive_handle(TF)
        assert 'inner-only' in inner.logs
        assert 'inner-error' in inner.logs
        assert 'outer-before' not in inner.logs
        assert outer_stdout.getvalue() == 'outer-before\nouter-after\n'
        assert outer_stderr.getvalue() == ''
        stored = cache.get_transformation_log(TF)
        assert stored['stdout'] == 'inner-only\n'
        assert stored['stderr'] == 'inner-error\n'
        assert stored['wall_time_seconds'] >= 0


@pytest.mark.parametrize('reply', [None, 'failure text'])
def test_local_nested_failure_or_cancel_does_not_leave_stale_log(cache, routing, monkeypatch, reply):
    with routing():
        cache.set_transformation_log(TF, RAW)
        monkeypatch.setattr(worker, '_execute_transformation_impl', lambda *args: reply)
        result = worker._execute_transformation({'tf_checksum': TF})
        assert result == reply
        assert cache.get_transformation_log(TF) is None
        handle = passive_handle(TF)
        if reply is None:
            handle._cancelled = True
            assert handle.logs is None
        else:
            handle._exception = result
            assert handle.logs == handle.exception == reply


@pytest.mark.parametrize('logs', [RAW, None])
def test_immediate_delegation_stores_only_parent_held_log(cache, monkeypatch, logs):
    async def request(kind, payload):
        assert kind == 'delegate_transformation_submit'
        return {'status': 'done', 'result': RESULT.hex(), 'logs': logs}
    monkeypatch.setattr(worker, '_request_parent_async', echo_poll_request_ids(request))
    result = asyncio.run(worker.forward_to_parent({}, tf_checksum=TF, tf_dunder={}, scratch=True))
    assert result == RESULT
    assert cache.get_transformation_log(TF) == logs
    handle = passive_handle(TF)
    if logs is None:
        assert handle.logs is None
    else:
        assert 'nested-output' in handle.logs
        assert 'nested-error' in handle.logs
        assert 'Execution time: 7.2 seconds' in handle.logs


@pytest.mark.parametrize('logs', [RAW, None])
def test_polled_delegation_preserves_log_to_nested_handle(cache, monkeypatch, logs):
    async def run():
        future = concurrent.futures.Future()
        monkeypatch.setattr(worker, '_DELEGATE_PROXY_FUTURES', {})
        monkeypatch.setattr(worker, '_DELEGATE_POLL_INTERVAL', 0)
        def register(token):
            worker._DELEGATE_PROXY_FUTURES[token] = future
            return future
        monkeypatch.setattr(worker, '_register_delegate_token', register)
        async def request(kind, payload):
            if kind == 'delegate_transformation_submit':
                return {'status': 'submitted', 'token': 'token'}
            assert kind == 'delegate_transformation_poll'
            return {'token': {'status': 'done', 'result': RESULT.hex(), 'logs': logs}}
        monkeypatch.setattr(worker, '_request_parent_async', echo_poll_request_ids(request))
        poller = asyncio.create_task(worker._delegate_poll_loop())
        try:
            return await asyncio.wait_for(worker.forward_to_parent({}, tf_checksum=TF, tf_dunder={}, scratch=True), 2)
        finally:
            poller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await poller
    assert asyncio.run(run()) == RESULT
    assert cache.get_transformation_log(TF) == logs
    assert (passive_handle(TF).logs is None) == (logs is None)


@pytest.mark.parametrize('logs', [RAW, None])
def test_parent_cached_submit_returns_existing_log_without_fabrication(cache, monkeypatch, logs):
    manager = object.__new__(worker._WorkerManager)
    manager._delegate_owner_by_handle = {}
    if logs is not None:
        cache.set_transformation_log(TF, logs)
    async def cached(checksum):
        return RESULT
    monkeypatch.setattr(manager, '_get_cached_transformation_result', cached)
    response = asyncio.run(manager._handle_delegate_transformation_submit(None, {'tf_checksum': TF}))
    assert response['status'] == 'done'
    assert response['result'] == RESULT.hex()
    assert response.get('logs') == logs
    assert response['dispatch_started'] is False


@pytest.mark.parametrize('logs', [RAW, None])
def test_parent_poll_and_completed_entry_carry_thin_log(cache, monkeypatch, logs):
    manager = object.__new__(worker._WorkerManager)
    thin = SimpleNamespace(cancelled=lambda: False, done=lambda: True,
                           result=lambda: (TF.hex(), RESULT.hex(), None, logs))
    entry = {'futures': SimpleNamespace(thin=thin)}
    async def get(token):
        return entry
    async def pop(token, **kwargs):
        return entry
    monkeypatch.setattr(manager, '_get_delegate_entry', get)
    monkeypatch.setattr(manager, '_pop_delegate_entry', pop)
    monkeypatch.setattr(manager, '_release_delegate_entry', lambda *args, **kwargs: None)
    completed = manager._delegate_payload_from_entry(entry)
    polled = asyncio.run(manager._handle_delegate_transformation_poll(None, {'tokens': ['token']}))['token']
    for response in (completed, polled):
        assert response['status'] == 'done'
        assert response['result'] == RESULT.hex()
        assert response.get('logs') == logs


def test_local_nested_log_uses_its_dispatch_clock(cache, routing, monkeypatch):
    with routing():
        import time
        ticks = iter([10.0, 14.5])
        monkeypatch.setattr(worker, 'time', SimpleNamespace(perf_counter=lambda: next(ticks), monotonic=time.monotonic))
        monkeypatch.setattr(worker, '_execute_transformation_impl', lambda *args: RESULT)
        assert worker._execute_transformation({'tf_checksum': TF}) == RESULT
        assert cache.get_transformation_log(TF)['wall_time_seconds'] == 4.5
        assert 'Execution time: 4.5 seconds' in passive_handle(TF).logs


@pytest.mark.parametrize('delegated', [False, True])
def test_spawned_outer_reads_nested_handle_log_only_in_child(delegated):
    from seamless_transformer import has_spawned, spawn
    import uuid
    assert not has_spawned(), 'this integration test owns its worker lifetime'
    spawn(1)
    try:
        @delayed
        def outer(label, delegated):
            import sys
            from seamless_transformer import delayed, worker
            # Force the protocol's real local fallback, independently of backend availability.
            original = worker._request_parent_async
            async def refused(kind, payload):
                if kind == 'delegate_transformation_submit':
                    if not delegated:
                        return {'status': 'refused'}
                    # A protocol fixture models the parent execution boundary.
                    # Remove its held entry before delivering the parent's reply.
                    import asyncio
                    from seamless_transformer.transformation_cache import get_transformation_cache
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(None, worker._execute_transformation, payload)
                    store = get_transformation_cache()
                    raw = store.get_transformation_log(payload['tf_checksum'])
                    store.clear_transformation_log(payload['tf_checksum'])
                    raw['wall_time_seconds'] = 7.25
                    return {'status': 'done', 'result': result.hex(), 'logs': raw}
                return await original(kind, payload)
            worker._request_parent_async = refused
            try:
                @delayed
                def inner(label):
                    import sys
                    print('inner-' + label)
                    print('inner-error-' + label, file=sys.stderr)
                    return label
                print('outer-before')
                nested = inner(label)
                nested.compute()
                print('outer-after')
                return {'value': nested.value, 'logs': nested.logs,
                        'checksum': nested.transformation_checksum.hex()}
            finally:
                worker._request_parent_async = original
        label = uuid.uuid4().hex
        handle = outer(label, delegated)
        handle.compute()
        assert handle.exception is None
        value = handle.value
        assert value['value'] == label
        assert 'inner-' + label in value['logs']
        assert 'inner-error-' + label in value['logs']
        assert 'Execution time:' in value['logs']
        if delegated:
            assert 'Execution time: 7.2 seconds' in value['logs']
        assert 'outer-before' in handle.logs
        assert 'outer-after' in handle.logs
        assert 'inner-' + label not in handle.logs
        assert tc.get_transformation_cache().get_transformation_log(value['checksum']) is None
    finally:
        manager = worker._worker_manager
        if manager is not None:
            manager.close(wait=True)
        worker._worker_manager = None
        worker._set_has_spawned(False)


def nested_uncached(cache, active=None):
    return cache._run_uncached(
        {'__language__': 'python', '__output__': ('result', 'mixed', None)},
        tf_checksum=TF, tf_dunder={}, scratch=True, require_value=False,
        force_local=False, store_execution_record=False, strict_dunder=False,
        record_mode=False, active_submission=active,
    )


@pytest.fixture
def nested_execution(cache, monkeypatch):
    monkeypatch.setattr(tc, 'is_worker', lambda: True)
    monkeypatch.setattr(tc, 'get_execution', lambda: 'process')
    monkeypatch.setattr(worker, 'has_spawned', lambda: False)
    async def record(*args, **kwargs):
        pass
    async def flush(*args, **kwargs):
        pass
    monkeypatch.setattr(cache, '_record_transformation_result', record)
    monkeypatch.setattr(tc, '_await_buffer_writer', flush)
    return cache


@pytest.mark.parametrize('polled', [False, True])
def test_parent_cache_without_log_preserves_previous_child_log(nested_execution, monkeypatch, polled):
    cache = nested_execution
    cache.set_transformation_log(TF, RAW)
    assert TF not in cache._transformation_cache
    async def run():
        future = concurrent.futures.Future()
        monkeypatch.setattr(worker, '_DELEGATE_PROXY_FUTURES', {})
        monkeypatch.setattr(worker, '_DELEGATE_POLL_INTERVAL', 0)
        def register(token):
            worker._DELEGATE_PROXY_FUTURES[token] = future
            return future
        monkeypatch.setattr(worker, '_register_delegate_token', register)
        async def request(kind, payload):
            if kind == 'delegate_transformation_submit':
                if polled:
                    return {'status': 'submitted', 'token': 'cache-token'}
                return {'status': 'done', 'result': RESULT.hex()}
            assert kind == 'delegate_transformation_poll'
            return {'cache-token': {'status': 'done', 'result': RESULT.hex()}}
        monkeypatch.setattr(worker, '_request_parent_async', echo_poll_request_ids(request))
        poller = asyncio.create_task(worker._delegate_poll_loop()) if polled else None
        try:
            return await asyncio.wait_for(nested_uncached(cache), 2)
        finally:
            if poller is not None:
                poller.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await poller
    assert asyncio.run(run()) == RESULT
    assert cache.get_transformation_log(TF) == RAW
    assert 'Execution time: 7.2 seconds' in passive_handle(TF).logs


@pytest.mark.parametrize('marker_only', [False, True])
@pytest.mark.parametrize('failure', ['cancelled', 'buffer_writer'])
def test_post_delegation_failure_removes_new_success_entry(nested_execution, monkeypatch, failure, marker_only):
    cache = nested_execution
    active = SimpleNamespace(canceled=False)
    async def request(kind, payload):
        assert kind == 'delegate_transformation_submit'
        if failure == 'cancelled':
            active.canceled = True
        return {'status': 'done', 'result': RESULT.hex(), 'logs': None if marker_only else RAW, 'dispatch_started': True}
    monkeypatch.setattr(worker, '_request_parent_async', echo_poll_request_ids(request))
    if failure == 'buffer_writer':
        async def flush(*args, **kwargs):
            if marker_only:
                assert cache.get_transformation_log(TF) is None
            else:
                assert cache.get_transformation_log(TF) == RAW
            raise RuntimeError('buffer writer failed after delegation')
        monkeypatch.setattr(tc, '_await_buffer_writer', flush)
        expected, message = RuntimeError, 'buffer writer failed'
    else:
        from seamless_transformer.transformation_cache import TransformationCancelledError
        expected, message = TransformationCancelledError, 'canceled'
    with pytest.raises(expected, match=message):
        asyncio.run(nested_uncached(cache, active))
    assert cache.get_transformation_log(TF) is None
    handle = passive_handle(TF)
    if failure == 'cancelled':
        handle._cancelled = True
        assert handle.logs is None


def test_worker_jobserver_execution_stores_remote_streams_and_clock(nested_execution, monkeypatch):
    cache = nested_execution
    async def remote(*args, **kwargs):
        return {'result_checksum': RESULT, 'record_runtime': {'wall_time_seconds': 8.125},
                'logs': {'stdout': 'jobserver-output', 'stderr': 'jobserver-error'}}
    monkeypatch.setattr(tc, 'get_execution', lambda: 'remote')
    monkeypatch.setattr(tc, 'jobserver_remote', SimpleNamespace(run_transformation=remote))
    monkeypatch.setattr(tc, 'database_remote', SimpleNamespace(has_write_server=lambda: True))
    monkeypatch.setattr(tc, 'buffer_remote', SimpleNamespace(has_write_server=lambda: True))
    assert asyncio.run(nested_uncached(cache)) == RESULT
    assert cache.get_transformation_log(TF) == {
        'stdout': 'jobserver-output', 'stderr': 'jobserver-error', 'wall_time_seconds': 8.125}


def test_parent_inline_dispatch_uses_real_manager_signature_and_requests_logs(cache, monkeypatch):
    manager = object.__new__(worker._WorkerManager)
    async def ready():
        pass
    async def child_request(kind, payload):
        assert kind == 'execute_transformation'
        assert payload['want_logs'] is True
        return RESULT, {'stdout': 'inline-child-output', 'stderr': 'inline-child-error'}
    handle = SimpleNamespace(name='child', wait_until_ready=ready, request=child_request)
    manager._handles = [handle]
    manager._limits = {}
    manager._load = {}
    manager._delegate_owner_by_handle = {}
    async def cached(checksum):
        return None
    async def prefetch(*args):
        pass
    monkeypatch.setattr(manager, '_get_cached_transformation_result', cached)
    monkeypatch.setattr(manager, '_prefetch_transformation_assets', prefetch)
    monkeypatch.setattr(manager, '_remember_active_handle', lambda *args: None)
    monkeypatch.setattr(manager, '_forget_active_handle', lambda *args: None)
    # No transformation dict avoids creating a Dask submission; use actual _dispatch.
    response = asyncio.run(manager._handle_delegate_transformation_submit(None, {'tf_checksum': TF, 'scratch': True}))
    assert response['status'] == 'done', response
    assert response['result'] == RESULT.hex()
    assert response['logs']['stdout'] == 'inline-child-output'
    assert response['logs']['stderr'] == 'inline-child-error'
    assert response['logs']['wall_time_seconds'] >= 0
    assert response['dispatch_started'] is True


def test_nested_log_clears_at_polled_dispatch_and_ignores_stale_start(nested_execution, monkeypatch):
    cache = nested_execution
    cache.set_transformation_log(TF, RAW)
    fresh = {'stdout': 'fresh-output', 'stderr': '', 'wall_time_seconds': 2.5}
    async def run():
        monkeypatch.setattr(worker, '_DELEGATE_PROXY_FUTURES', {})
        monkeypatch.setattr(worker, '_DELEGATE_PROXY_DISPATCH_CALLBACKS', {})
        monkeypatch.setattr(worker, '_ensure_delegate_poller', lambda: None)
        monkeypatch.setattr(worker, '_DELEGATE_POLL_INTERVAL', 0)
        parent_lookup, allow_lookup = asyncio.Event(), asyncio.Event()
        poll_pending, allow_start = asyncio.Event(), asyncio.Event()
        after_start, allow_finish = asyncio.Event(), asyncio.Event()
        stale_seen, allow_stale = asyncio.Event(), asyncio.Event()
        token = 'in-flight-token'
        completed_request_id = None
        calls = 0
        async def request(kind, payload):
            nonlocal calls, completed_request_id
            if kind == 'delegate_transformation_submit':
                completed_request_id = payload['delegate_request_id']
                parent_lookup.set()
                await allow_lookup.wait()
                return {'status': 'submitted', 'token': token}
            assert kind == 'delegate_transformation_poll'
            calls += 1
            if calls == 1:
                poll_pending.set()
                await allow_start.wait()
                return {token: {'status': 'started'}}
            if calls == 2:
                after_start.set()
                await allow_finish.wait()
                return {token: {'status': 'done', 'result': RESULT.hex(), 'logs': fresh, 'dispatch_started': True}}
            stale_seen.set()
            await allow_stale.wait()
            return {token: {'status': 'started', 'delegate_request_id': completed_request_id}}
        monkeypatch.setattr(worker, '_request_parent_async', echo_poll_request_ids(request))
        poller = asyncio.create_task(worker._delegate_poll_loop())
        handle_task = asyncio.create_task(nested_uncached(cache))
        try:
            await asyncio.wait_for(parent_lookup.wait(), 2)
            assert cache.get_transformation_log(TF) == RAW
            allow_lookup.set()
            await asyncio.wait_for(poll_pending.wait(), 2)
            assert cache.get_transformation_log(TF) == RAW
            assert not handle_task.done()
            allow_start.set()
            await asyncio.wait_for(after_start.wait(), 2)
            assert cache.get_transformation_log(TF) is None
            assert not handle_task.done(), 'started must not complete the proxy future'
            assert token in worker._DELEGATE_PROXY_FUTURES
            allow_finish.set()
            assert await asyncio.wait_for(handle_task, 2) == RESULT
            assert cache.get_transformation_log(TF) == fresh
            # Keep polling alive for another request while a late old-token event arrives.
            worker._DELEGATE_PROXY_FUTURES['other-token'] = concurrent.futures.Future()
            await asyncio.wait_for(stale_seen.wait(), 2)
            allow_stale.set()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert cache.get_transformation_log(TF) == fresh
        finally:
            poller.cancel()
            handle_task.cancel()
            await asyncio.gather(poller, handle_task, return_exceptions=True)
    asyncio.run(run())


@pytest.mark.parametrize('started', [False, True])
@pytest.mark.parametrize('terminal', ['error', 'cancelled'])
def test_polled_failure_preserves_predispatch_log_but_clears_after_dispatch(nested_execution, monkeypatch, started, terminal):
    cache = nested_execution
    cache.set_transformation_log(TF, RAW)
    async def run():
        monkeypatch.setattr(worker, '_DELEGATE_PROXY_FUTURES', {})
        monkeypatch.setattr(worker, '_DELEGATE_PROXY_DISPATCH_CALLBACKS', {})
        monkeypatch.setattr(worker, '_ensure_delegate_poller', lambda: None)
        monkeypatch.setattr(worker, '_DELEGATE_POLL_INTERVAL', 0)
        async def request(kind, payload):
            if kind == 'delegate_transformation_submit':
                return {'status': 'submitted', 'token': 'failed-token'}
            assert kind == 'delegate_transformation_poll'
            return {'failed-token': {'status': terminal, 'error': 'parent dispatch failed',
                                     'dispatch_started': started}}
        monkeypatch.setattr(worker, '_request_parent_async', echo_poll_request_ids(request))
        poller = asyncio.create_task(worker._delegate_poll_loop())
        try:
            with pytest.raises(RuntimeError, match='cancelled' if terminal == 'cancelled' else 'parent dispatch failed'):
                await asyncio.wait_for(nested_uncached(cache), 2)
        finally:
            poller.cancel()
            await asyncio.gather(poller, return_exceptions=True)
    asyncio.run(run())
    assert cache.get_transformation_log(TF) == (None if started else RAW)


def test_inline_parent_start_notification_clears_before_submit_result(nested_execution, monkeypatch):
    cache = nested_execution
    cache.set_transformation_log(TF, RAW)
    async def run():
        request_ids = []
        entered, allow_start = asyncio.Event(), asyncio.Event()
        dispatched, allow_result = asyncio.Event(), asyncio.Event()
        async def request(kind, payload):
            assert kind == 'delegate_transformation_submit'
            entered.set()
            await allow_start.wait()
            request_id = payload['delegate_request_id']
            request_ids.append(request_id)
            worker._notify_nested_dispatch_started(request_id)
            dispatched.set()
            await allow_result.wait()
            return {'status': 'done', 'result': RESULT.hex(), 'logs': RAW, 'dispatch_started': True}
        monkeypatch.setattr(worker, '_request_parent_async', echo_poll_request_ids(request))
        task = asyncio.create_task(nested_uncached(cache))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            assert cache.get_transformation_log(TF) == RAW
            allow_start.set()
            await asyncio.wait_for(dispatched.wait(), 2)
            assert cache.get_transformation_log(TF) is None
            assert not task.done()
            allow_result.set()
            assert await asyncio.wait_for(task, 2) == RESULT
            assert cache.get_transformation_log(TF) == RAW
            worker._notify_nested_dispatch_started(request_ids[0])
            assert cache.get_transformation_log(TF) == RAW
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    asyncio.run(run())


@pytest.mark.parametrize('channel', ['endpoint', 'token'])
def test_old_dispatch_nonce_cannot_clear_new_same_checksum_run(cache, monkeypatch, channel):
    old_id, new_id = 'old-request-id', 'new-request-id'
    callbacks = []
    def old_started():
        callbacks.append('old')
        cache.clear_transformation_log(TF)
    def new_started():
        callbacks.append('new')
        cache.clear_transformation_log(TF)
    monkeypatch.setattr(worker, '_NESTED_DISPATCH_CALLBACKS', {})
    worker._register_nested_dispatch_callback(old_id, old_started)
    worker._unregister_nested_dispatch_callback(old_id, old_started)
    worker._register_nested_dispatch_callback(new_id, new_started)
    fresh = {'stdout': 'new same-checksum log', 'stderr': '', 'wall_time_seconds': 9.5}
    cache.set_transformation_log(TF, fresh)
    try:
        if channel == 'endpoint':
            worker._notify_nested_dispatch_started(old_id)
        else:
            async def run():
                monkeypatch.setattr(worker, '_DELEGATE_PROXY_FUTURES', {'old-token': concurrent.futures.Future()})
                monkeypatch.setattr(worker, '_DELEGATE_PROXY_DISPATCH_CALLBACKS', {'old-token': old_id})
                monkeypatch.setattr(worker, '_DELEGATE_POLL_INTERVAL', 0)
                received = asyncio.Event()
                async def request(kind, payload):
                    received.set()
                    return {'old-token': {'status': 'started', 'delegate_request_id': old_id}}
                monkeypatch.setattr(worker, '_request_parent_async', request)
                poller = asyncio.create_task(worker._delegate_poll_loop())
                try:
                    await asyncio.wait_for(received.wait(), 2)
                    await asyncio.sleep(0)
                finally:
                    poller.cancel()
                    await asyncio.gather(poller, return_exceptions=True)
            asyncio.run(run())
        assert callbacks == []
        assert cache.get_transformation_log(TF) == fresh
        worker._notify_nested_dispatch_started(new_id)
        assert callbacks == ['new']
        assert cache.get_transformation_log(TF) is None
    finally:
        worker._unregister_nested_dispatch_callback(new_id, new_started)


@pytest.mark.parametrize('failure', ['cancelled', 'buffer_writer'])
def test_cached_parent_held_log_is_not_fresh_dispatch_on_postforward_failure(nested_execution, monkeypatch, failure):
    cache = nested_execution
    cache.set_transformation_log(TF, RAW)
    active = SimpleNamespace(canceled=False)
    async def request(kind, payload):
        assert kind == 'delegate_transformation_submit'
        if failure == 'cancelled':
            active.canceled = True
        return {'status': 'done', 'result': RESULT.hex(), 'logs': RAW, 'dispatch_started': False}
    monkeypatch.setattr(worker, '_request_parent_async', echo_poll_request_ids(request))
    if failure == 'buffer_writer':
        async def flush(*args, **kwargs):
            raise RuntimeError('cached result buffer writer failed')
        monkeypatch.setattr(tc, '_await_buffer_writer', flush)
    from seamless_transformer.transformation_cache import TransformationCancelledError
    expected = TransformationCancelledError if failure == 'cancelled' else RuntimeError
    with pytest.raises(expected, match='canceled' if failure == 'cancelled' else 'cached result buffer writer failed'):
        asyncio.run(nested_uncached(cache, active))
    assert cache.get_transformation_log(TF) == RAW


@pytest.mark.parametrize('marker_source,started,outcome', [
    (source, started, outcome)
    for source in ('event', 'thin_result')
    for started in (False, True)
    for outcome in ('success', 'missing_result', 'error', 'cancelled')
    if not (source == 'thin_result' and started and outcome == 'cancelled')
])
def test_parent_started_future_marker_survives_without_log(cache, monkeypatch, outcome, started, marker_source):
    manager = object.__new__(worker._WorkerManager)
    result = None if outcome == 'missing_result' else RESULT.hex()
    error = 'fresh dispatch error' if outcome == 'error' else None
    thin = SimpleNamespace(cancelled=lambda: outcome == 'cancelled', done=lambda: True,
                           result=lambda: (TF.hex(), result, error, {'_seamless_dispatch_started': True} if started and marker_source == 'thin_result' else None))
    entry = {'futures': SimpleNamespace(thin=thin, _seamless_dispatch_started=started and marker_source == 'event'),
             'delegate_request_id': 'marker-request'}
    async def get(token):
        return entry
    async def pop(token, **kwargs):
        return entry
    monkeypatch.setattr(manager, '_get_delegate_entry', get)
    monkeypatch.setattr(manager, '_pop_delegate_entry', pop)
    monkeypatch.setattr(manager, '_release_delegate_entry', lambda *args, **kwargs: None)
    completed = manager._delegate_payload_from_entry(entry)
    polled = asyncio.run(manager._handle_delegate_transformation_poll(None, {'tokens': ['token']}))['token']
    for response in (completed, polled):
        assert response['dispatch_started'] is started
        expected = 'done' if outcome == 'success' else 'cancelled' if outcome == 'cancelled' else 'error'
        assert response['status'] == expected
        assert response.get('logs') is None or response['logs'] == {'_seamless_dispatch_started': True}


def test_fresh_polled_success_without_stream_log_clears_previous_entry(nested_execution, monkeypatch):
    cache = nested_execution
    cache.set_transformation_log(TF, RAW)
    async def run():
        monkeypatch.setattr(worker, '_DELEGATE_PROXY_FUTURES', {})
        monkeypatch.setattr(worker, '_DELEGATE_PROXY_DISPATCH_CALLBACKS', {})
        monkeypatch.setattr(worker, '_ensure_delegate_poller', lambda: None)
        monkeypatch.setattr(worker, '_DELEGATE_POLL_INTERVAL', 0)
        async def request(kind, payload):
            if kind == 'delegate_transformation_submit':
                return {'status': 'submitted', 'token': 'fresh-no-log-token'}
            return {'fresh-no-log-token': {'status': 'done', 'result': RESULT.hex(),
                                          'logs': None, 'dispatch_started': True}}
        monkeypatch.setattr(worker, '_request_parent_async', echo_poll_request_ids(request))
        poller = asyncio.create_task(worker._delegate_poll_loop())
        try:
            assert await asyncio.wait_for(nested_uncached(cache), 2) == RESULT
        finally:
            poller.cancel()
            await asyncio.gather(poller, return_exceptions=True)
    asyncio.run(run())
    assert cache.get_transformation_log(TF) is None


def test_same_token_poll_with_wrong_nonce_cannot_finish_or_clear_new_request(nested_execution, monkeypatch):
    cache = nested_execution
    cache.set_transformation_log(TF, RAW)
    async def run():
        monkeypatch.setattr(worker, '_DELEGATE_PROXY_FUTURES', {})
        monkeypatch.setattr(worker, '_DELEGATE_PROXY_DISPATCH_CALLBACKS', {})
        monkeypatch.setattr(worker, '_ensure_delegate_poller', lambda: None)
        monkeypatch.setattr(worker, '_DELEGATE_POLL_INTERVAL', 0)
        stale_processed, allow_valid = asyncio.Event(), asyncio.Event()
        request_id = None
        count = 0
        async def request(kind, payload):
            nonlocal request_id, count
            if kind == 'delegate_transformation_submit':
                request_id = payload['delegate_request_id']
                return {'status': 'submitted', 'token': 'reused-token'}
            assert payload['delegate_request_ids']['reused-token'] == request_id
            count += 1
            if count == 1:
                return {'reused-token': {'status': 'done', 'result': RESULT.hex(),
                                        'dispatch_started': True, 'logs': None,
                                        'delegate_request_id': 'old-unrelated-nonce'}}
            stale_processed.set()
            await allow_valid.wait()
            return {'reused-token': {'status': 'done', 'result': RESULT.hex(),
                                    'dispatch_started': False, 'delegate_request_id': request_id}}
        monkeypatch.setattr(worker, '_request_parent_async', request)
        poller = asyncio.create_task(worker._delegate_poll_loop())
        task = asyncio.create_task(nested_uncached(cache))
        try:
            await asyncio.wait_for(stale_processed.wait(), 2)
            assert not task.done()
            assert 'reused-token' in worker._DELEGATE_PROXY_FUTURES
            assert cache.get_transformation_log(TF) == RAW
            allow_valid.set()
            assert await asyncio.wait_for(task, 2) == RESULT
            assert cache.get_transformation_log(TF) == RAW
        finally:
            poller.cancel()
            task.cancel()
            await asyncio.gather(poller, task, return_exceptions=True)
    asyncio.run(run())


def test_old_inflight_poll_snapshot_cannot_consume_rebound_token(cache, monkeypatch):
    cache.set_transformation_log(TF, RAW)
    old_future = concurrent.futures.Future()
    new_future = concurrent.futures.Future()
    monkeypatch.setattr(worker, '_DELEGATE_PROXY_FUTURES', {'token': old_future})
    monkeypatch.setattr(worker, '_DELEGATE_PROXY_DISPATCH_CALLBACKS', {'token': 'old-nonce'})
    monkeypatch.setattr(worker, '_DELEGATE_POLL_INTERVAL', 0)
    async def run():
        sent, allow_old_response = asyncio.Event(), asyncio.Event()
        rebound_poll = asyncio.Event()
        calls = 0
        async def request(kind, payload):
            nonlocal calls
            assert kind == 'delegate_transformation_poll'
            calls += 1
            if calls == 1:
                assert payload['delegate_request_ids']['token'] == 'old-nonce'
                sent.set()
                await allow_old_response.wait()
                return {'token': {'status': 'done', 'result': RESULT.hex(), 'dispatch_started': True,
                                  'delegate_request_id': 'old-nonce'}}
            assert payload['delegate_request_ids']['token'] == 'new-nonce'
            rebound_poll.set()
            return {'token': {'status': 'done', 'result': RESULT.hex(), 'dispatch_started': False,
                              'delegate_request_id': 'new-nonce'}}
        monkeypatch.setattr(worker, '_request_parent_async', request)
        poller = asyncio.create_task(worker._delegate_poll_loop())
        try:
            await asyncio.wait_for(sent.wait(), 2)
            worker._DELEGATE_PROXY_FUTURES['token'] = new_future
            worker._DELEGATE_PROXY_DISPATCH_CALLBACKS['token'] = 'new-nonce'
            allow_old_response.set()
            await asyncio.wait_for(rebound_poll.wait(), 2)
            assert not old_future.done()
            assert await asyncio.wait_for(asyncio.wrap_future(new_future), 2) == RESULT
            assert cache.get_transformation_log(TF) == RAW
        finally:
            poller.cancel()
            await asyncio.gather(poller, return_exceptions=True)
    asyncio.run(run())
