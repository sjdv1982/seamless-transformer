"""Phase E: caller-process tees retain only transformation-thread output."""
import io
import sys
import threading

import pytest
from seamless_transformer import delayed
import seamless_transformer.transformation_cache as tc

STARS = '*' * 49


@pytest.fixture
def process_mode(monkeypatch):
    monkeypatch.setattr(tc, 'get_execution', lambda: 'process')
    monkeypatch.setattr(tc.worker, 'has_spawned', lambda: False)
    monkeypatch.setattr(tc, 'is_worker', lambda: False)


def caller_streams(monkeypatch):
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, 'stdout', out)
    monkeypatch.setattr(sys, 'stderr', err)
    return out, err


def test_real_process_success_is_live_and_retained(process_mode, monkeypatch):
    @delayed
    def succeed(value):
        import sys
        print('PROCESS_OUT', value)
        print('PROCESS_ERR', file=sys.stderr)
        return value + 1

    out, err = caller_streams(monkeypatch)
    original_start = threading.Thread.start
    handle = succeed(18271)
    assert handle.run() == 18272
    assert 'PROCESS_OUT 18271\n' in out.getvalue()
    assert 'PROCESS_ERR\n' in err.getvalue()
    assert f'* Standard output\n{STARS}\nPROCESS_OUT 18271\n\n' in handle.logs
    assert f'* Standard error\n{STARS}\nPROCESS_ERR\n\n' in handle.logs
    assert 'Execution time:' in handle.logs
    assert sys.stdout is out and sys.stderr is err
    assert threading.Thread.start is original_start


def test_real_process_failure_has_full_live_trailers(process_mode, monkeypatch):
    @delayed
    def fail(value):
        import sys
        print('OUT_BEGIN' + 'x' * 12001 + 'OUT_END')
        print('ERR_BEGIN' + 'y' * 12001 + 'ERR_END', file=sys.stderr)
        raise ValueError('process-failure-' + str(value))

    out, err = caller_streams(monkeypatch)
    handle = fail(18272)
    assert handle.compute() is None
    assert handle.logs == handle.exception
    assert 'process-failure-18272' in handle.logs
    assert f'* Standard output\n{STARS}\n' + out.getvalue() in handle.logs
    assert f'* Standard error\n{STARS}\n' + err.getvalue() in handle.logs
    assert '...(skipped ' not in handle.logs
    assert 'Execution time:' not in handle.logs
    assert sys.stdout is out and sys.stderr is err


def test_real_process_user_thread_output_is_live_only(process_mode, monkeypatch):
    @delayed
    def threaded(value):
        import sys
        import threading
        def user_thread():
            print('USER_THREAD_OUT')
            print('USER_THREAD_ERR', file=sys.stderr)
        thread = threading.Thread(target=user_thread)
        thread.start()
        thread.join()
        print('OWN_THREAD_OUT')
        print('OWN_THREAD_ERR', file=sys.stderr)
        return value

    out, err = caller_streams(monkeypatch)
    original_start = threading.Thread.start
    handle = threaded(18273)
    assert handle.run() == 18273
    assert 'USER_THREAD_OUT' in out.getvalue()
    assert 'USER_THREAD_ERR' in err.getvalue()
    assert 'OWN_THREAD_OUT' in handle.logs and 'OWN_THREAD_ERR' in handle.logs
    assert 'USER_THREAD_OUT' not in handle.logs and 'USER_THREAD_ERR' not in handle.logs
    assert threading.Thread.start is original_start
    assert sys.stdout is out and sys.stderr is err


def test_cancelled_process_task_eventually_restores_caller_streams(process_mode, monkeypatch):
    import asyncio

    @delayed
    def slow(value):
        import time
        print('CANCEL_CAPTURE_STARTED')
        time.sleep(0.3)
        print('CANCEL_CAPTURE_FINISHED')
        return value

    out, err = caller_streams(monkeypatch)
    handle = slow(18274)

    async def cancel_after_start():
        task = handle.task()
        for _ in range(500):
            if 'CANCEL_CAPTURE_STARTED' in out.getvalue():
                break
            await asyncio.sleep(0.01)
        else:
            task.cancel()
            raise AssertionError('process execution did not start')
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert handle.logs is None

    # asyncio.run waits for its executor to finish, even after awaiter cancellation.
    asyncio.run(cancel_after_start())
    assert 'CANCEL_CAPTURE_FINISHED' in out.getvalue()
    assert handle.logs is None
    assert sys.stdout is out and sys.stderr is err


def test_overlapping_process_routes_isolate_output_and_restore_last_exit(monkeypatch):
    from seamless_transformer.output_routing import process_output_route

    out, err = caller_streams(monkeypatch)
    original_start = threading.Thread.start
    entered = threading.Barrier(3)
    first_exit = threading.Event()
    second_exit = threading.Event()
    captures, errors = {}, []

    def run(name, release):
        try:
            with process_output_route() as streams:
                captures[name] = streams
                print(name + '_OUT')
                print(name + '_ERR', file=sys.stderr)
                entered.wait(timeout=5)
                assert release.wait(timeout=5)
                print(name + '_END')
        except BaseException as exc:
            errors.append(exc)

    a = threading.Thread(target=run, args=('FIRST', first_exit))
    b = threading.Thread(target=run, args=('SECOND', second_exit))
    a.start()
    b.start()
    try:
        entered.wait(timeout=5)
        installed_out, installed_err = sys.stdout, sys.stderr
        assert installed_out is not out and installed_err is not err
        print('UNROUTED_CALLER_OUT')
        print('UNROUTED_CALLER_ERR', file=sys.stderr)
        first_exit.set()
        a.join(timeout=5)
        assert not a.is_alive()
        assert sys.stdout is installed_out and sys.stderr is installed_err
    finally:
        first_exit.set()
        second_exit.set()
        a.join(timeout=5)
        b.join(timeout=5)
    assert not a.is_alive() and not b.is_alive()
    assert not errors
    for name, other in [('FIRST', 'SECOND'), ('SECOND', 'FIRST')]:
        captured_out, captured_err = captures[name]
        assert captured_out.getvalue() == name + '_OUT\n' + name + '_END\n'
        assert captured_err.getvalue() == name + '_ERR\n'
        assert other not in captured_out.getvalue() + captured_err.getvalue()
    assert 'UNROUTED_CALLER_OUT' in out.getvalue()
    assert 'UNROUTED_CALLER_ERR' in err.getvalue()
    assert sys.stdout is out and sys.stderr is err
    assert threading.Thread.start is original_start


@pytest.mark.parametrize('stream_name', ['stdout', 'stderr'])
def test_user_stream_replacement_survives_route_release(monkeypatch, stream_name):
    from seamless_transformer.output_routing import process_output_route

    out, err = caller_streams(monkeypatch)
    replacement = io.StringIO()
    with process_output_route():
        setattr(sys, stream_name, replacement)
        print('REPLACEMENT_WRITE', file=getattr(sys, stream_name))
    assert getattr(sys, stream_name) is replacement
    other_name, original = ('stderr', err) if stream_name == 'stdout' else ('stdout', out)
    assert getattr(sys, other_name) is original
    assert replacement.getvalue() == 'REPLACEMENT_WRITE\n'


def test_process_route_exception_restores_original_objects(monkeypatch):
    from seamless_transformer.output_routing import process_output_route

    out, err = caller_streams(monkeypatch)
    with pytest.raises(RuntimeError, match='capture cleanup'):
        with process_output_route() as (captured_out, captured_err):
            print('BEFORE_EXCEPTION')
            print('BEFORE_EXCEPTION_ERR', file=sys.stderr)
            raise RuntimeError('capture cleanup')
    assert sys.stdout is out and sys.stderr is err
    assert captured_out.getvalue() == out.getvalue() == 'BEFORE_EXCEPTION\n'
    assert captured_err.getvalue() == err.getvalue() == 'BEFORE_EXCEPTION_ERR\n'


@pytest.mark.parametrize('payload,retained', [
    (b'\xff\x00abc', '\ufffd\x00abc'),
    ('é'.encode('utf-8'), 'é'),
])
def test_process_binary_tee_preserves_live_bytes_and_write_count(monkeypatch, payload, retained):
    from seamless_transformer.output_routing import process_output_route

    raw = io.BytesIO()
    out = io.TextIOWrapper(raw, encoding='utf-8', errors='strict')
    err = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', out)
    monkeypatch.setattr(sys, 'stderr', err)
    with process_output_route() as (captured_out, captured_err):
        assert sys.stdout.buffer.write(payload) == len(payload)
        sys.stdout.flush()
    assert raw.getvalue() == payload
    assert captured_out.getvalue() == retained
    assert captured_err.getvalue() == ''
    assert sys.stdout is out and sys.stderr is err


def test_process_failure_with_cancellation_message_keeps_output(process_mode, monkeypatch):
    @delayed
    def fail(value):
        print('FAILED_CANCELLATION_MESSAGE_MARKER')
        raise RuntimeError('Transformation was canceled')

    out, err = caller_streams(monkeypatch)
    handle = fail(18275)
    assert handle.compute() is None
    assert handle.logs == handle.exception
    assert 'RuntimeError: Transformation was canceled' in handle.exception
    assert f'* Standard output\n{STARS}\nFAILED_CANCELLATION_MESSAGE_MARKER\n' in handle.exception
    assert 'Execution time:' not in handle.logs
    assert 'FAILED_CANCELLATION_MESSAGE_MARKER\n' in out.getvalue()
    assert sys.stdout is out and sys.stderr is err
