"""Phase D public logs surface; capture/transport is covered by Phase C."""
import re

import pytest
from seamless import Checksum
from seamless_transformer import delayed, has_spawned, spawn
from seamless_transformer.transformation_class import Transformation
import seamless_transformer.transformation_cache as tc

STARS = '*' * 49


def forbidden(*args, **kwargs):
    raise AssertionError('logs must neither construct nor execute nor resolve a result')


def handle(checksum=None):
    result = Transformation('mixed', forbidden, forbidden, forbidden, forbidden)
    result._transformation_checksum = checksum
    return result


@pytest.fixture
def log_store(monkeypatch):
    cache = tc.TransformationCache(log_cache_size=2)
    monkeypatch.setattr(tc, 'get_transformation_cache', lambda: cache)
    return cache


@pytest.mark.parametrize('stdout,stderr,expected', [
    ('', '', f'{STARS}\nExecution time: 2.4 seconds\n{STARS}'),
    ('hello\n', '', f'{STARS}\n* Standard output\n{STARS}\nhello\n\n{STARS}\nExecution time: 2.4 seconds\n{STARS}'),
    ('', 'warning', f'{STARS}\n* Standard error\n{STARS}\nwarning\n{STARS}\nExecution time: 2.4 seconds\n{STARS}'),
    ('out', 'err\n', f'{STARS}\n* Standard output\n{STARS}\nout\n{STARS}\n* Standard error\n{STARS}\nerr\n\n{STARS}\nExecution time: 2.4 seconds\n{STARS}'),
])
def test_exact_success_layout(log_store, stdout, stderr, expected):
    checksum = Checksum('1' * 64)
    log_store.set_transformation_log(checksum, dict(stdout=stdout, stderr=stderr, wall_time_seconds=2.375))
    transformation = handle(checksum)
    assert transformation.logs == expected
    assert not transformation.logs.endswith('\n')
    assert 'Result' not in transformation.logs
    assert not transformation._constructed and not transformation._evaluated
    with pytest.raises(AttributeError):
        transformation.logs = 'replacement'


def test_capture_truncation_marker_survives_formatting(log_store):
    checksum = Checksum('2' * 64)
    # Transport has already truncated streams at the capture boundary.
    stdout = 'α' * 4960 + '\n...(skipped 120 characters)...\n' + '尾' * 5000
    stderr = 'β' * 4960 + '\n...(skipped 41 characters)...\n' + '終' * 5000
    log_store.set_transformation_log(checksum, dict(stdout=stdout, stderr=stderr, wall_time_seconds=0))
    assert handle(checksum).logs == (f'{STARS}\n* Standard output\n{STARS}\n{stdout}\n'
                                   f'{STARS}\n* Standard error\n{STARS}\n{stderr}\n'
                                   f'{STARS}\nExecution time: 0.0 seconds\n{STARS}')


def test_passive_unknown_identity_and_cache_without_log(log_store):
    assert handle().logs is None
    transformation = handle(Checksum('3' * 64))
    transformation._evaluated = True  # a result cache hit does not imply execution
    transformation._result_checksum = Checksum('4' * 64)
    assert transformation.logs is None


def test_shared_identity_and_lru_eviction_are_visible_to_all_handles(log_store):
    first, second, third = [Checksum(c * 64) for c in '567']
    a, b = handle(first), handle(first)
    entry = dict(stdout='shared', stderr='', wall_time_seconds=1)
    log_store.set_transformation_log(first, entry)
    assert a.logs == b.logs
    log_store.set_transformation_log(second, entry)
    assert a.logs is not None  # refresh first through the public accessor
    log_store.set_transformation_log(third, entry)
    assert handle(second).logs is None
    assert a.logs == b.logs
    log_store.clear_transformation_log(first)
    assert a.logs is b.logs is None  # fresh dispatch clears the shared log


def test_failure_returns_exception_and_cancellation_returns_none(log_store):
    checksum = Checksum('8' * 64)
    log_store.set_transformation_log(checksum, dict(stdout='old', stderr='', wall_time_seconds=1))
    transformation = handle(checksum)
    transformation._exception = 'failure\n' + 'full-output' * 1100
    assert transformation.logs == transformation.exception
    assert 'skipped' not in transformation.logs
    transformation._cancelled = True
    assert transformation.logs is None


def test_real_spawn_success_and_failure_public_logs():
    if not has_spawned():
        spawn(1)

    @delayed
    def succeed(value):
        import sys
        print('SURFACE_OUT', value)
        print('SURFACE_ERR', file=sys.stderr)
        return value + 1

    @delayed
    def fail(value):
        import sys
        print('SURFACE_FAIL_OUT')
        print('SURFACE_FAIL_ERR', file=sys.stderr)
        raise ValueError('surface-failure-' + str(value))

    success = succeed(8123)
    assert success.logs is None
    assert success.run() == 8124
    assert f'* Standard output\n{STARS}\nSURFACE_OUT 8123\n\n' in success.logs
    assert f'* Standard error\n{STARS}\nSURFACE_ERR\n\n' in success.logs
    assert re.search(r'Execution time: \d+\.\d seconds\n\*{49}$', success.logs)
    second = succeed(8123)
    assert second.logs is None  # identity is still unknown
    second.construct()
    assert second.logs == success.logs  # construction alone reads the shared entry
    failure = fail(8123)
    assert failure.compute() is None
    assert failure.logs == failure.exception
    assert 'surface-failure-8123' in failure.logs
    assert 'SURFACE_FAIL_OUT' in failure.logs and 'SURFACE_FAIL_ERR' in failure.logs
    assert 'Execution time:' not in failure.logs
