"""Focused tests for the shared insert-only transformation recording path."""
import asyncio
from types import SimpleNamespace

import pytest
from seamless import CacheMissError, Checksum
from seamless_remote.database_client import TransformationResultConflict
import seamless_transformer.transformation_cache as cache_mod


def test_new_same_and_mismatch_share_one_recording_path(monkeypatch):
    cache = cache_mod.TransformationCache()
    tf, recorded, observed = (Checksum(char * 64) for char in "abc")
    puts, records, reports = [], [], []
    async def put(t, r):
        puts.append((t, r))
    async def record():
        records.append(True)
    async def report(t, r):
        reports.append((t, r))
        return True
    monkeypatch.setattr(cache_mod, "database_remote", SimpleNamespace(set_transformation_result=put, report_irreproducible_result=report))
    async def run():
        assert await cache._record_transformation_result(tf, recorded, execution_record=record) == "NEW"
        assert await cache._record_transformation_result(tf, recorded, execution_record=record) == "SAME"
        assert await cache._record_transformation_result(tf, observed, execution_record=record) == "MISMATCH"
    asyncio.run(run())
    assert puts == [(tf, recorded)] and records == [True] and reports == [(tf, observed)]
    assert cache._transformation_cache[tf] == recorded
    assert cache.get_reverse_transformations(observed) == []


def test_database_race_learns_existing_result_and_reports_divergence(monkeypatch):
    cache = cache_mod.TransformationCache()
    tf, recorded, observed = (Checksum(char * 64) for char in "def")
    reports = []
    async def conflict(*args):
        raise TransformationResultConflict()
    async def get(t):
        return recorded
    async def report(t, r):
        reports.append((t, r))
        return True
    async def record():
        raise AssertionError("mismatch wrote execution record")
    monkeypatch.setattr(cache_mod, "database_remote", SimpleNamespace(set_transformation_result=conflict, get_transformation_result=get, report_irreproducible_result=report))
    assert asyncio.run(cache._record_transformation_result(tf, observed, execution_record=record)) == "MISMATCH"
    assert cache._transformation_cache[tf] == recorded
    assert cache.get_reverse_transformations(recorded) == [tf]
    assert cache.get_reverse_transformations(observed) == []
    assert reports == [(tf, observed)]


def test_database_hit_is_learned_before_unreachable_result_is_discarded(monkeypatch):
    cache = cache_mod.TransformationCache()
    tf, recorded = Checksum("1" * 64), Checksum("2" * 64)
    async def get(t):
        return recorded
    async def resolution(checksum, *args, **kwargs):
        raise CacheMissError(checksum)
    async def recompute(*args, **kwargs):
        assert cache._transformation_cache[tf] == recorded
        return recorded
    monkeypatch.setattr(cache_mod, "database_remote", SimpleNamespace(get_transformation_result=get))
    monkeypatch.setattr(Checksum, "resolution", resolution)
    monkeypatch.setattr(cache, "_run_active_or_execute", recompute)
    assert asyncio.run(cache.run({}, tf_checksum=tf, tf_dunder={}, scratch=True, require_value=True)) == recorded


@pytest.mark.parametrize("exception", [RuntimeError, TimeoutError])
def test_recompute_does_not_hide_non_materialization_errors(monkeypatch, exception):
    async def resolution(*args, **kwargs):
        raise exception("unavailable")
    def resolve(*args, **kwargs):
        raise exception("unavailable")
    monkeypatch.setattr(Checksum, "resolution", resolution)
    monkeypatch.setattr(Checksum, "resolve", resolve)
    with pytest.raises(exception):
        asyncio.run(cache_mod.recompute_from_transformation_checksum("3" * 64))
    with pytest.raises(exception):
        cache_mod.recompute_from_transformation_checksum_sync("3" * 64)
