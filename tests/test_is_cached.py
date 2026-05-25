import asyncio

import pytest

from seamless import Checksum
from seamless.transformer import delayed
import seamless_transformer.transformation_cache as transformation_cache


class _FakeDatabaseRemote:
    def __init__(self, result):
        self.result = result
        self.queries = []

    def has_read_server(self):
        return True

    async def get_transformation_result(self, tf_checksum):
        self.queries.append(Checksum(tf_checksum))
        return self.result


class _NoReadDatabaseRemote:
    def has_read_server(self):
        return False


@delayed
def _add(a, b):
    return a + b


def test_transformation_is_cached_true(monkeypatch):
    fake_database = _FakeDatabaseRemote(Checksum("1" * 64))
    monkeypatch.setattr(transformation_cache, "database_remote", fake_database)
    monkeypatch.setattr(transformation_cache, "_close_all_clients", None)

    tf = _add(1, 2)

    assert tf.is_cached() is True
    assert fake_database.queries == [tf.transformation_checksum]


def test_transformation_is_cached_false(monkeypatch):
    fake_database = _FakeDatabaseRemote(None)
    monkeypatch.setattr(transformation_cache, "database_remote", fake_database)
    monkeypatch.setattr(transformation_cache, "_close_all_clients", None)

    tf = _add(1, 2)

    assert tf.is_cached() is False
    assert fake_database.queries == [tf.transformation_checksum]


def test_transformation_is_cached_requires_database_init(monkeypatch):
    monkeypatch.setattr(
        transformation_cache, "database_remote", _NoReadDatabaseRemote()
    )
    monkeypatch.setattr(transformation_cache, "_close_all_clients", None)

    tf = _add(1, 2)

    with pytest.raises(RuntimeError, match="seamless.config.init"):
        tf.is_cached()


def test_transformation_is_cached_async(monkeypatch):
    fake_database = _FakeDatabaseRemote(Checksum("1" * 64))
    monkeypatch.setattr(transformation_cache, "database_remote", fake_database)

    tf = _add(1, 2)

    assert asyncio.run(tf.is_cached_async()) is True
    assert fake_database.queries == [tf.transformation_checksum]
