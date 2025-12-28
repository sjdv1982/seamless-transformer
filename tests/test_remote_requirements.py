from seamless.transformer import delayed

import seamless_transformer.transformation_cache as transformation_cache
import seamless_transformer.transformation_class as transformation_class


def test_remote_requires_storage(monkeypatch):
    @delayed
    def add(a, b):
        return a + b

    tf = add(1, 2)
    monkeypatch.setattr(transformation_cache, "get_execution", lambda: "remote")
    monkeypatch.setattr(transformation_class, "_dask_available", lambda: False)
    monkeypatch.setattr(transformation_cache, "jobserver_remote", object())
    monkeypatch.setattr(transformation_cache, "buffer_remote", None)
    monkeypatch.setattr(transformation_cache, "database_remote", None)

    tf.compute()
    assert tf.exception is not None
    assert "hashserver and database server" in tf.exception


def test_local_bypasses_remote_storage(monkeypatch):
    @delayed
    def add(a, b):
        return a + b

    add.local = True
    tf = add(1, 2)
    monkeypatch.setattr(transformation_cache, "get_execution", lambda: "remote")
    monkeypatch.setattr(transformation_class, "_dask_available", lambda: False)
    monkeypatch.setattr(transformation_cache, "jobserver_remote", object())
    monkeypatch.setattr(transformation_cache, "buffer_remote", None)
    monkeypatch.setattr(transformation_cache, "database_remote", None)

    assert tf.run() == 3
    assert tf.exception is None
