"""Fixtures for the cancellation suite.

Inherits the seamless open/close fixtures from ``tests/conftest.py`` (parent).
"""

import pytest

from _harness import build_cluster


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "remote: multi-tenant test requiring a local cluster (services)."
    )


# --------------------------------------------------------------------------- #
# In-process isolation: a fresh cache with every remote backend disabled, so the
# membership-set behaviour is exercised purely in-process (mirrors the existing
# tests/test_transformation_cache_active.py::no_remote_cache fixture).
# Requested explicitly by the in-process test files; remote tests do not use it.
# --------------------------------------------------------------------------- #
@pytest.fixture
def inproc_cache(monkeypatch):
    from seamless_transformer import transformation_cache

    monkeypatch.setattr(transformation_cache, "database_remote", None)
    monkeypatch.setattr(transformation_cache, "buffer_remote", None)
    monkeypatch.setattr(transformation_cache, "jobserver_remote", None)
    monkeypatch.setattr(transformation_cache, "get_execution", lambda: "process")
    monkeypatch.setattr(transformation_cache, "is_worker", lambda: False)
    return transformation_cache.TransformationCache()


# --------------------------------------------------------------------------- #
# Remote multi-tenant clusters (module-scoped: start services once per file).
# Distinct port ranges so a jobserver and a dask file can run back to back.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def jobserver_cluster(tmp_path_factory):
    root = tmp_path_factory.mktemp("mt_jobserver")
    with build_cluster(
        root, remote_kind="jobserver",
        cluster_name="cancel-mt-jobserver", port_base=61600,
    ) as cluster:
        yield cluster


@pytest.fixture(scope="module")
def dask_cluster(tmp_path_factory):
    root = tmp_path_factory.mktemp("mt_dask")
    with build_cluster(
        root, remote_kind="daskserver",
        cluster_name="cancel-mt-dask", port_base=61800,
    ) as cluster:
        yield cluster
