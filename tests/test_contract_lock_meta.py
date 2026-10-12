"""Boolean lock metadata defaults, coupling and immutable build snapshots."""

import pytest

from seamless.transformer import delayed
from seamless_transformer.transformation_utils import resolve_lock_meta


def _identity(value: int):
    return value


@pytest.mark.parametrize("meta, expected", [
    ({}, (True, False)),
    ({"driver": False}, (True, False)),
    ({"driver": True}, (False, True)),
    ({"process_lock": True}, (True, False)),
    ({"process_lock": False}, (False, False)),
    ({"driver": False, "process_lock": False}, (False, False)),
    ({"driver": True, "process_lock": False}, (False, True)),
])
def test_resolve_lock_meta_defaults_and_legal_states(meta, expected):
    original = meta.copy()
    assert resolve_lock_meta(meta) == expected
    assert meta == original


@pytest.mark.parametrize("meta", [
    {"driver": True, "process_lock": True},
    *[{key: value} for key in ("driver", "process_lock") for value in (1, 0, "true", None, [], {})],
])
def test_resolve_lock_meta_rejects_invalid_values(meta):
    with pytest.raises((TypeError, ValueError)):
        resolve_lock_meta(meta)


@pytest.mark.parametrize("route", ["property", "meta"])
def test_driver_coupling_and_explicit_reenable(route):
    builder = delayed(_identity)
    assert builder.driver is False
    assert builder.process_lock is True
    assert "driver" not in builder.meta
    assert "process_lock" not in builder.meta

    def update(key, value):
        if route == "property":
            setattr(builder, key, value)
        else:
            builder.meta = {key: value}

    update("driver", True)
    assert (builder.process_lock, builder.driver) == (False, True)
    before = dict(builder.meta)
    with pytest.raises((TypeError, ValueError)):
        update("process_lock", True)
    assert dict(builder.meta) == before
    update("driver", False)
    assert (builder.process_lock, builder.driver) == (False, False)
    update("process_lock", True)
    assert (builder.process_lock, builder.driver) == (True, False)
    update("process_lock", False)
    assert (builder.process_lock, builder.driver) == (False, False)
    update("driver", True)
    update("driver", None)
    assert (builder.process_lock, builder.driver) == (False, False)


def test_meta_combined_updates_are_validated_atomically():
    builder = delayed(_identity)
    builder.meta = {"ncores": 2}
    before = dict(builder.meta)
    with pytest.raises((TypeError, ValueError)):
        builder.meta = {"driver": True, "process_lock": True, "ncores": 8}
    assert dict(builder.meta) == before
    builder.meta = {"driver": True, "process_lock": False}
    assert (builder.process_lock, builder.driver) == (False, True)
    builder.meta = {"driver": False, "process_lock": True}
    assert (builder.process_lock, builder.driver) == (True, False)
    assert builder.meta["ncores"] == 2


@pytest.mark.parametrize("route", ["property", "meta"])
@pytest.mark.parametrize("key", ["driver", "process_lock"])
@pytest.mark.parametrize("value", [1, 0, "true", [], {}])
def test_builder_rejects_nonboolean_lock_values(route, key, value):
    builder = delayed(_identity)
    before = dict(builder.meta)
    with pytest.raises((TypeError, ValueError)):
        if route == "property":
            setattr(builder, key, value)
        else:
            builder.meta = {key: value}
    assert dict(builder.meta) == before


@pytest.mark.parametrize("settings, expected", [
    ({}, (True, False)),
    ({"driver": True}, (False, True)),
    ({"process_lock": False}, (False, False)),
    ({"driver": False, "process_lock": True}, (True, False)),
])
def test_build_writes_both_flags_and_keeps_immutable_snapshot(settings, expected):
    builder = delayed(_identity)
    builder.meta = settings
    handle = builder.build(17)
    handle.construct()
    assert handle.exception is None
    assert handle.meta["process_lock"] is expected[0]
    assert handle.meta["driver"] is expected[1]
    with pytest.raises(TypeError):
        handle.meta["process_lock"] = not expected[0]
    with pytest.raises(Exception, match="immutable"):
        handle.meta = {"driver": False, "process_lock": True}
    builder.meta = {"driver": False, "process_lock": not expected[0]}
    assert handle.meta["process_lock"] is expected[0]
    assert handle.meta["driver"] is expected[1]


def test_lock_metadata_does_not_change_transformation_identity():
    builder = delayed(_identity)
    default = builder.build(19)
    builder.driver = True
    driver = builder.build(19)
    builder.driver = False
    unlocked = builder.build(19)
    assert default.construct() == driver.construct() == unlocked.construct()
    assert [handle.meta["process_lock"] for handle in (default, driver, unlocked)] == [True, False, False]
