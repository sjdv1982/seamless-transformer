from copy import deepcopy
import asyncio
from types import MappingProxyType

import pytest

from seamless import Buffer
from seamless.transformer import delayed
from seamless_transformer.transformation_class import TransformationError
from seamless_transformer.transformation_utils import tf_get_buffer


def _checksum(transformation):
    return tf_get_buffer(transformation).get_checksum().hex()


def _base_transformation():
    code_checksum = Buffer("result = a + 1", "text").get_checksum().hex()
    value_checksum = Buffer(12, "mixed").get_checksum().hex()
    return {
        "__language__": "python",
        "__output__": ("result", "mixed", None),
        "code": ("text", "transformer", code_checksum),
        "a": ("mixed", None, value_checksum),
    }


@pytest.mark.parametrize(
    ("key", "value1", "value2"),
    [
        ("__language__", "python", "bash"),
        ("__output__", ("result", "mixed", None), ("result", "plain", None)),
        ("__as__", {"a": "a"}, {"a": "renamed"}),
        ("__format__", {"a": {"celltype": "mixed"}}, {"a": {"celltype": "plain"}}),
        ("__schema__", "1" * 64, "2" * 64),
    ],
)
def test_load_bearing_dunders_change_checksum(key, value1, value2):
    transformation1 = _base_transformation()
    transformation2 = deepcopy(transformation1)
    transformation1[key] = value1
    transformation2[key] = value2

    assert _checksum(transformation1) != _checksum(transformation2)


@pytest.mark.parametrize(
    ("key", "value1", "value2"),
    [
        ("__meta__", {"local": True}, {"local": False}),
        ("__env__", "1" * 64, "2" * 64),
        ("__compilation__", "1" * 64, "2" * 64),
        ("__record_probe__", {"mode": "capture"}, {"mode": "replay"}),
        ("__code_checksum__", "1" * 64, "2" * 64),
        ("__code_text__", "result = a + 1", "result = a + 2"),
        ("__compilers__", {"c": "gcc"}, {"c": "clang"}),
        ("__languages__", {"c": {}}, {"cpp": {}}),
        ("__compiled__", True, False),
        ("__header__", "1" * 64, "2" * 64),
        ("META__THREADS", ("int", None, "1" * 64), ("int", None, "2" * 64)),
    ],
)
def test_orthogonal_and_derived_dunders_do_not_change_checksum(key, value1, value2):
    transformation1 = _base_transformation()
    transformation2 = deepcopy(transformation1)
    transformation1[key] = value1
    transformation2[key] = value2

    assert _checksum(transformation1) == _checksum(transformation2)


def test_transformation_meta_and_scratch_are_immutable():
    @delayed
    def func(a):
        return a + 1

    func.local = True
    tf = func(12)

    assert isinstance(tf.meta, MappingProxyType)
    with pytest.raises(TransformationError, match="immutable"):
        tf.meta = {"local": False}
    with pytest.raises(TypeError):
        tf.meta["local"] = False
    with pytest.raises(TransformationError, match="immutable"):
        tf.scratch = True


def test_transformation_copies_meta_before_return():
    @delayed
    def func(a):
        return a + 1

    func.meta = {"nested": {"value": 1}}
    tf = func(12)
    func.meta["nested"]["value"] = 2

    assert tf.meta["nested"]["value"] == 1
    with pytest.raises(TypeError):
        tf.meta["nested"]["value"] = 3


def test_cancel_started_transformation_is_terminal():
    @delayed
    def func(delay):
        import time

        time.sleep(delay)
        return 10

    func.local = True
    tf = func(1.0).start()

    assert tf.cancel() is True
    assert tf.status == "Status: canceled"
    assert tf.exception == "Transformation was canceled"
    with pytest.raises(TransformationError, match="Transformation was canceled"):
        _ = tf.result_checksum

    tf.clear_exception()
    assert tf.status == "Status: canceled"
    with pytest.raises(TransformationError, match="Transformation was canceled"):
        _ = tf.result_checksum
    assert tf.cancel() is False


def test_cancelled_task_marks_transformation_cancelled():
    @delayed
    def func(delay):
        import time

        time.sleep(delay)
        return 10

    func.local = True
    tf = func(1.0)

    async def main():
        task = tf.task()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    assert tf.status == "Status: canceled"
    with pytest.raises(TransformationError, match="Transformation was canceled"):
        _ = tf.result_checksum
