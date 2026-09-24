"""Contract tests for docs/agent/contracts/transformers.md (standalone builder).

Target repo location: seamless-transformer/tests/test_contract_transformer_builder.py

Pins behaviour itself is owned by pins.md; this file only pins the rules that
transformers.md states: decorators, the build/transformation/get_transformation
snapshot operation, snapshot isolation from later builder mutation, the named
work methods of a standalone Transformer, and bound-only attributes.
"""

import asyncio

import pytest

from seamless_transformer import Transformer, delayed, direct
from seamless_transformer.transformation_class import Transformation


def add(a, b):
    return a + b


def mul(a, b):
    return a * b


def length(x):
    return len(x)


def _builder(mode):
    tf = (direct if mode == "direct" else delayed)(add)
    tf.local = True
    return tf


# --- decorators --------------------------------------------------------------


def test_decorators_do_not_execute_the_function_at_decoration_time(tmp_path):
    marker = tmp_path / "executed"

    def side_effect():
        open(marker, "w").write("x")
        return 1

    d = delayed(side_effect)
    D = direct(side_effect)
    assert not marker.exists()
    assert type(d).__name__ == "PythonTransformer"
    assert type(D).__name__ == "DirectPythonTransformer"
    assert not isinstance(d, Transformation)
    assert not isinstance(D, Transformation)
    with pytest.raises(TypeError):
        direct(add, language="bash")  # pylint: disable=unexpected-keyword-arg
    with pytest.raises(TypeError):
        delayed(add, language="bash")  # pylint: disable=unexpected-keyword-arg


def test_delayed_call_builds_without_executing(tmp_path):
    marker = tmp_path / "executed"

    def side_effect():
        open(marker, "w").write("x")
        return 1

    tf = delayed(side_effect)
    tf.local = True
    tr = tf()
    assert isinstance(tr, Transformation)
    assert not marker.exists()


# --- build / transformation / get_transformation ----------------------------


@pytest.mark.parametrize("mode", ["delayed", "direct"])
def test_build_returns_transformation_for_every_call_mode(mode):
    tf = _builder(mode)
    tr = tf.build(2, 3)
    assert isinstance(tr, Transformation)
    assert tr.run() == 5


@pytest.mark.parametrize("mode", ["delayed", "direct"])
@pytest.mark.parametrize("alias", ["transformation", "get_transformation"])
def test_transformation_aliases_accept_call_arguments(mode, alias):
    tf = _builder(mode)
    tr = getattr(tf, alias)(2, b=3)
    assert isinstance(tr, Transformation)
    assert tr.run() == 5


@pytest.mark.parametrize(
    "mode",
    [
        "delayed",
        "direct",
    ],
)
@pytest.mark.parametrize("alias", ["transformation", "get_transformation"])
def test_transformation_aliases_return_transformation_for_every_call_mode(
    mode, alias
):
    tf = _builder(mode)
    tf.pins.a = 2
    tf.pins.b = 3
    tr = getattr(tf, alias)()
    assert isinstance(tr, Transformation)
    assert tr.run() == 5


def test_directness_does_not_change_snapshot_identity():
    d = _builder("delayed")
    D = _builder("direct")
    t1 = d.build(2, 3)
    t2 = D.build(2, 3)
    t1.construct()
    t2.construct()
    assert t1.transformation_checksum == t2.transformation_checksum


def test_compiled_builders_have_build():
    for is_direct in (False, True):
        tf = Transformer("c", compiled=True, direct=is_direct)
        assert callable(getattr(tf, "build", None))


def test_build_raises_for_malformed_arguments_without_executing():
    tf = _builder("delayed")
    with pytest.raises(TypeError):
        tf(1, 2, c=3)
    with pytest.raises(TypeError):
        tf(object(), 2)


# --- snapshot isolation --------------------------------------------------------


def test_mutating_builder_after_build_does_not_alter_the_transformation():
    tf = _builder("delayed")
    tf.pins.a = 2
    tf.pins.b = 3
    tr = tf()

    tf.pins.a = 100
    tf.code = mul
    tf.celltypes.result = "text"
    tf.meta = {"extra": 1}

    assert tr.run() == 5
    assert tf().run() == "300"


def test_mutating_original_input_object_after_build_does_not_alter_it():
    tf = delayed(length)
    tf.local = True
    data = [1, 2]
    tr = tf(data)
    data.append(3)
    assert tr.run() == 2


# --- named work methods, standalone -------------------------------------------


@pytest.mark.parametrize(
    "mode",
    [
        "delayed",
        "direct",
    ],
)
def test_standalone_named_methods_ignore_call_mode(mode):
    tf = _builder(mode)
    tf.pins.a = 2
    tf.pins.b = 3
    reference = _builder("delayed")
    reference.pins.a = 2
    reference.pins.b = 3
    expected_checksum = reference().compute()

    assert tf.compute() == expected_checksum
    assert tf.run() == 5
    assert asyncio.run(tf.computation()) == expected_checksum

    async def via_task():
        return await tf.task()

    asyncio.run(via_task())


@pytest.mark.parametrize("mode", ["delayed", "direct"])
@pytest.mark.parametrize("method", ["prune", "clear_exception"])
def test_standalone_prune_and_clear_exception_are_unavailable(mode, method):
    tf = _builder(mode)
    with pytest.raises(AttributeError):
        getattr(tf, method)()


@pytest.mark.parametrize("mode", ["delayed", "direct"])
@pytest.mark.parametrize("attr", ["result", "state", "block_reason", "exception"])
def test_standalone_has_no_live_node_attributes(mode, attr):
    tf = _builder(mode)
    with pytest.raises(AttributeError):
        getattr(tf, attr)


def test_compiled_standalone_has_no_result_and_readonly_language():
    tf = Transformer("c", compiled=True)
    with pytest.raises(AttributeError):
        tf.result
    with pytest.raises(AttributeError):
        tf.language = "cpp"
    assert tf.language == "c"


# --- factory example ---------------------------------------------------------


def test_workflow_namespace_bash_factory_example():
    from seamless.workflow import Transformer as WorkflowTransformer

    assert WorkflowTransformer is Transformer
    tf = WorkflowTransformer("bash", direct=True)
    tf.local = True
    tf.code = "cat input > RESULT"
    tf.pins.input = "hi"
    assert tf() == "hi\n"
