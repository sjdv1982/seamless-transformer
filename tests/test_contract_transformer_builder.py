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
    with pytest.raises(Exception):  # type unstated by the contract
        direct(add, language="bash")  # pylint: disable=unexpected-keyword-arg
    with pytest.raises(Exception):  # type unstated by the contract
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


_COMPILED_SCHEMA = """\
inputs:
  - {name: a, dtype: int32}
  - {name: b, dtype: int32}
outputs:
  - {name: result, dtype: int32}
"""


@pytest.mark.parametrize("is_direct", [False, True])
def test_compiled_build_matches_call(is_direct):
    """build() on compiled builders returns a Transformation without executing;
    for a delayed builder it has the same identity as the call path."""
    tf = Transformer("c", compiled=True, direct=is_direct)
    tf.schema = _COMPILED_SCHEMA
    tf.code = "int transform(int a, int b, int *result) { *result = a + b; return 0; }"
    built = tf.build(a=1, b=2)
    assert isinstance(built, Transformation)
    built.construct()
    if not is_direct:
        called = tf(a=1, b=2)
        assert isinstance(called, Transformation)
        called.construct()
        assert built.transformation_checksum == called.transformation_checksum


def _compiled(is_direct):
    tf = Transformer("c", compiled=True, direct=is_direct)
    tf.schema = _COMPILED_SCHEMA
    tf.code = "int transform(int a, int b, int *result) { *result = a + b; return 0; }"
    tf.pins.a = 1
    tf.pins.b = 2
    return tf


def test_compiled_directness_does_not_change_snapshot_identity():
    t1 = _compiled(False).build()
    t2 = _compiled(True).build()
    t1.construct()
    t2.construct()
    assert t1.transformation_checksum == t2.transformation_checksum


@pytest.mark.parametrize("is_direct", [False, True])
def test_compiled_standalone_named_methods_ignore_call_mode(is_direct):
    """transformers.md §Named work methods: the table holds for compiled
    builders too; compute()/run() go through build(), not self()."""
    tf = _compiled(is_direct)
    expected_checksum = _compiled(False).build().compute()
    assert tf.compute() == expected_checksum
    assert tf.run() == 3
    assert asyncio.run(tf.computation()) == expected_checksum


@pytest.mark.parametrize("mode", ["delayed", "direct"])
@pytest.mark.parametrize("op", ["build", "transformation", "get_transformation"])
def test_build_raises_for_malformed_arguments_without_executing(tmp_path, mode, op):
    """transformers.md §Building a Transformation / §Unspecified exception
    classes: an unknown keyword, a missing required argument and too many
    positional arguments all make a build raise; the class is unspecified."""
    marker = tmp_path / "executed"

    def two_args(a, b):
        open(marker, "w").write("x")
        return a + b

    tf = (direct if mode == "direct" else delayed)(two_args)
    tf.local = True
    build = getattr(tf, op)
    with pytest.raises(Exception):  # unknown keyword; type unstated
        build(1, 2, c=3)
    with pytest.raises(Exception):  # missing required argument; type unstated
        build(1)
    with pytest.raises(Exception):  # too many positional arguments; type unstated
        build(1, 2, 3)
    with pytest.raises(Exception):  # unserializable concrete input; type unstated
        build(object(), 2)
    assert not marker.exists()


def test_call_raises_for_malformed_arguments():
    tf = _builder("delayed")
    with pytest.raises(Exception):  # type unstated by the contract
        tf(1, 2, c=3)
    with pytest.raises(Exception):  # type unstated by the contract
        tf(object(), 2)


@pytest.mark.parametrize("op", ["build", "transformation", "get_transformation"])
def test_direct_build_never_executes(tmp_path, op):
    """transformers.md §Building a Transformation: build() and its aliases
    return a Transformation and never execute it, whatever the call mode."""
    marker = tmp_path / "executed"

    def side_effect(path):
        open(path, "w").write("x")
        return 1

    tf = direct(side_effect)
    tf.local = True
    tr = getattr(tf, op)(str(marker))
    assert isinstance(tr, Transformation)
    assert not marker.exists()
    assert tr.run() == 1
    assert marker.exists()


@pytest.mark.parametrize("mode", ["delayed", "direct"])
def test_call_is_layered_on_build(mode):
    """transformers.md §Delayed and direct calls: delayed_tf(...) ==
    delayed_tf.build(...); direct_tf(...) == direct_tf.build(...).run()."""
    tf = _builder(mode)
    built = tf.build(2, 3)
    if mode == "delayed":
        called = tf(2, 3)
        assert isinstance(called, Transformation)
        built.construct()
        called.construct()
        assert built.transformation_checksum == called.transformation_checksum
    else:
        assert tf(2, 3) == built.run() == 5


def test_bash_undeclared_keyword_raises_at_build():
    tf = Transformer("bash", direct=True)
    tf.local = True
    tf.code = "cat input > RESULT"
    with pytest.raises(TypeError, match="Unexpected keyword argument: 'input'"):
        tf.build(input="hi")
    with pytest.raises(TypeError, match="Unexpected keyword argument: 'input'"):
        tf(input="hi")


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
        task = tf.task()
        assert isinstance(task, asyncio.Task)
        return await task

    asyncio.run(via_task())


@pytest.mark.parametrize("mode", ["delayed", "direct"])
@pytest.mark.parametrize("method", ["prune", "clear_exception"])
def test_standalone_prune_and_clear_exception_are_unavailable(mode, method):
    """transformers.md §Live-node members: the attributes exist on a
    standalone Transformer (hasattr is True); only calling them raises."""
    tf = _builder(mode)
    assert hasattr(tf, method)
    bound_method = getattr(tf, method)  # attribute access itself does not raise
    assert callable(bound_method)
    with pytest.raises(AttributeError):
        bound_method()


@pytest.mark.parametrize("mode", ["delayed", "direct"])
@pytest.mark.parametrize("attr", ["result", "state", "block_reason", "exception"])
def test_standalone_has_no_live_node_attributes(mode, attr):
    tf = _builder(mode)
    with pytest.raises(AttributeError):
        getattr(tf, attr)
    assert not hasattr(tf, attr)


def test_compiled_standalone_has_no_result_and_readonly_language():
    tf = Transformer("c", compiled=True)
    with pytest.raises(AttributeError):
        tf.result
    with pytest.raises(Exception):  # contract: read-only; type unstated
        tf.language = "cpp"
    assert tf.language == "c"


@pytest.mark.parametrize(
    "make",
    [
        lambda: _builder("delayed"),
        lambda: _builder("direct"),
        lambda: Transformer("python"),
        lambda: Transformer("bash", direct=True),
    ],
    ids=["delayed", "direct", "python-codeless", "bash-codeless"],
)
def test_language_is_read_only_on_ordinary_builders(make):
    """transformers.md §Canonical construction API: a builder's language is
    fixed at construction and read-only; exception class unspecified."""
    tf = make()
    language = tf.language
    with pytest.raises(Exception):  # type unstated by the contract
        tf.language = "bash" if language == "python" else "python"
    assert tf.language == language


# --- factory example ---------------------------------------------------------


def test_workflow_namespace_bash_factory_example():
    from seamless.workflow import Transformer as WorkflowTransformer

    assert WorkflowTransformer is Transformer
    tf = WorkflowTransformer("bash", direct=True)
    tf.local = True
    tf.code = "cat input > RESULT"
    tf.pins.input = "hi"
    assert tf() == "hi\n"
