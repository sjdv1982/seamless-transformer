import inspect

import pytest

from seamless.transformer import Transformer as NamespaceTransformer
from seamless_transformer import Transformer, delayed, direct


def test_factory_returns_code_less_python_and_bash_builders():
    expected = {
        ("python", False): "PythonTransformer",
        ("python", True): "DirectPythonTransformer",
        ("bash", False): "BashTransformer",
        ("bash", True): "DirectBashTransformer",
    }
    for (language, is_direct), class_name in expected.items():
        tf = Transformer(language, direct=is_direct)
        assert type(tf).__name__ == class_name
        assert tf.language == language
        assert tf.code is None
        with pytest.raises(AttributeError, match="read-only"):
            tf.language = language


def test_factory_rejects_noncompiled_languages_other_than_python_and_bash():
    with pytest.raises(ValueError, match="python.*bash"):
        Transformer("c")


def test_direct_and_delayed_are_python_only_and_keep_clone_conversion():
    assert tuple(inspect.signature(direct).parameters) == ("func",)
    assert tuple(inspect.signature(delayed).parameters) == ("func",)

    @delayed
    def add(a, b):
        return a + b

    direct_add = direct(add)
    delayed_again = delayed(direct_add)
    assert type(add).__name__ == "PythonTransformer"
    assert type(direct_add).__name__ == "DirectPythonTransformer"
    assert type(delayed_again).__name__ == "PythonTransformer"
    assert {add.language, direct_add.language, delayed_again.language} == {"python"}
    assert direct_add(2, 3) == 5
    assert delayed_again(2, 3).run() == 5


def test_seamless_transformer_namespace_exports_the_same_factory():
    assert NamespaceTransformer is Transformer


def test_concrete_builder_classes_are_not_top_level_exports():
    import seamless_transformer

    for name in (
        "PythonTransformer",
        "DirectPythonTransformer",
        "BashTransformer",
        "DirectBashTransformer",
        "CompiledTransformer",
        "DirectCompiledTransformer",
    ):
        assert not hasattr(seamless_transformer, name)


def test_compiled_factory_forwards_language_and_call_mode():
    delayed_compiled = Transformer("c", compiled=True)
    direct_compiled = Transformer("c", compiled=True, direct=True)
    assert type(delayed_compiled).__name__ == "CompiledTransformer"
    assert type(direct_compiled).__name__ == "DirectCompiledTransformer"
    assert delayed_compiled.language == direct_compiled.language == "c"

