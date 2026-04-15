import builtins

import pytest

from seamless_transformer import CompiledObject, CompiledTransformer
from seamless_transformer.transformation_class import Transformation


SCALAR_SCHEMA = """\
function_name: add
inputs:
  - {name: a, dtype: int32}
  - {name: b, dtype: int32}
outputs:
  - {name: result, dtype: int32}
"""


MULTI_SCHEMA = """\
function_name: addmul
inputs:
  - {name: a, dtype: int32}
  - {name: b, dtype: int32}
outputs:
  - {name: sum, dtype: int32}
  - {name: product, dtype: int32}
"""


WILDCARD_SCHEMA = """\
function_name: take_positive
inputs:
  - {name: values, dtype: int32, shape: [N]}
outputs:
  - {name: result, dtype: int32, shape: [K]}
"""


def test_compiled_transformer_api(tmp_path):
    tf = CompiledTransformer("c")
    assert tf.language == "c"
    with pytest.raises(AttributeError):
        tf.language = "cpp"

    schema_path = tmp_path / "schema.yaml"
    code_path = tmp_path / "code.c"
    schema_path.write_text(SCALAR_SCHEMA)
    code_path.write_text("int transform(int a, int b, int *result) { return 0; }")

    tf.schema = schema_path
    tf.code = code_path
    assert "int transform(" in tf.header
    assert tf.code.startswith("int transform")

    result = tf(a=1, b=2)
    assert isinstance(result, Transformation)


def test_unregistered_language_and_missing_signature(monkeypatch):
    with pytest.raises(KeyError):
        CompiledTransformer("not-a-language")

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "seamless_signature":
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="seamless-signature is required"):
        CompiledTransformer("c")


def test_metavars_rebuild_and_schema_validation():
    tf = CompiledTransformer("c")
    tf.schema = WILDCARD_SCHEMA
    tf.metavars.maxK = 10
    assert tf.metavars.maxK == 10
    tf.schema = SCALAR_SCHEMA
    with pytest.raises(AttributeError):
        _ = tf.metavars.maxK

    with pytest.raises(NotImplementedError):
        tf.schema = """\
function_name: bad
inputs:
  - name: item
    dtype:
      fields:
        - {name: x, dtype: int32}
outputs:
  - {name: result, dtype: int32}
"""


def test_multi_output_result_celltypes():
    tf = CompiledTransformer("c")
    tf.schema = MULTI_SCHEMA
    tf.celltypes.result = "mixed"
    tf.celltypes.result = "deepcell"
    with pytest.raises(TypeError):
        tf.celltypes.result = "int"


def test_incomplete_transformer_and_objects():
    tf = CompiledTransformer("c")
    with pytest.raises(ValueError):
        tf(a=1)

    tf.schema = WILDCARD_SCHEMA
    tf.code = "int transform(void) { return 0; }"
    with pytest.raises(ValueError, match="metavars"):
        tf(values=[1, 2, 3])

    obj = CompiledObject(language="fortran")
    obj.code = "subroutine helper()\nend subroutine\n"
    tf.objects.append(obj)
    assert tf.objects[0].language == "fortran"
