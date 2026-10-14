"""Phase 5 checksum parity: ordinary results and deepcell member classification.

Phase 4's execution helper also compares every deterministic CLI success to a
fresh in-memory execution. Process IDs, times, interpreter locations and held
lock choreography are explicitly listed there as observations outside parity.
Run this file separately from the Dask file, which owns remote service lifetime.
"""
import uuid

import pytest

from seamless import Buffer, Checksum
from seamless_transformer import Transformer
from seamless_transformer.run import run_transformation_dict
from test_python_cli_backend import _run_python, _run_handle, _held


RESULT_CASES = [
    pytest.param("None", id="none"),
    pytest.param("True", id="bool"),
    pytest.param("42", id="int"),
    pytest.param("2**63 - 1", id="large-int"),
    pytest.param("1.5", id="float"),
    pytest.param("float('nan')", id="nan"),
    pytest.param("float('inf')", id="inf"),
    pytest.param("float('-inf')", id="negative-inf"),
    pytest.param("'unicode λ\\x00'", id="string"),
    pytest.param("''", id="empty-string"),
    pytest.param("b''", id="empty-bytes"),
    pytest.param("b'\\x93NUMPY\\x00raw'", id="npy-magic-bytes"),
    pytest.param("[1, True, None, {'x': 2}]", id="plain-list"),
    pytest.param("(1, 2)", id="tuple"),
    pytest.param("{1: 2}", id="integer-dict-key"),
    pytest.param("np.int64(3)", id="numpy-int"),
    pytest.param("np.float64(1.5)", id="numpy-float"),
    pytest.param("np.bool_(True)", id="numpy-bool"),
    pytest.param("{'nested': [np.bool_(True), {'flag': np.bool_(False)}]}", id="nested-numpy-bools"),
    pytest.param("np.array(3, dtype=np.int16)", id="zero-dimensional-array"),
    pytest.param("np.arange(6, dtype=np.float32).reshape(2, 3)", id="matrix"),
    pytest.param("np.empty((0, 3), dtype=np.uint16)", id="empty-array"),
    pytest.param("np.array(['λ', 'word'])", id="unicode-array"),
    pytest.param("np.array([(1, 2)], dtype=np.dtype([('x', 'i4'), ('y', 'i4')], align=True))", id="structured-array"),
    pytest.param("{'array': np.arange(3), 'plain': {'x': 3}}", id="mixed-dict"),
    pytest.param("[np.arange(3), np.array([1.5])]", id="mixed-list"),
    pytest.param("{'tuple': (np.arange(3), b'raw')}", id="nested-mixed"),
]


def _both(code, pins=None, **kwargs):
    # Backend is outside identity. Scratch executions and an unused nonce ensure
    # these assertions measure executions rather than backend cache reuse.
    pins = dict(pins or {})
    pins['parity_nonce'] = (uuid.uuid4().hex, 'str')
    results = [_run_python(code, pins, backend=backend, **kwargs)[0]
               for backend in ('mem', 'cli')]
    assert results[0] == results[1]
    return results[0]


@pytest.mark.parametrize('expression', RESULT_CASES)
@pytest.mark.parametrize('output', ['mixed', 'deepcell'])
def test_result_type_and_deepcell_member_checksum_parity(expression, output):
    assignment = expression if output == 'mixed' else "{'member': " + expression + "}"
    checksum = _both('import numpy as np\nresult = ' + assignment, output=output)
    if output == 'deepcell':
        index = checksum.resolve().get_value('plain')
        member_checksum = _both('import numpy as np\nresult = ' + expression)
        assert index == {'member': member_checksum.hex()}


@pytest.mark.parametrize('code', [
    'result = value + 5',
    'def calculate(value, **kwargs):\n    return value + 5',
    'lambda value, **kwargs: value + 5',
    'value + 5',
])
def test_four_code_modes_checksum_parity(code):
    _both(code, {'value': (9, 'int')})


@pytest.mark.parametrize('output,expression', [
    ('plain', "{'x': [True, None, 3]}"), ('text', "'some text'"),
    ('str', "'some string'"), ('int', '42'), ('float', '1.5'),
    ('bool', 'True'), ('bytes', "b'\\x93NUMPYraw'"),
    ('binary', 'np.arange(3, dtype=np.int16)'),
    ('folder', "{'a': b'raw', 'b': b'\\x93NUMPYraw'}"),
])
def test_result_celltype_checksum_parity(output, expression):
    _both('import numpy as np\nresult = ' + expression, output=output)


@pytest.mark.parametrize('expression', ["{1, 2}", 'object()'])
@pytest.mark.parametrize('output', ['mixed', 'deepcell'])
def test_unserializable_result_fails_under_both_backends(expression, output):
    expression = expression if output == 'mixed' else "{'member': " + expression + "}"
    for backend in ('mem', 'cli'):
        with pytest.raises(Exception):
            _run_python('result = ' + expression, backend=backend, output=output)



@pytest.mark.parametrize("output", ["mixed", "deepcell"])
@pytest.mark.parametrize("container", [
    pytest.param("array", id="direct"),
    pytest.param("{'nested': array}", id="dict"),
    pytest.param("[array]", id="list"),
    pytest.param("(array,)", id="tuple"),
    pytest.param("{'nested': [{'leaf': array}]}", id="mixed"),
])
def test_padded_aligned_structured_array_is_one_sided_failure(
    output, container, capfd
):
    value = container if output == "mixed" else "{'member': " + container + "}"
    code = (
        "import numpy as np\n"
        "array = np.empty(1, dtype=np.dtype([('x', 'i4'), ('y', 'f8')], align=True))\n"
        "array.view(np.uint8)[...] = 165\n"
        "array['x'] = 1\n"
        "array['y'] = 2.5\n"
        f"result = {value}"
    )
    in_memory = _run_python(code, backend="mem", output=output)[0]
    assert isinstance(in_memory, Checksum)
    assert in_memory.resolve() is not None

    with pytest.raises(Exception):
        _run_python(code, backend="cli", output=output)
    assert "structured NumPy array with padding" in capfd.readouterr().err


@pytest.mark.parametrize("backend", ["mem", "cli"])
@pytest.mark.parametrize("output", ["mixed", "deepcell"])
@pytest.mark.parametrize("container", [
    pytest.param("array", id="direct"),
    pytest.param("(array,)", id="tuple"),
])
def test_padded_structured_subarray_fails_in_both_backends(
    backend, output, container, capfd
):
    value = container if output == "mixed" else "{'member': " + container + "}"
    code = (
        "import numpy as np\n"
        "inner = np.dtype([('x', 'i4'), ('y', 'f8')], align=True)\n"
        "outer = np.dtype([('items', inner, (2,))], align=True)\n"
        "array = np.empty(1, dtype=outer)\n"
        "array.view(np.uint8)[...] = 165\n"
        "array['items']['x'] = 1\n"
        "array['items']['y'] = 2.5\n"
        f"result = {value}"
    )
    with pytest.raises(Exception):
        _run_python(code, backend=backend, output=output)
    if backend == "cli":
        assert "structured NumPy array with padding" in capfd.readouterr().err


def test_checksum_member_is_reference_but_hex_string_is_value():
    member = _held('member value', 'mixed')
    hex_value = member.get_checksum().hex()
    checksum = _both(
        "from seamless import Checksum\nresult = {'reference': Checksum(value), 'string': value}",
        {'value': (hex_value, 'str')}, output='deepcell',
    )
    assert checksum.resolve().get_value('plain') == {
        'reference': hex_value, 'string': Buffer(hex_value, 'mixed').get_checksum().hex(),
    }


def test_globals_are_one_sided_failure():
    tf = Transformer('python')
    tf.code = 'result = constant'
    tf.globals.constant = 42
    tf.pins['parity_nonce'] = uuid.uuid4().hex
    tf.metavars.BACKEND = 'cli'
    with pytest.raises(NotImplementedError, match='BACKEND="mem"'):
        _run_handle(tf.build())
    tf.metavars.BACKEND = 'mem'
    assert _run_handle(tf.build()) == Buffer(42, 'mixed').get_checksum()


def test_modules_are_one_sided_failure():
    module = _held({'language': 'python', 'code': 'answer = 42'}, 'plain')
    code = _held('result = helper.answer', 'text')
    nonce = _held(uuid.uuid4().hex, 'str')
    transformation = {
        '__language__': 'python',
        'code': ('text', None, code.get_checksum().hex()),
        'helper': ('plain', 'module', module.get_checksum().hex()),
        'parity_nonce': ('str', None, nonce.get_checksum().hex()),
        '__output__': ('result', 'mixed', None),
    }
    checksum = Buffer(transformation, 'plain').get_checksum()
    def run(backend):
        return run_transformation_dict(transformation, checksum,
            tf_dunder={'__meta__': {'metavars': {'BACKEND': backend}}},
            scratch=True, require_value=True)
    with pytest.raises(NotImplementedError, match='BACKEND="mem"'):
        run('cli')
    assert Checksum(run('mem')) == Buffer(42, 'mixed').get_checksum()


def test_bytes_pin_type_is_explicitly_outside_parity():
    pins = {'value': (b'raw', 'bytes')}
    code = "result = type(value).__name__"
    mem = _run_python(code, pins, backend='mem')[0]
    cli = _run_python(code, pins, backend='cli')[0]
    assert mem.resolve().get_value('mixed') == 'Buffer'
    assert cli.resolve().get_value('mixed') == 'bytes'


@pytest.mark.parametrize('expression', [
    "Checksum('e' * 64)", "{'nested': [Checksum('e' * 64)]}",
    "{'nested': {'leaf': Checksum('e' * 64)}}",
])
def test_checksum_objects_cannot_hide_in_ordinary_mixed_results(expression):
    for backend in ('mem', 'cli'):
        with pytest.raises(Exception):
            _run_python('from seamless import Checksum\nresult = ' + expression,
                        backend=backend)


def test_deep_result_requires_dict_under_both_backends():
    code = "from collections import UserDict\nresult = UserDict({'member': 42})"
    for backend in ('mem', 'cli'):
        with pytest.raises(Exception):
            _run_python(code, backend=backend, output='deepcell')


@pytest.mark.parametrize('expression', ["Checksum('e' * 64)", "'e' * 64"])
def test_checksum_output_parity(expression):
    _both('from seamless import Checksum\nresult = ' + expression, output='checksum')


def test_direct_nested_function_deterministic_result_parity():
    _both("""
from seamless.transformer import direct
@direct
def local(value):
    return value + 1
result = local(41)
""")


def test_deepcell_member_resolution_without_dask_is_one_sided():
    member = _held('local member ' + uuid.uuid4().hex, 'text')
    index = _held({'member': member.get_checksum().hex()}, 'plain')
    pins = {'values': (index, 'deepcell')}
    code = "result = values['member'].resolve().get_value('text')"
    mem = _run_python(code, pins, backend='mem', output='text')[0]
    assert mem == member.get_checksum()
    # No client configuration is exported to a plain CLI job, so it cannot
    # resolve a member held only in the parent's cache.
    with pytest.raises(Exception):
        _run_python(code, pins, backend='cli', output='text')


def test_custom_namespace_checksum_parity():
    _both("custom = [OUTPUTPIN, PINS['value'], FILESYSTEM['value']['celltype'], __name__, __package__]",
          {'value': (9, 'int')}, output='plain', output_name='custom',
          formats={'value': {'celltype': 'int', 'filesystem': {'filesystem': False}}})


@pytest.mark.parametrize('code', ['pass', "raise ValueError('parity failure')"])
def test_missing_result_and_body_exception_fail_under_both_backends(code):
    for backend in ('mem', 'cli'):
        with pytest.raises(Exception):
            _run_python(code, backend=backend)


def test_decimal_str_result_is_one_sided_failure():
    code = "from decimal import Decimal\nresult = Decimal('1.25')"
    mem = _run_python(code, backend='mem', output='str')[0]
    assert mem == Buffer('1.25', 'str').get_checksum()
    with pytest.raises(Exception):
        _run_python(code, backend='cli', output='str')
