import pytest
from seamless import Buffer, Cell
from seamless.transformer import delayed
from seamless_transformer.transformation_class import TransformationError


@delayed
def identity(value):
    return value


@delayed
def null_result():
    return None


@pytest.mark.parametrize("prebound", [False, True])
def test_typed_transformation_is_converted_to_text(prebound):
    identity.local = True
    identity.celltypes.value = 'str'
    identity.celltypes.result = 'str'
    upstream = identity('hello')
    identity.celltypes.value = 'text'
    identity.celltypes.result = 'text'
    if prebound:
        identity.pins.value = upstream
        transformation = identity()
    else:
        transformation = identity(upstream)
    assert transformation.run() == 'hello'
    if prebound:
        assert delayed(identity)().run() == 'hello'


def test_cell_expression_is_converted_to_text():
    identity.local = True
    identity.celltypes.value = 'text'
    identity.celltypes.result = 'text'
    cell = Cell('str')
    cell.set('hello')
    assert identity(cell.build()).run() == 'hello'


@pytest.mark.parametrize('celltype', ['plain', 'mixed', 'bytes'])
def test_null_result_completes(celltype):
    null_result.local = True
    null_result.celltypes.result = celltype
    tf = null_result()
    assert tf.run() == (b'' if celltype == 'bytes' else None)
    assert tf.result_checksum == Buffer(None, 'plain').get_checksum()


def test_binary_null_result_rejected():
    null_result.local = True
    null_result.celltypes.result = 'binary'
    with pytest.raises(TransformationError, match='Null result.*binary'):
        null_result().run()


def test_required_int_null_rejected_but_cell_accepts_null():
    cell = Cell('int')
    cell.set(None)
    assert cell.run() is None
    identity.local = True
    identity.celltypes.value = 'int'
    identity.celltypes.result = 'int'
    with pytest.raises(TypeError, match="Required pin 'value'.*int.*null"):
        identity(None)
    tf = identity(cell.build())
    assert tf.construct() is None
    assert "Required pin 'value'" in tf.exception


def test_optional_binary_dependency_null_is_absent(monkeypatch):
    @delayed
    def consume(value=None):
        return value is None

    consume.local = True
    consume.celltypes.value = 'binary'
    assert 'value' in consume.optional_pins
    null_result.local = True
    null_result.celltypes.result = 'plain'
    # A null dependency must bypass expression decoding/conversion entirely.
    from seamless.checksum import expression as expression_evaluator
    def reject_decode(*args, **kwargs):
        raise AssertionError('optional null must never be decoded')
    monkeypatch.setattr(expression_evaluator, '_deserialize_for_expression', reject_decode)
    absent = consume()
    connected = consume(null_result())
    assert connected.run() is True
    assert connected.construct() == absent.construct()


def test_converted_pin_retains_string_checksum():
    identity.local = True
    identity.celltypes.value = identity.celltypes.result = 'str'
    upstream = identity('42')
    identity.celltypes.value = identity.celltypes.result = 'int'
    converted = identity(upstream)
    literal = identity(42)
    assert converted.construct() != literal.construct()
    payload = converted.construct().resolve('plain')
    assert payload['value'][2] == Buffer('42', 'str').get_checksum().hex()
    assert converted.run() == 42


def test_empty_bytes_result_and_required_pin_are_null():
    @delayed
    def empty():
        return b''

    @delayed
    def size(value):
        return len(value)

    empty.local = size.local = True
    empty.celltypes.result = size.celltypes.value = 'bytes'
    upstream = empty()
    assert upstream.run() == b''
    assert upstream.result_checksum == Buffer(None, 'plain').get_checksum()
    assert size(upstream).run() == 0
    assert size(None).run() == 0


def test_bare_empty_bytes_checksum_is_optional_absence():
    @delayed
    def consume(value=None):
        return value is None

    consume.local = True
    consume.celltypes.value = 'bytes'
    assert 'value' in consume.optional_pins
    empty = consume(Buffer(b'').get_checksum())
    assert empty.construct() == consume().construct()
    assert empty.run() is True
