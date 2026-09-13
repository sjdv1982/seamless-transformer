import pytest
from seamless import AuthorityError, Buffer, Cell, CellBase, Expression
from seamless_transformer import Pin, delayed


def identity(value):
    return value


def builder(celltype='int'):
    tf = delayed(identity)
    tf.local = True
    tf.celltypes.value = celltype
    tf.celltypes.result = celltype
    return tf


def test_fresh_unwired_handle_and_reference_storage():
    tf = builder()
    pin = tf.pins.value
    assert isinstance(pin, Pin) and isinstance(pin, CellBase)
    assert not isinstance(pin, Cell)
    assert pin is not tf.pins.value
    assert pin.state == 'unwired' and pin.input_celltype is None
    assert pin.value is None and pin.checksum is None
    with pytest.raises(AttributeError):
        tf.pins.typo
    tf.pins.value = 10
    assert tf._args['value'] == (Buffer(10, 'int').get_checksum(), 'int')
    assert pin.value == pin.run() == pin.build().run() == 10
    assert pin.celltype == 'int' and pin.input_celltype == 'int'
    assert tf().run() == 10


def test_retype_link_and_declared_checksum():
    tf = builder('str')
    pin = tf.pins.value
    pin.set('42')
    pin.celltype = int
    assert tf.celltypes.value == 'int'
    assert pin.input_celltype == 'str'
    assert pin.value == 42
    tf.celltypes.value = 'text'
    assert pin.celltype == 'text' and pin.value == '42'
    pin.set_checksum(Buffer(43, 'int').get_checksum(), input_celltype='int')
    assert pin.value == '43'
    with pytest.raises(AttributeError):
        pin.input_celltype = 'text'


@pytest.mark.parametrize('write', ['value', 'buffer', 'checksum', 'set', 'set_buffer', 'set_checksum'])
def test_write_matrix(write):
    tf = builder()
    pin = tf.pins.value
    value = 12 if write in ('value', 'set') else Buffer(12, 'int')
    if 'checksum' in write:
        value = value.get_checksum()
    if write.startswith('set'):
        getattr(pin, write)(value)
    else:
        setattr(pin, write, value)
    assert pin.checksum == Buffer(12, 'int').get_checksum()
    assert pin.value == 12


def test_source_ownership_and_pin_rejection():
    tf = builder()
    upstream = Cell('int')
    upstream.set(12)
    tf.pins.value = upstream.build()
    pin = tf.pins.value
    assert isinstance(pin.source, Expression)
    for method, value in [('set', 4), ('set_buffer', Buffer(4, 'int')),
                          ('set_checksum', Buffer(4, 'int').get_checksum())]:
        with pytest.raises(AuthorityError):
            getattr(pin, method)(value)
    for make in [lambda: Cell(source=pin), lambda: Expression(pin),
                 lambda: tf(pin), lambda: setattr(tf.pins, 'value', pin)]:
        with pytest.raises(TypeError, match="Pin can't be a source.*pin.source"):
            make()
    pin.value = 4
    assert pin.source is None and pin.value == 4
    for name in ('item', 'slice', 'validator', 'mount', 'with_input', '_workflow_endpoint'):
        assert not hasattr(pin, name)


def test_invalid_assignment_and_clear():
    tf = builder()
    with pytest.raises((TypeError, ValueError)):
        tf.pins.value = 'invalid integer'
    assert tf.pins.value.state == 'unwired'
    tf.pins.value = 4
    tf.pins.value.checksum = None
    assert tf.pins.value.state == 'unwired'
    assert 'value' in tf.celltypes._celltypes
    with pytest.raises(AttributeError):
        del tf.pins.value


def test_compiled_pins():
    from seamless_transformer import CompiledTransformer
    from test_compiled_e2e import ADD_C, ADD_SCHEMA
    tf = CompiledTransformer('c')
    tf.schema = ADD_SCHEMA
    tf.code = ADD_C
    tf.local = True
    tf.pins.a = 2
    tf.pins.b = 3
    assert isinstance(tf.pins.a, Pin)
    assert tf.pins.a.value == 2
    tf.pins.a.celltype = int
    assert tf.celltypes.a == 'int'
    tf.celltypes.a = 'mixed'
    assert tf.pins.a.celltype == 'mixed'
    assert tf().run() == 5


def test_signatureless_declaration_delete_and_null_clear():
    tf = delayed('result = value')
    tf.pins.value = None
    pin = tf.pins.value
    assert pin.checksum == Buffer(None, 'plain').get_checksum()
    assert pin.state == 'complete'
    pin.buffer = None
    assert pin.state == 'unwired'
    tf.optional_pins.add('value')
    del tf.pins.value
    assert 'value' not in tf.optional_pins
    with pytest.raises(AttributeError):
        tf.pins.value
    with pytest.raises(AttributeError):
        pin.value


def test_failed_retype_reports_pin_exception_and_recovers():
    tf = builder('str')
    tf.pins.value = 'hello'
    pin = tf.pins.value
    pin.celltype = 'int'
    assert pin.state == 'failed'
    assert pin.exception is not None
    assert pin.checksum is None
    pin.celltype = 'text'
    assert pin.state == 'complete' and pin.exception is None
    assert pin.value == 'hello'


def test_retyped_clone_preserves_original_input():
    tf = builder('str')
    tf.pins.value = '42'
    tf.pins.value.celltype = 'int'
    clone = delayed(tf)
    assert clone.pins.value.input_celltype == 'str'
    assert clone._args == tf._args
    clone.pins.value.celltype = 'text'
    assert clone.pins.value.value == '42'
    assert tf.pins.value.value == 42


def test_cell_source_follows_input_type_and_retypes_at_call():
    tf = builder('text')
    upstream = Cell('str')
    upstream.set('42')
    tf.pins.value = upstream
    pin = tf.pins.value
    assert pin.source is upstream
    assert pin.input_celltype == 'str'
    assert tf().run() == '42'
    upstream.celltype = 'int'
    assert pin.input_celltype == 'int'
    assert pin.celltype == 'text'
    assert tf().run() == '42'


def test_pin_cell_source_snapshot_is_frozen():
    tf = builder('text')
    upstream = Cell('str')
    upstream.set('42')
    tf.pins.value = upstream
    snapshot = tf._snapshot_for_call()
    upstream.value = '43'
    upstream.celltype = 'int'
    assert tf._build_from_snapshot(snapshot).run() == '42'
    assert tf().run() == '43'
