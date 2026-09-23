import asyncio

import pytest
from seamless import Buffer, Cell, CellBase, Expression
from seamless_transformer import Pin, delayed


def defaults(a=3, *args, b=4, **kwargs):
    return [a, b]


def test_signature_view_toggle_clone_and_null_identity():
    tf = delayed(defaults)
    tf.local = True
    view = tf.optional_pins
    assert view == {'a', 'b'}
    assert 'args' not in dir(tf.pins) and 'kwargs' not in dir(tf.pins)
    assert tf().run() == [3, 4]
    absent = tf().construct()
    assert tf(a=None).construct() == absent
    with pytest.raises(AttributeError):
        tf.optional_pins = {'a'}
    with pytest.raises(AttributeError):
        view.add('a')
    view.a.disable()
    assert view == {'b'} and 'a' in dir(view)
    with pytest.raises(TypeError, match='a'):
        tf()
    clone = delayed(tf)
    assert clone.optional_pins == {'b'}
    assert tf(a=None).construct() != absent
    assert tf(a=None).run() == [None, 4]
    view.a.enable()
    assert tf().construct() == absent
    assert clone.optional_pins == {'b'}
    assert delayed('result = 1').optional_pins == set()


def test_sister_pin_reads_and_call_time_cell_conversion():
    tf = delayed(defaults)
    tf.local = True
    tf.celltypes.a = 'text'
    source = Cell('str')
    source.set('hello')
    assert tf(a=source).run() == ['hello', 4]
    tf.pins.a = source
    pin = tf.pins.a
    assert isinstance(pin, (Pin, CellBase)) and not isinstance(pin, Cell)
    assert pin is not tf.pins.a
    assert pin.build(Buffer('other', 'text').get_checksum()).run() == 'other'
    for factory in (Cell, Expression):
        with pytest.raises(TypeError, match='pin.source'):
            factory(source=pin) if factory is Cell else factory(pin)
    pin.celltype = 'int'
    assert pin.checksum is None
    assert isinstance(pin.exception, str)
    pin.celltype = 'text'
    assert pin.value == 'hello' and pin.exception is None
    source.set('changed')
    assert pin.value == tf.pins.a.value == 'changed'
    # Recovery uses the memo shared by fresh handles, without evaluating a pin.
    assert tf.pins.a.fingertip().content == Buffer('changed', 'text').content

    tf.pins.b = 8
    async def inspect_literal():
        assert tf.pins.b.checksum == Buffer(8, 'mixed').get_checksum()
        assert tf.pins.b.exception is None
    asyncio.run(inspect_literal())


def test_pin_null_revalidated_when_default_is_disabled():
    tf = delayed(defaults)
    tf.celltypes.a = 'int'
    tf.pins.a = None
    pin = tf.pins.a
    assert pin.checksum == Buffer(None, 'plain').get_checksum()
    tf.optional_pins.a.disable()
    assert pin.checksum is None
    assert "Required pin 'a'" in pin.exception
    tf.optional_pins.a.enable()
    assert pin.checksum == Buffer(None, 'plain').get_checksum()
    assert pin.exception is None


def test_pin_materialization_failure_is_shared_and_recovery_does_not_compute(monkeypatch):
    from seamless import Checksum
    tf = delayed(defaults)
    tf.pins.a = 3
    pin = tf.pins.a
    original = Checksum.resolve
    def fail(checksum, *args, **kwargs):
        raise ValueError('cannot decode pin')
    monkeypatch.setattr(Checksum, 'resolve', fail)
    from seamless.error_envelope import WorkflowExecutionError
    with pytest.raises(WorkflowExecutionError, match='cannot decode pin'):
        pin.value
    assert tf.pins.a.exception == 'cannot decode pin'
    monkeypatch.setattr(Checksum, 'resolve', original)
    tf.pins.a.clear_exception()
    assert pin.value == 3
    tf.celltypes.a = 'text'
    assert tf.pins.a.fingertip() is None
    assert pin.compute() is not None
    assert tf.pins.a.fingertip().content == Buffer('3', 'text').content
