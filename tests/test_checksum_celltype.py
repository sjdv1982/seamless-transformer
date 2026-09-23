"""Transformation pins and results of celltype checksum."""
from seamless import Buffer, Checksum
from seamless_transformer import Transformer, delayed


def test_checksum_pin_receives_a_checksum():
    @delayed
    def describe(pointer):
        return type(pointer).__name__ + ":" + pointer.hex()

    describe.local = True
    describe.celltypes.pointer = "checksum"
    describe.celltypes.result = "str"
    pointer = Checksum("ab" * 32)
    assert describe(Buffer(pointer, "checksum")).run() == "Checksum:" + pointer.hex()


def test_checksum_result_serializes_to_the_bare_digest():
    @delayed
    def make_pointer():
        from seamless import Checksum

        return Checksum("cd" * 32)

    make_pointer.local = True
    make_pointer.celltypes.result = "checksum"
    transformation = make_pointer()
    assert transformation.run() == Checksum("cd" * 32)
    assert transformation.compute().resolve().content == b"cd" * 32


def test_bash_checksum_pin_is_the_hex_digest():
    tf = Transformer("bash", direct=True)
    tf.code = "cat pointer > RESULT"
    tf.celltypes.pointer = "checksum"
    tf.celltypes.result = "text"
    pointer = Checksum("ef" * 32)
    assert tf(pointer=Buffer(pointer, "checksum")).strip() == pointer.hex()


def test_checksum_is_a_value_for_checksum_pins_only():
    @delayed
    def describe(pointer):
        return type(pointer).__name__ + ":" + pointer.hex()

    describe.local = True
    describe.celltypes.pointer = "checksum"
    describe.celltypes.result = "str"
    pointer = Checksum("ab" * 32)
    expected = "Checksum:" + pointer.hex()
    assert describe(pointer).run() == expected
    describe.pins.pointer = pointer
    assert describe.pins.pointer.value == pointer
    assert describe().run() == expected
    describe.pins.pointer.set(Checksum("cd" * 32))
    assert describe.pins.pointer.value == Checksum("cd" * 32)

    @delayed
    def echo(value):
        return value

    echo.local = True
    echo.celltypes.value = echo.celltypes.result = "str"
    pointed = Buffer("pointed", "str")
    pointed.tempref()
    # For any other celltype, a Checksum is a declared reference.
    assert echo(pointed.get_checksum()).run() == "pointed"


def test_hex_string_argument_is_a_value():
    @delayed
    def echo(value):
        return value

    echo.local = True
    echo.celltypes.value = echo.celltypes.result = "str"
    text = "ab" * 32
    assert echo(text).run() == text
