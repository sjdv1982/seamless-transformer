"""Standalone Cell reads stop at the Transformation execution boundary."""
from seamless import Buffer, Cell


def _identity(value):
    return value


def test_checksum_does_not_start_source_transformation():
    from seamless_transformer import delayed

    transformation = delayed(_identity)(12)
    cell = Cell("int", source=transformation)

    assert cell.checksum is None
    assert cell.state == "waiting"
    assert transformation._evaluated is False

    assert cell.compute() == Buffer(12, "int").get_checksum()
    assert transformation._evaluated is True


