"""Cell/Pin boundary integration tests.

Contracts: cells.md (CellBase and input rejection), pins.md (A Pin is not a
source), direct-delayed-and-transformation.md (decorators return builders).
Lives in seamless-transformer because constructing Pins requires its builder.
"""
import pytest
from seamless import Cell
from seamless.cell_errors import ProjectionError
from seamless_transformer import delayed

def identity(x):
    return x


@pytest.fixture
def endpoints():
    transformer = delayed(identity)
    # Keep the builder alive without supplying inputs or running a body.
    yield Cell("plain"), transformer.pins.x


@pytest.mark.parametrize("operation", ["constructor", "with_input", "set", "value", "build_override"])
def test_cell_rejects_pin_with_source_guidance(endpoints, operation):
    cell, pin = endpoints
    with pytest.raises(TypeError, match=r"pin\.source"):
        if operation == "constructor":
            Cell(source=pin)
        elif operation == "with_input":
            cell.with_input(pin)
        elif operation == "set":
            cell.set(pin)
        elif operation == "value":
            cell.value = pin
        else:
            cell.build(pin)
    assert cell.source is None
    assert cell.checksum is None


@pytest.mark.parametrize("operation", [
    lambda p: p == 1, lambda p: p != 1,
    lambda p: p < 1, lambda p: p <= 1,
    lambda p: p > 1, lambda p: p >= 1,
    bool, len, iter,
], ids=["eq", "ne", "lt", "le", "gt", "ge", "bool", "len", "iter"])
def test_cellbase_handle_guards_also_cover_pins(endpoints, operation):
    _, pin = endpoints
    with pytest.raises(ProjectionError):
        operation(pin)
