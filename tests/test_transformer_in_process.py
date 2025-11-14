import seamless

from seamless_transformer import transformer


def test_in_process_transformer_execution():
    """Simple smoke test mirroring the original Seamless direct test."""

    @transformer(in_process=True)
    def func(a, b):
        return 10 * a + 2 * b

    assert func(30, 12) == 324
    assert func(40, 2) == 404
