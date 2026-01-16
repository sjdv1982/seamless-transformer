import time

from seamless.transformer import direct, delayed


def test_in_process_transformer_execution():
    """Test arguments"""

    @direct
    def func(a, b, c):
        return 10 * a + 2 * b + c

    assert func(10, 4, 1) == 109
    func.args.c = 8
    assert func(20, 2) == 212
    func.args.a = 1
    assert func(10, 2) == 112
    assert func(b=6) == 30
    try:
        func(6)
        raise AssertionError
    except TypeError:
        pass
    func.args.b = 0
    assert func() == 18
