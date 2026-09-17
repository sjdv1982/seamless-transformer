import time
import uuid

import seamless.config

seamless.config.init()

from seamless.transformer import direct, delayed


def test_in_process_transformer_execution():
    """Simple smoke test mirroring the original Seamless direct test."""

    @direct
    def func(a: int, b: int) -> int:
        return 10 * a + 2 * b

    assert func(30, 12) == 324
    assert func(40, 2) == 404

    # Keep the cold call independent of a transformation left in a shared
    # Dask/database service by an earlier serial test process.
    nonce = uuid.uuid4().hex

    @direct
    def func2(a: int, b: int, _nonce: str) -> int:
        import time

        time.sleep(2)
        return 8 * a - 3 * b

    start = time.perf_counter()
    result1 = func2(3, 12, nonce)
    first_duration = time.perf_counter() - start

    start = time.perf_counter()
    result2 = func2(3, 12, nonce)
    second_duration = time.perf_counter() - start

    print(first_duration)
    print(second_duration)

    assert result1 == result2 == -12
    assert first_duration >= 2
    assert second_duration < 0.5
