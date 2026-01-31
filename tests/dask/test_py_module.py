"""Version of test_py_module that is executed remotely
Added a sleep timer to verify cache hits
"""

import numpy as np
import testingmodule

import seamless.config

seamless.config.init()


def logsum(x: list[float] | np.ndarray, sleep) -> float:
    import time

    time.sleep(sleep)

    import numpy as np

    return np.log(x).sum()


def logsum_notypes(x, sleep):
    import time

    time.sleep(sleep)

    import numpy as np

    return np.log(x).sum()


from seamless.transformer import direct


def func1(x, sleep: float):
    return logsum_notypes(x, sleep=sleep)


def func2(x: np.ndarray, sleep: float) -> float:
    return logsum_notypes(x, sleep=sleep)


def func3(x: np.ndarray, sleep: float) -> float:
    return logsum(x, sleep=sleep)


def func4(x: np.ndarray, sleep: float) -> float:
    import time

    time.sleep(sleep)

    return testingmodule.stdev_unbiased(x)


def test_func1():
    testdata = np.array([4, 5, 2.8, 1, 16, 66])
    result1 = func1(testdata, sleep=0)
    func1a = direct(func1)
    func1a.globals.logsum_notypes = logsum_notypes
    result2 = func1a(testdata, sleep=1)
    assert result1 == result2


def test_func2():
    testdata = np.array([4, 5, 2.8, 1, 16, 66])
    result1 = func2(testdata, sleep=0)
    func2a = direct(func2)
    func2a.globals.logsum_notypes = logsum_notypes
    result2 = func2a(testdata, sleep=1)
    assert result1 == result2


def test_func3():
    testdata = np.array([4, 5, 2.8, 1, 16, 66])
    result1 = func3(testdata, sleep=0)
    func3a = direct(func3)
    func3a.globals.logsum = logsum
    result2 = func3a(testdata, sleep=1)
    assert result1 == result2


def test_func4():
    testdata = np.array([4, 5, 2.8, 1, 16, 66])
    result1 = func4(testdata, sleep=0)
    func4a = direct(func4)
    func4a.modules.testingmodule = testingmodule
    result2 = func4a(testdata, sleep=1)
    assert result1 == result2
