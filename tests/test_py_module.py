import numpy as np
import testingmodule


def logsum(x: list[float] | np.ndarray) -> float:
    import numpy as np

    return np.log(x).sum()


def logsum_notypes(x):
    import numpy as np

    return np.log(x).sum()


from seamless.transformer import direct


def func1(x):
    return logsum_notypes(x)


def func2(x: np.ndarray) -> float:
    return logsum_notypes(x)


def func3(x: np.ndarray) -> float:
    return logsum(x)


def func4(x: np.ndarray) -> float:
    return testingmodule.stdev_unbiased(x)


def test_func1():
    testdata = np.array([4, 5, 2.8, 1, 16, 66])
    result1 = func1(testdata)
    func1a = direct(func1)
    func1a.globals.logsum_notypes = logsum_notypes
    result2 = func1a(testdata)
    assert result1 == result2


def test_func2():
    testdata = np.array([4, 5, 2.8, 1, 16, 66])
    result1 = func2(testdata)
    func2a = direct(func2)
    func2a.globals.logsum_notypes = logsum_notypes
    result2 = func2a(testdata)
    assert result1 == result2


def test_func3():
    testdata = np.array([4, 5, 2.8, 1, 16, 66])
    result1 = func3(testdata)
    func3a = direct(func3)
    func3a.globals.logsum = logsum
    result2 = func3a(testdata)
    assert result1 == result2


def test_func4():
    testdata = np.array([4, 5, 2.8, 1, 16, 66])
    result1 = func4(testdata)
    func4a = direct(func4)
    func4a.modules.testingmodule = testingmodule
    result2 = func4a(testdata)
    assert result1 == result2
