import numpy as np


def stdev_unbiased(x: list | np.ndarray) -> float:
    return np.sum(x)
