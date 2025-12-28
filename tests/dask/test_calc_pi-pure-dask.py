import os
import time

import seamless.config

seamless.config.init()

from distributed import as_completed, get_client

from dask import delayed
from tqdm import tqdm


@delayed
def calc_pi(seed, ndots):
    import numpy as np

    np.random.seed(seed)
    CHUNKSIZE = 1000000
    in_circle = 0
    for n in range(0, ndots, CHUNKSIZE):
        nndots = min(n + CHUNKSIZE, ndots) - n
        x = 2 * np.random.rand(nndots) - 1
        y = 2 * np.random.rand(nndots) - 1
        dis = x**2 + y**2
        in_circle += (dis <= 1).sum()
    frac = in_circle / ndots
    pi = 4 * frac
    return pi


def test_calc_pi():
    import numpy as np

    seed = 0
    np.random.seed(seed)
    # ntrials = 1000 ### doesn't work
    ntrials = 300
    seeds = np.random.randint(0, 999999, ntrials)
    ndots = 1000000000

    start = time.perf_counter()
    tasks = [calc_pi(seeds[idx], ndots) for idx in range(ntrials)]

    import distributed

    client = distributed.get_client()
    assert client is not None
    futures = client.compute(tasks)
    results = []
    for fut in tqdm(as_completed(futures), total=len(futures)):
        results.append(fut.result())
    results = np.array(results)

    duration = time.perf_counter() - start
    print(duration)
    print(results.mean(), results.std(), np.pi)


if __name__ == "__main__":
    test_calc_pi()
