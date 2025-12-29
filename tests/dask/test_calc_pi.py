import os
import time

import seamless.config

seamless.config.init()

import seamless
from seamless.transformer import delayed
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

    seed = 86246
    np.random.seed(seed)
    # ntrials = 1000 ### doesn't work
    ntrials = int(os.environ.get("SEAMLESS_TEST_PI_TRIALS", "10"))
    seeds = np.random.randint(0, 999999, ntrials)
    ndots = int(os.environ.get("SEAMLESS_TEST_PI_DOTS", "1000000"))

    start = time.perf_counter()
    tasks = [calc_pi(seeds[idx], ndots).start() for idx in range(ntrials)]
    results0 = [tf.compute() for tf in tqdm(tasks)]
    print(sum([tasks[n].exception is not None for n in range(len(tasks))]))
    for n in range(ntrials):
        if tasks[n].exception:
            print(n, tasks[n].exception)
            return

    results = [cs.resolve("mixed") for cs in tqdm(results0)]
    results = np.array(results)

    duration = time.perf_counter() - start
    print(duration)
    print(results.mean(), results.std(), np.pi)
