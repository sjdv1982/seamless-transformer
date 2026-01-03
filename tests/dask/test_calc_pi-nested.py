import os

import seamless.config

seamless.config.init()

from seamless.transformer import delayed, direct


@direct
def calc_pi_all(seed, ntrials, *, checksum_only, ndots=1000000000):

    import os
    import numpy as np

    from seamless.transformer import delayed, direct

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

    np.random.seed(seed)
    seeds = np.random.randint(0, 999999, ntrials)

    tasks = [calc_pi(seeds[idx], ndots).start() for idx in range(ntrials)]
    results0 = [tf.compute() for tf in tasks]
    for n in range(ntrials):
        if tasks[n].exception:
            raise RuntimeError(n, tasks[n].exception)

    if checksum_only:
        return [str(cs) for cs in results0]

    results = [cs.resolve("mixed") for cs in results0]
    results = np.array(results)

    return results.mean(), results.std(), np.pi


# calc_pi_all.celltypes["result"] = "bytes"
calc_pi_all.driver = True


def test_calc_pi():
    seed = 0
    ntrials = int(os.environ.get("SEAMLESS_TEST_PI_TRIALS", "1000"))
    checksum_only = False
    ndots = int(os.environ.get("SEAMLESS_TEST_PI_DOTS", "1000000000"))
    result = calc_pi_all(seed, ntrials, checksum_only=checksum_only, ndots=ndots)
    print(result)


if __name__ == "__main__":
    test_calc_pi()
