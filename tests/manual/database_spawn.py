from tqdm import tqdm
from seamless_transformer import direct, delayed
from seamless_transformer.worker import spawn

import seamless.config
import seamless

seamless.config.init()

DELAY = 0.5
WORKERS = 8  # make this no bigger than #cores on your system
N = 4

OFFSET = 1  # vary this to force cache misses


@direct
def func(a, b, delay):
    import time

    print("RUN", a, b)
    time.sleep(delay)
    return 12 * a + 7 * b + 12


if __name__ == "__main__":
    spawn(WORKERS)

    print("START")
    print(func(100 * OFFSET + 3, 18, DELAY))

    print("START2")
    print(func(100 * OFFSET + 4, 18, DELAY))

    print("START3")
    print(func(100 * OFFSET + 5, 18, DELAY))

    func2 = func.copy(return_transformation=True)

    print("ITER")
    for a in tqdm(list(range(16))):
        tfs = []
        for b in range(16):
            tf = func2(a + 10000 + 10 * OFFSET, 42 + b, DELAY)
            tf.start()
            tfs.append(tf)
        for b in range(16):
            aa = a + 10000 + 10 * OFFSET
            bb = 42 + b
            tf = tfs[b]
            result = tf.run()
            assert result == 12 * aa + 7 * bb + 12, result

seamless.close()
