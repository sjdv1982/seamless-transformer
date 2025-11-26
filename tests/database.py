from tqdm import tqdm
from seamless_transformer import transformer

import seamless.config

seamless.config.init()


@transformer
def func(a, b):
    print("RUN", a, b)
    return 12 * a + 7 * b + 12


OFFSET = 1
print("START")
print(func(100 * OFFSET + 3, 18))

print("START2")
print(func(100 * OFFSET + 4, 18))


print("START3")
print(func(100 * OFFSET + 5, 18))

print("ITER")
OFFSET = 3  # vary this to force cache misses
for a in tqdm(list(range(10))):
    for b in range(10):
        func(a + 10000 + 10 * OFFSET, 42 + b)
