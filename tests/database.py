import asyncio
from seamless import Buffer, Checksum
from seamless_transformer import transformer

import seamless.config

seamless.config.init()
b = Buffer(b"testbuffer")

### b.incref()  # launch hashserver in background
asyncio.run(b.write())  # wait for hashserver


@transformer
def func(a, b):
    print("RUN", a, b)
    import time

    time.sleep(1.5)
    return 12 * a + 7 * b


print("START")
print(func(3, 8))
print(func(3, 8))

print("START2")
print(func(4, 8))
print(func(4, 8))


print("START3")
print(func(5, 8))
print(func(5, 8))
