import seamless
from seamless_transformer import transformer


@transformer
def func(a, b):
    return 10000 * a + 4 * b


print(func(1, 2))

seamless.close()
