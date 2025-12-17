import seamless
import seamless.config
from seamless.transformer import direct, delayed

seamless.config.set_stage("simple-remote")
seamless.config.init()


@direct
def func(a, b):
    return 10000 * a + 4 * b + 0.0


print("START")
print(func(1, 2))
print(func(2, 3))

seamless.close()
