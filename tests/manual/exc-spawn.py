import seamless
from seamless_transformer import transformer
from seamless_transformer.worker import spawn

if __name__ == "__main__":

    spawn(1)

    @transformer
    def func(a, b):
        raise Exception
        return 10000 * a + 4 * b

    print(func(1, 2))

    seamless.close()
