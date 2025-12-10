import seamless
from seamless.transformer import direct, delayed
from seamless.transformer import spawn

if __name__ == "__main__":

    spawn(1)

    @direct
    def func(a, b):
        raise Exception
        return 10000 * a + 4 * b

    print(func(1, 2))

    seamless.close()
