import seamless
from seamless.transformer import direct, delayed

#####


@delayed
def func(a, b):
    return 10000 * a + 4 * b


#####

print(func(1, 2).run())

#####

seamless.close()
