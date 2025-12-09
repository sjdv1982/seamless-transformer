import seamless
import seamless_config
from seamless import Buffer
from seamless.transformer import direct, delayed
from seamless_config.select import get_execution

seamless_config.set_stage("jobserver-test")
seamless_config.init()


@delayed
def func(a, b):
    import subprocess

    ### return subprocess.check_output("hostname").decode()
    return 18 * a - 2 * b


print(get_execution())

tf = func(2, 3)
tf.construct()
print(tf.transformation_checksum)
print(tf.run())

tf = func(4, 5)
tf.construct()
print(tf.transformation_checksum)
print(tf.run())

KEY = 1  # change this to enforce cache misses
func = func.copy(return_transformation=False)
for n in range(10):
    print(n, func(n, -KEY))

seamless.close()
