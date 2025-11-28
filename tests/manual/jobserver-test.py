import seamless
import seamless_config
from seamless import Buffer
from seamless_transformer import transformer
from seamless_config.select import get_execution

seamless_config.set_stage("jobserver-test")
seamless_config.init()


@transformer(return_transformation=True)
def func(a, b):
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

seamless.close()
