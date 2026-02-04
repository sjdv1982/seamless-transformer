import seamless.config

seamless.config.set_stage("persistent-test")
seamless.config.init()

from seamless.transformer import delayed


def test_transformation_checksum():
    """Print out a transformation checksum, to be used with seamless-run-transformation"""

    @delayed
    def func(a, b, random_value) -> int:
        return a + b

    import numpy as np

    tf = func(12, 13, np.random.random())
    print(tf.construct())
    print(tf.exception)
