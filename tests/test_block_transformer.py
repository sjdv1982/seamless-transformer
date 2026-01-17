from seamless.transformer import direct


def test_block_transformer():
    code = "result = 100 * a + 10 * b + c"
    tf = direct(code)
    tf.celltypes.a = int
    tf.celltypes.b = int
    tf.celltypes.c = int
    result = tf(a=2, b=3, c=4)
    assert result == 234
