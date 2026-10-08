from seamless_transformer import delayed, direct


def test_transformer_streaming_is_passed_to_transformation():
    def add(a, b):
        return a + b

    tf = delayed(add)
    assert tf.streaming is False
    assert tf(1, 2).streaming is False

    tf.streaming = True
    t1 = tf(1, 2)
    assert t1.streaming is True
    assert direct(tf).streaming is True  # clones keep the flag

    tf.streaming = False
    assert tf(1, 2).streaming is False
    assert t1.streaming is True  # already built: unaffected
