from seamless.transformer import direct


def test_bash_transformer():

    testdata = "a\nb\nc\nd\ne\nf\n"
    bashcode = (
        "head -$lines testdata > firstdata; "
        "mkdir -p RESULT/input; "
        "cp firstdata RESULT; "
        "cp testdata RESULT/input"
    )

    tf = direct(bashcode, "bash")
    tf.args.testdata = testdata
    tf.celltypes.lines = int

    expected_3 = {
        "firstdata": "a\nb\nc\n",
        "input/testdata": "a\nb\nc\nd\ne\nf\n",
    }
    expected_4 = {
        "firstdata": "a\nb\nc\nd\n",
        "input/testdata": "a\nb\nc\nd\ne\nf\n",
    }

    result = tf(lines=3)
    print(result)
    assert result == expected_3
    result = tf(lines=4)
    print(result)
    assert result == expected_4
