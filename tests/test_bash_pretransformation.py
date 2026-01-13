from seamless import Checksum

from seamless_transformer.pretransformation import PreTransformation
from seamless_transformer.transformation_class import transformation_from_pretransformation


def _make_bash_pretransformation(lines, testdata, bashcode):
    return PreTransformation(
        {
            "__language__": "bash",
            "__output__": ("result", "mixed", None),
            "code": ("text", None, bashcode),
            "lines": ("int", None, lines),
            "testdata": ("text", None, testdata),
        }
    )


def test_bash_pretransformation_construct():
    testdata = "a\nb\nc\nd\ne\nf\n"
    bashcode = (
        "head -$lines testdata > firstdata; "
        "mkdir -p RESULT/input; "
        "cp firstdata RESULT; "
        "cp testdata RESULT/input"
    )
    expected_3 = {
        "firstdata": "a\nb\nc\n",
        "input/testdata": "a\nb\nc\nd\ne\nf\n",
    }
    expected_4 = {
        "firstdata": "a\nb\nc\nd\n",
        "input/testdata": "a\nb\nc\nd\ne\nf\n",
    }

    pre = _make_bash_pretransformation(3, testdata, bashcode)
    tf = transformation_from_pretransformation(
        pre,
        upstream_dependencies={},
        meta={},
        scratch=False,
    )
    checksum = tf.construct()
    assert isinstance(checksum, Checksum)
    assert len(checksum.hex()) == 64
    result = tf.run()
    assert result == expected_3

    pre = _make_bash_pretransformation(4, testdata, bashcode)
    tf = transformation_from_pretransformation(
        pre,
        upstream_dependencies={},
        meta={},
        scratch=False,
    )
    result = tf.run()
    assert result == expected_4
