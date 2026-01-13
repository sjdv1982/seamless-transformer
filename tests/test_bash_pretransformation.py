from seamless import Checksum

from seamless_transformer.pretransformation import PreTransformation
from seamless_transformer.transformation_class import transformation_from_pretransformation


def test_bash_pretransformation_construct():
    testdata = "a\nb\nc\nd\ne\nf\n"
    bashcode = (
        "head -$lines testdata > firstdata; "
        "mkdir -p RESULT/input; "
        "cp firstdata RESULT; "
        "cp testdata RESULT/input"
    )
    pre = PreTransformation(
        {
            "__language__": "bash",
            "__output__": ("result", "text", None),
            "code": ("text", None, bashcode),
            "lines": ("int", None, 3),
            "testdata": ("text", None, testdata),
        }
    )
    tf = transformation_from_pretransformation(
        pre,
        upstream_dependencies={},
        meta={},
        scratch=False,
    )
    checksum = tf.construct()
    assert isinstance(checksum, Checksum)
    assert len(checksum.hex()) == 64
