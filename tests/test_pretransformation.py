from seamless import Buffer
from seamless.caching.buffer_cache import get_buffer_cache
from seamless_transformer.pretransformation import PreTransformation


def test_pretransformation_non_scratch_input_is_a_refholder():
    checksum = Buffer(b"pretransformation input").get_checksum()
    pre = PreTransformation(
        {
            "__language__": "python",
            "__output__": ("result", "mixed", None),
            "value": ("text", None, checksum),
        }
    )
    pre.prepare_transformation()
    assert len(pre._value_refs) == 1
    assert get_buffer_cache().reference_snapshot()[checksum][0] == 1
    pre.release()
    pre.release()
    assert get_buffer_cache().reference_snapshot().get(checksum, (0, 0, False))[0] == 0


def test_pretransformation_scratch_input_is_tempref_only():
    checksum = Buffer(b"pretransformation scratch").get_checksum()
    checksum.tempref(scratch=True)
    pre = PreTransformation(
        {
            "__language__": "python",
            "__output__": ("result", "mixed", None),
            "value": ("text", None, checksum),
        }
    )
    pre.prepare_transformation()
    assert pre._value_refs == []
    assert get_buffer_cache().reference_snapshot().get(checksum, (0, 0, False))[0] == 0
    pre.release()
