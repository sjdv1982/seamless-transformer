from seamless_transformer.builder_snapshot import TransformerBuilderSnapshot
from seamless_transformer.transformer_class import Transformer


def test_followup_public_surface_has_no_reserved_inp_namespace():
    assert not hasattr(Transformer, "inp")
    fields = set(TransformerBuilderSnapshot.__dataclass_fields__)
    assert {"codebuf", "celltypes", "args", "optional_pins", "call_mode"} <= fields
