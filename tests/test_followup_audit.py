from seamless_transformer.frozen_transformer import FrozenTransformer
from seamless_transformer.transformer_class import PythonBashBaseTransformer


def test_followup_public_surface_has_no_reserved_inp_namespace():
    assert not hasattr(PythonBashBaseTransformer, "inp")
    fields = set(FrozenTransformer.__dataclass_fields__)
    assert {"codebuf", "celltypes", "args", "optional_pins", "call_mode"} <= fields
