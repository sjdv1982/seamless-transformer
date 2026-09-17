import inspect

from seamless_transformer import delayed
from seamless_transformer.builder_snapshot import TransformerBuilderSnapshot
from seamless_transformer.transformation_class import Transformation


def test_bound_backend_snapshot_uses_shared_transformation_assembly():
    def identity(value):
        return value

    tf = delayed(identity)
    standalone = tf(value=3)

    class Backend:
        def snapshot_for_call(self):
            return TransformerBuilderSnapshot(
                codebuf=tf._codebuf,
                language="python",
                celltypes={"value": "mixed", "result": "mixed"},
                optional_pins=frozenset(),
                args={},
                modules={},
                globals={},
                meta={"local": False},
                environment=None,
                scratch=False,
                direct_print=False,
                local=False,
                call_mode="delayed",
                callable=identity,
                signature=inspect.signature(identity),
            )

    object.__setattr__(tf, "_workflow_backend", Backend())
    bound = tf(value=3)
    assert isinstance(bound, Transformation)
    standalone.construct()
    bound.construct()
    assert bound.transformation_checksum == standalone.transformation_checksum
