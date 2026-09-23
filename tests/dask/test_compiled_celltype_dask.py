"""Compiled pin metadata and validation survive Dask submission."""

import pytest
import seamless.config
from seamless_dask.dummy_scheduler import create_dummy_client
from seamless_dask.transformer_client import set_seamless_dask_client
from seamless_transformer import CompiledPinSchemaError, Transformer, delayed

import seamless
from seamless import Buffer


def test_compiled_dask_celltype_contract():
    seamless.config.init()
    client = create_dummy_client(workers=1, worker_threads=2, spawn_workers=2)
    set_seamless_dask_client(client)
    try:
        tf = Transformer("c", compiled=True)
        tf.schema = "inputs:\n  - {name: x, dtype: int32}\noutputs:\n  - {name: result, dtype: int32}\n"
        tf.celltypes.x = "int"
        tf.code = "#include <stdint.h>\nint transform(int32_t x,int32_t *result) {*result=x+2;return 0;}"

        @delayed
        def upstream():
            return 37

        assert tf(x=upstream()).run() == 39

        @delayed
        def null_upstream():
            return None

        from seamless_transformer.transformation_class import TransformationError

        with pytest.raises(
            TransformationError, match="CompiledPinSchemaError.*x.*null"
        ):
            tf(x=null_upstream()).run()
        with pytest.raises(CompiledPinSchemaError, match="null"):
            tf(x=Buffer(None, "plain").get_checksum())
    finally:
        set_seamless_dask_client(None)
        seamless.close()
