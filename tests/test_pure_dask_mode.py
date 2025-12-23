import pytest

from seamless_transformer.transformation_class import TransformationError
from seamless_transformer.transformer_class import delayed


def test_pure_dask_mode_forbids_transformer(monkeypatch):
    monkeypatch.setenv("SEAMLESS_PURE_DASK", "1")

    @delayed
    def add(a, b):
        return a + b

    with pytest.raises(TransformationError, match="pure Dask"):
        add(1, 2)
