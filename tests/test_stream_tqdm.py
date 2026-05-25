from __future__ import annotations

from seamless_transformer.stream_tqdm import install_tqdm_patch


def test_tqdm_patch_emits_open_update_close() -> None:
    chunks = []
    with install_tqdm_patch(chunks.append):
        from tqdm import tqdm

        bar = tqdm(total=2, desc="remote")
        bar.update(1)
        bar.update(1)
        bar.close()

    kinds = [chunk["kind"] for chunk in chunks]
    assert kinds == ["tqdm_open", "tqdm_update", "tqdm_close"]
    assert chunks[0]["desc"] == "remote"
    assert chunks[-1]["n"] == 2


def test_tqdm_patch_restores_original_class() -> None:
    import tqdm as tqdm_module

    original = tqdm_module.tqdm
    with install_tqdm_patch(lambda _chunk: None):
        assert tqdm_module.tqdm is not original
    assert tqdm_module.tqdm is original

