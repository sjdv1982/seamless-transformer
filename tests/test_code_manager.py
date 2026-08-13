from seamless import Buffer
from seamless.caching.buffer_cache import get_buffer_cache
from seamless_transformer.code_manager import CodeManager


def test_code_manager_roles_have_logical_multiplicity_and_one_bridge():
    manager = CodeManager()
    code = Buffer(b"def f(x):\n    return x + 1\n", "python")
    semantic, syntactic = manager.track_code_buffer(code)
    manager.incref_syntactic(syntactic)
    manager.incref_syntactic(syntactic)
    manager.incref_semantic(semantic)
    cache = get_buffer_cache()
    snapshot = cache.reference_snapshot()
    assert snapshot[syntactic][0] == 3
    assert snapshot[semantic][0] == 3
    assert snapshot[syntactic][2] is True
    assert snapshot[semantic][2] is True
    assert len(list(manager._refheld_checksums())) == 6
    manager._release_refholds()
    assert cache.reference_snapshot().get(syntactic, (0, 0, False))[0] == 0
    assert cache.reference_snapshot().get(semantic, (0, 0, False))[0] == 0
