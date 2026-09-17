from seamless import Buffer
import pytest
from seamless.caching.buffer_cache import get_buffer_cache
from seamless.reference_lifecycle import collect_refholder_claims
from seamless_transformer.code_manager import CodeManager


def _state(manager, semantic, syntactic):
    cache = get_buffer_cache()
    claims = collect_refholder_claims([manager])
    return {
        "syntactic": cache.reference_snapshot().get(syntactic, (0, 0, False)),
        "semantic": cache.reference_snapshot().get(semantic, (0, 0, False)),
        "claims": {
            "syntactic": [role for _, role in claims.get(syntactic, [])],
            "semantic": [role for _, role in claims.get(semantic, [])],
        },
    }


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


def test_semantic_direct_release_is_balanced_per_multiplicity():
    manager = CodeManager()
    code = Buffer(b"def g(x):\n    return x\n", "python")
    semantic, syntactic = manager.track_code_buffer(code)
    manager.incref_semantic(semantic)
    manager.incref_semantic(semantic)
    cache = get_buffer_cache()
    assert cache.reference_snapshot()[semantic][0] == 2
    manager.decref_semantic(semantic)
    assert cache.reference_snapshot()[semantic][0] == 1
    manager.decref_semantic(semantic)
    assert cache.reference_snapshot().get(semantic, (0, 0, False))[0] == 0
    manager._release_refholds()


def test_syntactic_direct_and_semantic_guard_multiplicity_agree_after_each_transition():
    manager = CodeManager()
    code = Buffer("def transition(x):\n    return x\n", "python")
    semantic, syntactic = manager.track_code_buffer(code)

    manager.incref_syntactic(syntactic)
    state = _state(manager, semantic, syntactic)
    assert state["syntactic"][:1] == (1,)
    assert state["semantic"][:1] == (1,)
    assert state["claims"]["syntactic"] == ["syntactic:direct"]
    assert state["claims"]["semantic"] == ["semantic:guard"]

    manager.incref_syntactic(syntactic)
    manager.incref_semantic(semantic)
    state = _state(manager, semantic, syntactic)
    assert state["syntactic"][0] == 3
    assert state["semantic"][0] == 3
    assert state["claims"]["syntactic"] == [
        "syntactic:direct",
        "syntactic:direct",
        "syntactic:guard",
    ]
    assert state["claims"]["semantic"] == [
        "semantic:direct",
        "semantic:guard",
        "semantic:guard",
    ]

    manager.decref_semantic(semantic)
    manager.decref_syntactic(syntactic)
    state = _state(manager, semantic, syntactic)
    assert state["syntactic"][0] == 1
    assert state["semantic"][0] == 1
    assert state["claims"]["syntactic"] == ["syntactic:direct"]
    assert state["claims"]["semantic"] == ["semantic:guard"]
    manager.decref_syntactic(syntactic)
    manager._release_refholds()


def test_multiple_syntactic_variants_share_semantic_guard_demand():
    manager = CodeManager()
    first = Buffer("def variant(x):\n    return x\n", "python")
    second = Buffer("def variant(x):\n    return (x)\n", "python")
    semantic_first, syntactic_first = manager.track_code_buffer(first)
    semantic_second, syntactic_second = manager.track_code_buffer(second)
    assert semantic_first == semantic_second
    assert syntactic_first != syntactic_second

    manager.incref_semantic(semantic_first)
    state = _state(manager, semantic_first, syntactic_first)
    assert state["semantic"][0] == 1
    assert state["claims"]["semantic"] == ["semantic:direct"]
    assert state["syntactic"][0] == 1
    assert state["claims"]["syntactic"] == ["syntactic:guard"]
    assert get_buffer_cache().reference_snapshot()[syntactic_second][0] == 1
    assert collect_refholder_claims([manager])[syntactic_second][0][1] == "syntactic:guard"

    manager.decref_semantic(semantic_first)
    assert get_buffer_cache().reference_snapshot().get(semantic_first, (0, 0, False))[0] == 0
    assert get_buffer_cache().reference_snapshot().get(syntactic_first, (0, 0, False))[0] == 0
    assert get_buffer_cache().reference_snapshot().get(syntactic_second, (0, 0, False))[0] == 0
    manager._release_refholds()


def test_release_orders_and_unknown_decrements_are_explicit():
    manager = CodeManager()
    code = Buffer("def order(x):\n    return x\n", "python")
    semantic, syntactic = manager.track_code_buffer(code)
    manager.incref_syntactic(syntactic)
    manager.incref_semantic(semantic)
    manager.decref_semantic(semantic)
    manager.decref_syntactic(syntactic)
    assert get_buffer_cache().reference_snapshot().get(semantic, (0, 0, False))[0] == 0
    assert get_buffer_cache().reference_snapshot().get(syntactic, (0, 0, False))[0] == 0
    with pytest.raises(KeyError):
        manager.decref_semantic(semantic)
    with pytest.raises(KeyError):
        manager.decref_syntactic(syntactic)
    manager._release_refholds()
    manager._release_refholds()
