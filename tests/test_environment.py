import pytest

from seamless_transformer import Environment


def test_environment_empty_save_and_lowlevel():
    env = Environment()
    assert env._save() is None
    assert env._to_lowlevel() is None


def test_environment_roundtrip_all_props():
    env = Environment()
    env.set_conda("name: test\ndependencies:\n  - python\n")
    env.set_conda_env("seamless1")
    env.set_which(["python", "gcc"])
    env.set_powers({"docker": True})
    env.set_docker({"name": "python:3.12"})

    state = env._save()
    restored = Environment()
    restored._load(state)

    assert restored._save() == state
    lowlevel = restored._to_lowlevel()
    assert lowlevel["conda_environment"] == "seamless1"
    assert lowlevel["conda"]["dependencies"] == ["python"]
    assert lowlevel["which"] == ["python", "gcc"]
    assert lowlevel["docker"] == {"name": "python:3.12"}


def test_environment_validation():
    env = Environment()
    with pytest.raises(TypeError):
        env.set_which("python")
    with pytest.raises(ValueError):
        env.set_docker({"image": "python"})
    with pytest.raises(ValueError):
        env.set_conda("name: test\n")
