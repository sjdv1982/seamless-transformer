import pytest

from seamless_transformer import Environment


def test_environment_empty_save_and_lowlevel():
    env = Environment()
    assert env._save() is None
    assert env._to_lowlevel() is None


def test_environment_roundtrip_all_props():
    for container in (False, True):
        env = Environment()
        env.set_conda("name: test\ndependencies:\n  - python\n")
        env.set_which(["python", "gcc"])
        if container:
            env.set_docker({"name": "python:3.12"})
        else:
            env.set_conda_env("seamless1")

        state = env._save()
        restored = Environment()
        restored._load(state)

        assert restored._save() == state
        lowlevel = restored._to_lowlevel()
        assert lowlevel["conda"]["dependencies"] == ["python"]
        assert lowlevel["which"] == ["python", "gcc"]
        if container:
            assert lowlevel["docker"] == {"name": "python:3.12"}
            assert "conda_environment" not in lowlevel
        else:
            assert lowlevel["conda_environment"] == "seamless1"
            assert "docker" not in lowlevel


def test_environment_validation():
    env = Environment()
    with pytest.raises(TypeError):
        env.set_which("python")
    invalid_docker = {"image": "python"}
    env.set_docker(invalid_docker)
    assert env.get_docker() == invalid_docker
    from seamless_transformer.container import validate_container_spec
    with pytest.raises(ValueError, match="name|Unknown Docker"):
        validate_container_spec(env.get_docker())
    with pytest.raises(ValueError):
        env.set_conda("name: test\n")
