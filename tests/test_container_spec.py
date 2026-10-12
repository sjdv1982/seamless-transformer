"""Container specification is checked on every submission face."""
import subprocess
import os
import uuid

import pytest

from seamless_transformer import Transformer
from seamless_transformer.container import validate_container_spec


VALID = [
    {'name': 'ubuntu:22.04'},
    {'name': 'ubuntu@sha256:' + 'a' * 64},
    {'name': 'image', 'gpus': False},
    {'name': 'image', 'gpus': True, 'shm_size': '1g'},
]
INVALID = [
    ({}, 'name'), ({'name': 42}, 'name'), ({'name': ''}, 'name'),
    ({'name': 'image', 'gpus': 'yes'}, 'gpus'),
    ({'name': 'image', 'shm_size': 42}, 'shm_size'),
    ({'name': 'image', 'options': {}}, 'shm_size.*gpus|gpus.*shm_size'),
    ({'name': 'image', 'version': 'latest'}, 'name@sha256'),
    ({'name': 'image', 'checksum': 'abc'}, 'name@sha256'),
    ({'name': 'image', 'privileged': True}, 'privileged'),
]


@pytest.mark.parametrize('spec', VALID)
def test_valid_image_spec_setter_and_validator(spec):
    tf = Transformer('bash')
    tf.environment.set_docker(spec)
    assert tf.environment.get_docker() == spec
    validated = validate_container_spec(spec)
    assert all(validated[key] == value for key, value in spec.items())
    spec['name'] = 'changed'
    assert tf.environment.get_docker()['name'] != 'changed'
    tf.environment.set_docker(None)
    assert tf.environment.get_docker() is None


@pytest.mark.parametrize('spec,message', INVALID)
def test_invalid_image_spec_is_rejected_by_setter_and_validator(spec, message):
    with pytest.raises((ValueError, TypeError), match=message):
        validate_container_spec(spec)
    with pytest.raises((ValueError, TypeError), match=message):
        Transformer('bash').environment.set_docker(spec)


def raw_run(environment, code='printf 42 > RESULT'):
    from seamless import Buffer
    from seamless_transformer.run import run_transformation_dict
    buffers = [Buffer('# ' + uuid.uuid4().hex + '\n' + code, 'text'),
               Buffer(environment, 'plain')]
    for buffer in buffers:
        buffer.tempref()
    transformation = {'__language__': 'bash',
                      'code': ('text', None, buffers[0].get_checksum().hex()),
                      '__output__': ('result', 'mixed', None)}
    checksum = Buffer(transformation, 'plain').get_checksum()
    return run_transformation_dict(transformation, checksum,
        {'__env__': buffers[1].get_checksum().hex()}, scratch=True, require_value=True)


@pytest.mark.parametrize('spec,message', INVALID)
def test_raw_dict_validates_image_before_launch(spec, message):
    with pytest.raises((ValueError, TypeError, RuntimeError), match=message):
        raw_run({'docker': spec})


def test_two_environments_refused_when_transformer_is_frozen():
    tf = Transformer('bash')
    tf.code = 'printf 42 > RESULT'
    tf.environment.set_docker({'name': 'image'})
    tf.environment.set_conda_env('seamless1')
    with pytest.raises(Exception, match='pick one|cannot be used together'):
        tf.transformation()


def test_two_environments_refused_on_raw_dict():
    with pytest.raises(RuntimeError, match='pick one|cannot be used together'):
        raw_run({'docker': {'name': 'image'}, 'conda_environment': 'seamless1'})


def test_two_environments_refused_on_unix_face(tmp_path):
    sentinel = tmp_path / 'executed'
    env = os.environ.copy()
    env['SEAMLESS_CACHE'] = str(tmp_path / 'cache')
    completed = subprocess.run(['seamless-run', '--conda', 'seamless1',
        '--docker', 'image', 'touch', str(sentinel)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert completed.returncode != 0
    assert 'conda' in completed.stderr.lower() and 'docker' in completed.stderr.lower()
    assert not sentinel.exists()
