"""Early bash-only environment checks and legacy environment compatibility."""
import uuid

import pytest

from seamless import Buffer
from seamless_transformer import Environment, Transformer


@pytest.mark.parametrize('language', ['python', 'c'])
@pytest.mark.parametrize('environment', ['docker', 'conda'])
def test_in_memory_environment_is_checked_when_submitted(language, environment):
    tf = Transformer(language, compiled=language == 'c')
    nonce = uuid.uuid4().hex
    if language == 'python':
        tf.code = '# ' + nonce + '\nresult = 42'
    else:
        tf.schema = 'inputs: []\noutputs:\n  - {name: result, dtype: int32}\n'
        tf.code = ('/* ' + nonce + ' */\n#include <stdint.h>\n'
                   'int transform(int32_t *result) { *result = 42; return 0; }')
    if environment == 'docker':
        tf.environment.set_docker({'name': 'never-launched:' + nonce})
    else:
        tf.environment.set_conda_env('never-created-' + nonce)
    handle = tf.build()
    assert handle.construct() is not None
    result = handle.compute()
    assert result is None
    message = (handle.exception or '').lower()
    assert 'docker image' in message or 'requires conda environment' in message


def test_legacy_powers_saved_state_loads_but_is_not_saved_or_emitted():
    env = Environment()
    env._load({'powers': {'docker': True}, 'which': ['bash']})
    assert env.get_which() == ['bash']
    assert env._save() == {'which': ['bash']}
    assert env._to_lowlevel() == {'which': ['bash']}
    assert not hasattr(env, 'set_powers')
    assert not hasattr(env, 'get_powers')


def test_legacy_powers_lowlevel_key_does_not_prevent_execution():
    from seamless_transformer.run import run_transformation_dict
    nonce = uuid.uuid4().hex
    code = Buffer('# ' + nonce + '\nresult = 42', 'text')
    environment = Buffer({'powers': {'docker': True}}, 'plain')
    code.tempref()
    environment.tempref()
    transformation = {
        '__language__': 'python',
        'code': ('text', None, code.get_checksum().hex()),
        '__output__': ('result', 'mixed', None),
    }
    checksum = Buffer(transformation, 'plain').get_checksum()
    result = run_transformation_dict(transformation, checksum,
        {'__env__': environment.get_checksum().hex()}, scratch=True, require_value=True)
    # Successful execution, rather than merely accepting an unrecognized key.
    assert result == Buffer(42, 'mixed').get_checksum()
