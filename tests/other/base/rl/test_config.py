import srl
from srl.envs import grid  # noqa F401
from srl.rl import dummy


def test_copy():
    config = dummy.Config()

    assert not config._is_set_env_config
    config.reset(srl.make_env("Grid"))
    assert config._is_set_env_config
    config.window_length = 2
    assert not config._is_set_env_config
    config.reset(srl.make_env("Grid"))
    assert config._is_set_env_config

    config2 = config.copy()
    assert config._is_set_env_config
    assert config.window_length == 2
    assert config2.name == "Dummy"
