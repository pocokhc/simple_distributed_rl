import numpy as np

import srl
from srl.algorithms import ql
from srl.envs import grid


def test_get_env_init_state():
    env_config = srl.EnvConfig("Grid", enable_sanitize=False, enable_assertion=False)
    env_config.processors = [grid.LayerProcessor()]
    rl_config = ql.Config()
    runner = srl.Runner(env_config, rl_config)

    env_state = runner.get_env_init_state(encode=False)
    print(env_state)
    assert isinstance(env_state, np.ndarray)
    assert env_state.shape == (5, 6, 1)
    assert env_state[3][1][0] == 1

    rl_state = runner.get_env_init_state(encode=True)
    print(rl_state)
    assert isinstance(rl_state, list)
    assert len(rl_state) == 30
    assert rl_state[19] == 1
