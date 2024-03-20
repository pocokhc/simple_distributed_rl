import srl
from srl.algorithms import ql
from srl.envs import grid


def test_get_env_init_state():
    env_config = srl.EnvConfig("Grid")
    rl_config = ql.Config()
    rl_config.processors = [grid.LayerProcessor()]
    runner = srl.Runner(env_config, rl_config)

    env_state = runner.get_env_init_state(encode=False)
    print(env_state)
    assert env_state == [1, 3]

    rl_state = runner.get_env_init_state(encode=True)
    print(rl_state)
    assert isinstance(rl_state, list)
    assert len(rl_state) == 30
    assert rl_state[19] == 1
