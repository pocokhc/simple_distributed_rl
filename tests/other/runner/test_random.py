import numpy as np

import srl
import srl.rl.dummy
from srl.envs import grid  # noqa E401
from srl.runner import sequence


def test_play():
    env_config = srl.EnvConfig("Grid")
    rl_config = srl.rl.dummy.Config()

    config = sequence.Config(env_config, rl_config, seed=1, seed_enable_gpu=True)
    parameter, _, _ = sequence.train(config, max_episodes=10, enable_file_logger=False)

    # reward1 == reward3
    # reward2 == reward4
    # reward1 != reward2
    config.seed = 10
    rewards1 = sequence.evaluate(config, parameter, max_episodes=10)
    config.seed = 11
    rewards2 = sequence.evaluate(config, parameter, max_episodes=10)
    config.seed = 10
    rewards3 = sequence.evaluate(config, parameter, max_episodes=10)
    config.seed = 11
    rewards4 = sequence.evaluate(config, parameter, max_episodes=10)
    np.testing.assert_array_equal(rewards1, rewards3)
    np.testing.assert_array_equal(rewards2, rewards4)
    assert not np.allclose(rewards1, rewards2)

    # 2回目
    config = sequence.Config(env_config, rl_config, seed=1, seed_enable_gpu=True)
    parameter, _, _ = sequence.train(config, max_episodes=10)

    config.seed = 10
    rewards5 = sequence.evaluate(config, parameter, max_episodes=10)
    config.seed = 11
    rewards6 = sequence.evaluate(config, parameter, max_episodes=10)
    config.seed = 10
    rewards7 = sequence.evaluate(config, parameter, max_episodes=10)
    config.seed = 11
    rewards8 = sequence.evaluate(config, parameter, max_episodes=10)
    np.testing.assert_array_equal(rewards1, rewards3)
    np.testing.assert_array_equal(rewards2, rewards4)
    assert not np.allclose(rewards1, rewards2)

    np.testing.assert_array_equal(rewards1, rewards5)
    np.testing.assert_array_equal(rewards2, rewards6)
    np.testing.assert_array_equal(rewards3, rewards7)
    np.testing.assert_array_equal(rewards4, rewards8)
