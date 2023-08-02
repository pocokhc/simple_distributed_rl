import numpy as np

import srl
import srl.rl.dummy
from srl.runner.runner import Runner


def test_play():
    env_config = srl.EnvConfig("Grid")
    rl_config = srl.rl.dummy.Config()

    runner = Runner(env_config, rl_config)
    runner.set_seed(seed=1)
    runner.train(max_episodes=10)

    # reward1 == reward3
    # reward2 == reward4
    # reward1 != reward2
    runner.set_seed(seed=10)
    rewards1 = runner.evaluate(max_episodes=10)
    runner.set_seed(seed=11)
    rewards2 = runner.evaluate(max_episodes=10)
    runner.set_seed(seed=10)
    rewards3 = runner.evaluate(max_episodes=10)
    runner.set_seed(seed=11)
    rewards4 = runner.evaluate(max_episodes=10)
    np.testing.assert_array_equal(rewards1, rewards3)
    np.testing.assert_array_equal(rewards2, rewards4)
    assert not np.allclose(rewards1, rewards2)

    # 2回目
    runner = Runner(env_config, rl_config)
    runner.set_seed(seed=1)
    runner.train(max_episodes=10)

    runner.set_seed(seed=10)
    rewards5 = runner.evaluate(max_episodes=10)
    runner.set_seed(seed=11)
    rewards6 = runner.evaluate(max_episodes=10)
    runner.set_seed(seed=10)
    rewards7 = runner.evaluate(max_episodes=10)
    runner.set_seed(seed=11)
    rewards8 = runner.evaluate(max_episodes=10)
    np.testing.assert_array_equal(rewards1, rewards3)
    np.testing.assert_array_equal(rewards2, rewards4)
    assert not np.allclose(rewards1, rewards2)

    np.testing.assert_array_equal(rewards1, rewards5)
    np.testing.assert_array_equal(rewards2, rewards6)
    np.testing.assert_array_equal(rewards3, rewards7)
    np.testing.assert_array_equal(rewards4, rewards8)
