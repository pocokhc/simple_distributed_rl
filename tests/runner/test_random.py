import unittest

import numpy as np
import srl
import srl.rl.dummy
from envs import grid  # noqa E401
from srl.runner import sequence


class Test(unittest.TestCase):
    def test_play(self):

        env_config = srl.EnvConfig("Grid")
        rl_config = srl.rl.dummy.Config()

        config = sequence.Config(env_config, rl_config)
        parameter, _, _ = sequence.train(config, max_episodes=10, seed=1, enable_file_logger=False)

        # reward1 == reward3
        # reward2 == reward4
        # reward1 != reward2
        rewards1 = sequence.evaluate(config, parameter, max_episodes=10, seed=10)
        rewards2 = sequence.evaluate(config, parameter, max_episodes=10)
        rewards3 = sequence.evaluate(config, parameter, max_episodes=10, seed=10)
        rewards4 = sequence.evaluate(config, parameter, max_episodes=10)
        np.testing.assert_array_equal(rewards1, rewards3)
        np.testing.assert_array_equal(rewards2, rewards4)
        self.assertFalse(np.allclose(rewards1, rewards2))

        # 2回目
        config = sequence.Config(env_config, rl_config)
        parameter, _, _ = sequence.train(config, max_episodes=10, seed=1)

        rewards5 = sequence.evaluate(config, parameter, max_episodes=10, seed=10)
        rewards6 = sequence.evaluate(config, parameter, max_episodes=10)
        rewards7 = sequence.evaluate(config, parameter, max_episodes=10, seed=10)
        rewards8 = sequence.evaluate(config, parameter, max_episodes=10)
        np.testing.assert_array_equal(rewards1, rewards3)
        np.testing.assert_array_equal(rewards2, rewards4)
        self.assertFalse(np.allclose(rewards1, rewards2))

        np.testing.assert_array_equal(rewards1, rewards5)
        np.testing.assert_array_equal(rewards2, rewards6)
        np.testing.assert_array_equal(rewards3, rewards7)
        np.testing.assert_array_equal(rewards4, rewards8)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_play", verbosity=2)
