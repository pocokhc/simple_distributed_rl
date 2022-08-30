import unittest

import numpy as np
import srl
from srl.runner import sequence


class Test(unittest.TestCase):
    def test_shuffle_player(self):

        env_config = srl.EnvConfig("OX")
        rl_config = None
        config = sequence.Config(env_config, rl_config)
        config.players = ["cpu", "random"]

        # shuffle した状態でも報酬は元の順序を継続する
        rewards = sequence.evaluate(config, parameter=None, max_episodes=100, shuffle_player=True)
        rewards = np.mean(rewards, axis=0)
        self.assertTrue(rewards[0] > 0.7)  # CPUがまず勝つ


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_shuffle_player", verbosity=2)
