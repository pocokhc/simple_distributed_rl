import unittest

import numpy as np
import srl
from algorithms import ql_agent57
from envs import grid
from srl import runner


class Test(unittest.TestCase):
    def test_shuffle_player(self):

        env_config = srl.EnvConfig("OX")
        config = runner.Config(env_config, None)
        config.players = ["cpu", "random"]

        # shuffle した状態でも報酬は元の順序を継続する
        rewards = runner.evaluate(config, parameter=None, max_episodes=100, shuffle_player=True)
        rewards = np.mean(rewards, axis=0)
        self.assertTrue(rewards[0] > 0.7)  # CPUがまず勝つ

    def test_modelbase_train_flow(self):
        env_config = srl.EnvConfig("Grid")
        rl_config = ql_agent57.Config()
        config = runner.Config(env_config, rl_config)

        _, memory, _ = runner.train(config, max_steps=10_000, disable_trainer=True)

        rl_config.memory_warmup_size = 1000
        parameter, _, _ = runner.train_only(config, remote_memory=memory, max_train_count=100_000)

        rewards = runner.evaluate(config, parameter, max_episodes=100)
        reward = np.mean(rewards)
        self.assertTrue(reward > 0.5, f"reward: {reward}")


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_modelbase_train_flow", verbosity=2)
