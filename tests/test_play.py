import unittest

import numpy as np
import srl.envs.neongrid
import srl.envs.oneroad
from srl import rl
from srl.runner import sequence


class TestRL(unittest.TestCase):
    def test_rl(self):

        env_list = [
            "FrozenLake-v1",
            "Grid-v0",
            "OneRoad-v0",
        ]

        rl_list = [
            (rl.ql.Config(), 100),
            (rl.dqn.Config(), 10),
        ]

        for env_name in env_list:
            for rl_config, max_episodes in rl_list:
                config = sequence.Config(
                    env_name=env_name,
                    rl_config=rl_config,
                    memory_config=rl.memory.replay_memory.Config(),
                )
                with self.subTest((env_name, rl_config.getName())):
                    # --- train
                    config.set_play_config(max_episodes=max_episodes, training=True)
                    episode_rewards, parameter, memory = sequence.play(config)
                    # self.assertTrue(np.mean(episode_rewards) > 0.01)

                    # test
                    config.set_play_config(max_episodes=10)
                    episode_rewards, _, _ = sequence.play(config, parameter)
                    # self.assertTrue(np.mean(episode_rewards) > 0.1)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="TestRL.test_rl", verbosity=2)
