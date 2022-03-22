import unittest

import srl.envs.neongrid  # noqa F401
import srl.envs.oneroad  # noqa F401
from srl import rl
from srl.runner import mp, sequence


class TestPlay(unittest.TestCase):
    def test_play(self):

        env_list = [
            "FrozenLake-v1",
            "Grid-v0",
            "OneRoad-v0",
        ]

        rl_list = [
            rl.ql.Config(),
            rl.ql_agent57.Config(multisteps=3),
            rl.dqn.Config(),
            rl.c51.Config(),
            rl.rainbow.Config(),
        ]

        for env_name in env_list:
            for rl_config in rl_list:
                config = sequence.Config(
                    env_name=env_name,
                    rl_config=rl_config,
                )
                with self.subTest((env_name, rl_config.getName())):
                    self._sequence(config)

                with self.subTest((env_name, rl_config.getName())):
                    self._mq(config)

    def _sequence(self, config):
        # --- train
        config.set_play_config(timeout=5, training=True)
        episode_rewards, parameter, memory = sequence.play(config)
        # self.assertTrue(np.mean(episode_rewards) > 0.01)

        # --- test
        config.set_play_config(max_episodes=10)
        episode_rewards, _, _ = sequence.play(config, parameter)
        # self.assertTrue(np.mean(episode_rewards) > 0.1)

    def _mq(self, config):
        # --- train
        mp_config = mp.Config(worker_num=2)
        mp_config.set_train_config(timeout=5)
        parameter = mp.train(config, mp_config)
        # self.assertTrue(np.mean(episode_rewards) > 0.01)

        # --- test
        config.set_play_config(max_episodes=10)
        episode_rewards, _, _ = sequence.play(config, parameter)
        # self.assertTrue(np.mean(episode_rewards) > 0.1)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="TestPlay.test_play", verbosity=2)
