import srl.envs.neongrid  # noqa F401
import srl.envs.oneroad  # noqa F401
from srl.runner import mp, sequence


class TestRL:
    def play_test(self, tester, rl_config):

        env_list = [
            "FrozenLake-v1",
            "OneRoad-v0",
        ]

        for env_name in env_list:
            config = sequence.Config(
                env_name=env_name,
                rl_config=rl_config,
            )
            with tester.subTest(("sequence", env_name, rl_config.getName())):
                self._sequence(config)

            with tester.subTest(("mp", env_name, rl_config.getName())):
                self._mp(config)

    def _sequence(self, config):
        # --- train
        config.set_play_config(timeout=5, training=True)
        episode_rewards, parameter, memory = sequence.play(config)
        # self.assertTrue(np.mean(episode_rewards) > 0.01)

        # --- test
        config.set_play_config(max_episodes=10)
        episode_rewards, _, _ = sequence.play(config, parameter)
        # self.assertTrue(np.mean(episode_rewards) > 0.1)

    def _mp(self, config):
        # --- train
        mp_config = mp.Config(worker_num=2)
        mp_config.set_train_config(timeout=5)
        parameter = mp.train(config, mp_config)
        # self.assertTrue(np.mean(episode_rewards) > 0.01)

        # --- test
        config.set_play_config(max_episodes=10)
        episode_rewards, _, _ = sequence.play(config, parameter)
        # self.assertTrue(np.mean(episode_rewards) > 0.1)
