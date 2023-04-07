import unittest

import numpy as np

import srl
from srl import runner
from srl.test import TestRL
from srl.utils.common import is_package_installed

try:
    from srl.algorithms import world_models
    from srl.envs import grid  # noqa F401
except ModuleNotFoundError:
    pass


@unittest.skipUnless(is_package_installed("tensorflow"), "no module")
class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()

    def test_Grid(self):
        rl_config = world_models.Config(
            z_size=1,
            sequence_length=10,
            rnn_units=8,
            num_mixture=3,
            batch_size=64,
        )
        rl_config.change_observation_render_image = True
        env_config = srl.EnvConfig("Grid")
        config = runner.Config(env_config, rl_config)
        config.rl_config.train_mode = 1
        _, memory, _ = runner.train(
            config,
            max_episodes=100,
            disable_trainer=True,
            enable_file_logger=False,
        )

        # vae
        rl_config.train_mode = 1
        rl_config.lr = 0.001
        rl_config.kl_tolerance = 4.0
        parameter, memory, history = runner.train_only(
            config,
            remote_memory=memory,
            max_train_count=20_000,
            enable_file_logger=False,
        )

        # rnn
        rl_config.train_mode = 2
        rl_config.lr = 0.001
        rl_config.memory_warmup_size = 100
        parameter, memory, history = runner.train_only(
            config,
            parameter=parameter,
            remote_memory=memory,
            max_train_count=40_000,
            enable_file_logger=False,
        )

        # controller
        rl_config.train_mode = 3
        rl_config.num_simulations = 10
        rl_config.num_individual = 4
        rl_config.blx_a = 0.3
        max_episodes = rl_config.num_simulations * rl_config.num_individual * 300
        parameter, memory, history = runner.train(
            config,
            parameter=parameter,
            max_episodes=max_episodes,
            enable_file_logger=False,
        )

        rewards = runner.evaluate(config, parameter, max_episodes=200, print_progress=True)
        true_reward = 0.3
        s = f"{np.mean(rewards)} >= {true_reward}"
        print(s)
        assert np.mean(rewards) >= true_reward, s


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_verify_grid", verbosity=2)
