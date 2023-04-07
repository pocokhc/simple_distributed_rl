import unittest

import numpy as np

import srl
from srl import runner
from srl.test import TestRL
from srl.utils.common import is_package_installed

try:
    from srl.algorithms import planet
    from srl.envs import grid  # noqa F401
except ModuleNotFoundError:
    pass


@unittest.skipUnless(is_package_installed("tensorflow"), "no module")
class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()

    def test_EasyGrid(self):
        rl_config = planet.Config(
            deter_size=50,
            stoch_size=10,
            num_units=100,
            cnn_depth=32,
            batch_size=8,
            batch_length=20,
            lr=0.001,
            free_nats=3.0,
            kl_scale=1.0,
            enable_overshooting_loss=False,
            # GA
            pred_action_length=10,
            num_generation=20,
            num_individual=10,
            num_simulations=5,
            print_ga_debug=False,
        )
        rl_config.memory_warmup_size = rl_config.batch_size + 1
        rl_config.change_observation_render_image = True
        env_config = srl.EnvConfig("EasyGrid")
        env_config.max_episode_steps = 21
        config = runner.Config(env_config, rl_config)

        # train
        _, memory, _ = runner.train(
            config,
            max_episodes=1000,
            enable_file_logger=False,
            disable_trainer=True,
        )
        parameter, _, _ = runner.train_only(
            config,
            remote_memory=memory,
            max_train_count=10_000,
            enable_evaluation=False,
            enable_file_logger=False,
            progress_max_time=60 * 2,
        )

        # eval
        rewards = runner.evaluate(config, parameter, max_episodes=10, print_progress=True)
        true_reward = 0.5
        s = f"{np.mean(rewards)} >= {true_reward}"
        print(s)
        assert np.mean(rewards) >= true_reward, s

    def test_Grid(self):
        rl_config = planet.Config(
            deter_size=50,
            stoch_size=10,
            num_units=100,
            cnn_depth=32,
            batch_size=8,
            batch_length=20,
            lr=0.001,
            free_nats=3.0,
            kl_scale=1.0,
            enable_overshooting_loss=False,
            # GA
            pred_action_length=10,
            num_generation=20,
            num_individual=10,
            num_simulations=5,
            print_ga_debug=False,
        )
        rl_config.memory_warmup_size = rl_config.batch_size + 1
        rl_config.change_observation_render_image = True
        env_config = srl.EnvConfig("Grid")
        env_config.max_episode_steps = 21
        config = runner.Config(env_config, rl_config)

        # train
        _, memory, _ = runner.train(
            config,
            max_episodes=1000,
            enable_file_logger=False,
            disable_trainer=True,
        )
        parameter, _, _ = runner.train_only(
            config,
            remote_memory=memory,
            max_train_count=40_000,
            enable_evaluation=False,
            enable_file_logger=False,
            progress_max_time=60 * 2,
        )

        # eval
        rewards = runner.evaluate(config, parameter, max_episodes=10, print_progress=True)
        true_reward = 0.1
        s = f"{np.mean(rewards)} >= {true_reward}"
        print(s)
        assert np.mean(rewards) >= true_reward, s

    def test_Grid_overshooting(self):
        rl_config = planet.Config(
            deter_size=50,
            stoch_size=10,
            num_units=100,
            cnn_depth=32,
            batch_size=8,
            batch_length=20,
            lr=0.001,
            free_nats=3.0,
            kl_scale=1.0,
            enable_overshooting_loss=True,
            # GA
            pred_action_length=10,
            num_generation=20,
            num_individual=10,
            num_simulations=5,
            print_ga_debug=False,
        )
        rl_config.memory_warmup_size = rl_config.batch_size + 1
        rl_config.change_observation_render_image = True
        env_config = srl.EnvConfig("Grid")
        env_config.max_episode_steps = 21
        config = runner.Config(env_config, rl_config)

        # train
        _, memory, _ = runner.train(
            config,
            max_episodes=1000,
            enable_file_logger=False,
            disable_trainer=True,
        )
        parameter, _, _ = runner.train_only(
            config,
            remote_memory=memory,
            max_train_count=40_000,
            enable_evaluation=False,
            enable_file_logger=False,
            progress_max_time=60 * 2,
        )

        # eval
        rewards = runner.evaluate(config, parameter, max_episodes=10, print_progress=True)
        true_reward = 0.1
        s = f"{np.mean(rewards)} >= {true_reward}"
        print(s)
        assert np.mean(rewards) >= true_reward, s


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_verify_grid_overshooting", verbosity=2)
