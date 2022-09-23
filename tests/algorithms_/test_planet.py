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

    def test_simple_check(self):
        rl_config = planet.Config(num_generation=2, num_individual=2, num_simulations=1)
        rl_config.change_observation_render_image = True
        self.tester.simple_check(rl_config)

    def test_simple_check_mp(self):
        rl_config = planet.Config(num_generation=2, num_individual=2, num_simulations=1)
        rl_config.change_observation_render_image = True
        self.tester.simple_check_mp(rl_config)

    def test_verify_grid(self):
        rl_config = planet.Config(
            sequence_length=10,
            z_size=2,
            rnn_units=64,
            batch_size=16,
            lr=0.001,
            hidden_block_kwargs={"hidden_layer_sizes": (8, 8)},
            reward_block_kwargs={"hidden_layer_sizes": (16, 16)},
            vae_beta=1000.0,
            enable_overshooting_loss=False,
            pred_action_length=3,
            num_generation=20,
            num_individual=5,
            num_simulations=10,
            print_ga_debug=False,
        )
        rl_config.memory_warmup_size = rl_config.batch_size + 1
        rl_config.change_observation_render_image = True
        env_config = srl.EnvConfig("Grid")
        env_config.max_episode_steps = 10
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
            max_train_count=50_000,
            enable_evaluation=False,
            enable_file_logger=False,
            progress_max_time=60 * 2,
        )

        # eval
        rewards = runner.evaluate(config, parameter, max_episodes=5, print_progress=True)
        true_reward = -0.1
        s = f"{np.mean(rewards)} >= {true_reward}"
        print(s)
        assert np.mean(rewards) >= true_reward, s


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_verify_grid", verbosity=2)
