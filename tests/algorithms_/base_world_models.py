from typing import Tuple

from srl.base.define import ObservationModes
from srl.base.rl.config import RLConfig
from tests.algorithms_.common_long_case import CommonLongCase
from tests.algorithms_.common_quick_case import CommonQuickCase


class QuickCase(CommonQuickCase):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        from srl.algorithms import world_models

        rl_config = world_models.Config()
        return rl_config, {}


class LongCase(CommonLongCase):
    def _create_rl_config(self):
        self.check_test_skip()

        from srl.algorithms import world_models

        return world_models.Config(
            z_size=1,
            sequence_length=10,
            rnn_units=8,
            num_mixture=3,
            batch_size=64,
        )

    def test_Grid(self):
        rl_config = self._create_rl_config()
        rl_config.observation_mode = ObservationModes.RENDER_IMAGE

        runner = self.create_test_runner("Grid", rl_config)

        rl_config.train_mode = 1
        runner.rollout(max_episodes=100)

        # vae
        rl_config.train_mode = 1
        rl_config.lr = 0.001
        rl_config.kl_tolerance = 4.0
        runner.train_only(max_train_count=20_000)

        # rnn
        rl_config.train_mode = 2
        rl_config.lr = 0.001
        rl_config.memory_warmup_size = 100
        runner.train_only(max_train_count=40_000)

        # controller
        rl_config.train_mode = 3
        rl_config.num_simulations = 10
        rl_config.num_individual = 4
        rl_config.blx_a = 0.3
        max_episodes = rl_config.num_simulations * rl_config.num_individual * 300
        runner.train(max_episodes=max_episodes)

        assert runner.evaluate_compare_to_baseline_single_player(episode=10, baseline=0.3)
