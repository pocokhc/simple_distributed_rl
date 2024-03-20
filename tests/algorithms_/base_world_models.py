from typing import Tuple

import pytest

import srl
from srl.base.define import ObservationModes
from srl.base.rl.config import RLConfig
from tests.algorithms_.common_base_case import CommonBaseCase
from tests.algorithms_.common_quick_case import CommonQuickCase


class QuickCase(CommonQuickCase):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")

        from srl.algorithms import world_models

        rl_config = world_models.Config()
        return rl_config, {}


class BaseCase(CommonBaseCase):
    def _create_rl_config(self):
        from srl.algorithms import world_models

        return world_models.Config(
            z_size=1,
            sequence_length=10,
            rnn_units=8,
            num_mixture=3,
            batch_size=64,
        )

    def test_Grid(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.observation_mode = ObservationModes.RENDER_IMAGE

        env_config = srl.EnvConfig("Grid")
        runner, tester = self.create_runner(env_config, rl_config)

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
        rl_config.memory.warmup_size = 100
        runner.train_only(max_train_count=40_000)

        # controller
        rl_config.train_mode = 3
        rl_config.num_simulations = 10
        rl_config.num_individual = 4
        rl_config.blx_a = 0.3
        max_episodes = rl_config.num_simulations * rl_config.num_individual * 300
        runner.train(max_episodes=max_episodes)

        tester.eval(runner, episode=200, baseline=0.3)
