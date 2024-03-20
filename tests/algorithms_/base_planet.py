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
        pytest.importorskip("tensorflow_probability")
        pytest.importorskip("pygame")

        from srl.algorithms import planet

        rl_config = planet.Config(
            batch_size=2,
            batch_length=2,
            pred_action_length=2,
            num_generation=2,
            num_individual=2,
            num_simulations=2,
            print_ga_debug=False,
        )
        rl_config.memory.warmup_size = 2

        rl_config.observation_mode = ObservationModes.RENDER_IMAGE
        return rl_config, {}


class BaseCase(CommonBaseCase):
    def _create_rl_config(self):
        from srl.algorithms import planet

        return planet.Config(
            deter_size=30,
            stoch_size=20,
            num_units=30,
            cnn_depth=32,
            batch_size=32,
            batch_length=21,
            lr=0.001,
            free_nats=0.1,
            kl_scale=1.0,
            enable_overshooting_loss=False,
            # GA
            pred_action_length=5,
            num_generation=10,
            num_individual=5,
            num_simulations=5,
            print_ga_debug=False,
        )

    def test_EasyGrid(self):
        self.check_skip()
        env_config = srl.EnvConfig("EasyGrid")
        env_config.max_episode_steps = 20

        rl_config = self._create_rl_config()
        rl_config.memory.warmup_size = rl_config.batch_size + 1
        rl_config.observation_mode = ObservationModes.RENDER_IMAGE

        runner, tester = self.create_runner(env_config, rl_config)

        # --- train
        runner.rollout(max_episodes=1000)
        runner.train_only(max_train_count=5_000)

        # --- eval
        tester.eval(runner, episode=5, baseline=0.2)

    def test_Grid(self):
        self.check_skip()
        env_config = srl.EnvConfig("Grid")
        env_config.max_episode_steps = 20

        rl_config = self._create_rl_config()
        rl_config.memory.warmup_size = rl_config.batch_size + 1
        rl_config.observation_mode = ObservationModes.RENDER_IMAGE

        runner, tester = self.create_runner(env_config, rl_config)

        # train
        runner.rollout(max_episodes=1000)
        runner.train_only(max_train_count=20_000)

        # eval
        tester.eval(runner, baseline=0.1)

    def test_EasyGrid_overshooting(self):
        self.check_skip()
        env_config = srl.EnvConfig("EasyGrid")
        env_config.max_episode_steps = 10

        rl_config = self._create_rl_config()
        rl_config.__init__(
            deter_size=50,
            stoch_size=10,
            num_units=100,
            cnn_depth=32,
            batch_size=32,
            batch_length=11,
            lr=0.001,
            free_nats=0.1,
            kl_scale=1.0,
            enable_overshooting_loss=True,
            # GA
            pred_action_length=5,
            num_generation=20,
            num_individual=5,
            num_simulations=5,
            print_ga_debug=False,
        )
        rl_config.memory.warmup_size = rl_config.batch_size + 1
        rl_config.observation_mode = ObservationModes.RENDER_IMAGE

        runner, tester = self.create_runner(env_config, rl_config)

        # train
        runner.rollout(max_episodes=1000)
        runner.train_only(max_train_count=10_000)

        # eval
        tester.eval(runner, baseline=0.1)
