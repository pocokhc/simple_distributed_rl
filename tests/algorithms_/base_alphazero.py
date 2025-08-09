from typing import Tuple

import pytest

from srl.base.rl.config import RLConfig
from tests.algorithms_.common_long_case import CommonLongCase
from tests.algorithms_.common_quick_case import CommonQuickCase


class QuickCase(CommonQuickCase):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        from srl.algorithms import alphazero

        rl_config = alphazero.Config()
        rl_config.set_go_config()

        rl_config.num_simulations = 2
        rl_config.sampling_steps = 2
        rl_config.memory.warmup_size = 2
        rl_config.batch_size = 2
        rl_config.input_block.set_alphazero_block(1, 2)
        rl_config.value_block.set((2, 2))

        return rl_config, dict(use_layer_processor=True)

    def test_simple_input_image(self, rl_param, tmpdir):
        pytest.skip()


class LongCase(CommonLongCase):
    def _create_rl_config(self):
        self.check_test_skip()

        from srl.algorithms import alphazero

        rl_config = alphazero.Config(
            num_simulations=100,
            sampling_steps=1,
            batch_size=32,
            discount=1.0,
        )
        rl_config.lr_scheduler.set_piecewise(
            [100, 1100],
            [0.02, 0.002, 0.0002],
        )
        rl_config.input_block.set_alphazero_block(1, 32)
        rl_config.value_block.set((32,))
        return rl_config

    def test_Grid(self):
        rl_config = self._create_rl_config()
        rl_config.discount = 0.9
        rl_config.num_simulations = 50
        rl_config.memory.warmup_size = 100
        runner = self.create_test_runner("Grid-layer", rl_config)
        runner.train(max_train_count=500)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_OX(self):
        rl_config = self._create_rl_config()
        rl_config.value_type = "rate"
        rl_config.memory.warmup_size = 100
        runner = self.create_test_runner("OX-layer", rl_config)
        runner.train(max_train_count=200)

        results = runner.evaluate_compare_to_baseline_multiplayer()
        assert all(results)

    def test_OX_mp(self):
        rl_config = self._create_rl_config()
        rl_config.value_type = "rate"
        rl_config.memory.warmup_size = 100
        runner = self.create_test_runner("OX-layer", rl_config)
        runner.set_seed(2)
        runner.train_mp(max_train_count=300)

        results = runner.evaluate_compare_to_baseline_multiplayer()
        assert all(results)

    def test_Othello4x4(self):
        rl_config = self._create_rl_config()
        rl_config.value_type = "rate"
        rl_config.batch_size = 32
        rl_config.memory.warmup_size = 500
        rl_config.lr_scheduler.set_piecewise(
            [1000, 6000],
            [0.001, 0.0005, 0.0002],
        )
        rl_config.input_block.set_alphazero_block(9, 32)
        rl_config.value_block.set((16, 16))
        rl_config.policy_block.set((32,))

        runner = self.create_test_runner("Othello4x4-layer", rl_config)
        runner.train(max_train_count=20_000)

        results = runner.evaluate_compare_to_baseline_multiplayer()
        assert all(results)
