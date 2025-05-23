from typing import Tuple

import pytest

from srl.base.rl.config import RLConfig
from tests.algorithms_.common_long_case import CommonLongCase
from tests.algorithms_.common_quick_case import CommonQuickCase


class QuickCase(CommonQuickCase):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        from srl.algorithms import muzero

        rl_config = muzero.Config()
        rl_config.set_atari_config()

        rl_config.batch_size = 2
        rl_config.memory.warmup_size = 2
        rl_config.num_simulations = 2
        rl_config.unroll_steps = 2
        rl_config.input_image_block.set_alphazero_block(1, 2)
        rl_config.dynamics_blocks = 1

        return rl_config, dict(use_layer_processor=True)

    def test_simple_input_image(self, rl_param, tmpdir):
        pytest.skip()


class LongCase(CommonLongCase):
    def _create_rl_config(self):
        self.check_test_skip()
        from srl.algorithms import muzero

        rl_config = muzero.Config(
            batch_size=16,
        )
        rl_config.memory.warmup_size = 50

        return rl_config

    def test_EasyGrid(self):
        rl_config = self._create_rl_config()
        rl_config.__init__(
            num_simulations=50,
            discount=0.9,
            batch_size=32,
            lr=0.001,
            reward_range=(-2, 2),
            reward_range_num=10,
            value_range=(-10, 10),
            value_range_num=100,
            unroll_steps=3,
            dynamics_blocks=1,
            enable_rescale=False,
            weight_decay=0,
        )
        rl_config.input_image_block.set_alphazero_block(1, 16)
        rl_config.memory.warmup_size = 100
        rl_config.memory.set_proportional()
        runner = self.create_test_runner("EasyGrid-layer", rl_config)
        runner.train(max_train_count=10000)
        assert runner.evaluate_compare_to_baseline_single_player(episode=5)
