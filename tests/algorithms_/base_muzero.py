from typing import Tuple

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
            num_simulations=20,
            discount=0.9,
            batch_size=16,
            v_min=-2,
            v_max=2,
            unroll_steps=1,
            dynamics_blocks=1,
            enable_rescale=False,
            weight_decay=0,
        )
        rl_config.lr = 0.001
        rl_config.input_image_block.set_alphazero_block(1, 16)
        rl_config.memory.warmup_size = 200
        rl_config.memory.set_replay_buffer()
        runner = self.create_test_runner("EasyGrid-Layer", rl_config)
        runner.train(max_train_count=3000)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_EasyGrid_PER(self):
        from srl.envs import grid

        rl_config = self._create_rl_config()
        rl_config.__init__(
            num_simulations=20,
            discount=0.9,
            batch_size=16,
            v_min=-2,
            v_max=2,
            unroll_steps=1,
            dynamics_blocks=1,
            enable_rescale=False,
            weight_decay=0,
        )
        rl_config.memory.warmup_size = 200
        rl_config.lr = 0.002
        rl_config.lr_scheduler.set_step(10_000, 0.0001)
        rl_config.input_image_block.set_alphazero_block(1, 16)
        runner = self.create_test_runner("EasyGrid-Layer", rl_config)
        runner.train(max_train_count=3000)
        assert runner.evaluate_compare_to_baseline_single_player()
