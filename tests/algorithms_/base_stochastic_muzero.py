from typing import Tuple

from srl.base.rl.config import RLConfig
from tests.algorithms_.common_long_case import CommonLongCase
from tests.algorithms_.common_quick_case import CommonQuickCase


class QuickCase(CommonQuickCase):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        from srl.algorithms import stochastic_muzero

        rl_config = stochastic_muzero.Config(
            num_simulations=1,
            unroll_steps=1,
            batch_size=2,
            dynamics_blocks=1,
        )
        rl_config.memory_warmup_size = 2
        rl_config.input_image_block.set_alphazero_block(1, 4)

        return rl_config, dict(use_layer_processor=True)


class LongCase(CommonLongCase):
    def _create_rl_config(self):
        self.check_test_skip()

        from srl.algorithms import stochastic_muzero

        rl_config = stochastic_muzero.Config(
            num_simulations=10,
            discount=0.9,
            batch_size=16,
            v_min=-2,
            v_max=2,
            unroll_steps=2,
            dynamics_blocks=1,
            enable_rescale=False,
            codebook_size=4,
        )
        rl_config.memory_warmup_size = 200
        rl_config.lr = 0.001
        rl_config.set_replay_memory()
        rl_config.input_image_block.set_alphazero_block(1, 16)
        return rl_config

    def test_Grid(self):
        from srl.envs import grid

        rl_config = self._create_rl_config()
        rl_config.processors = [grid.LayerProcessor()]

        runner = self.create_test_runner("Grid", rl_config)
        runner.train(max_train_count=10000)
        assert runner.evaluate_compare_to_baseline_single_player(baseline=0.4)
