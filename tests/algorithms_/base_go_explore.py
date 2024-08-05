from typing import Tuple

from srl.base.rl.config import RLConfig
from tests.algorithms_.common_base_case import CommonBaseCase
from tests.algorithms_.common_quick_case import CommonQuickCase


class QuickCase(CommonQuickCase):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:

        from srl.algorithms import go_explore

        rl_config = go_explore.Config()
        rl_config.batch_size = 2
        rl_config.memory_warmup_size = 2
        rl_config.target_model_update_interval = 1
        rl_config.input_image_block.set_dqn_block(filters=2)
        rl_config.hidden_block.set((2,))

        return rl_config, {}


class BaseCase(CommonBaseCase):
    def _create_rl_config(self):
        from srl.algorithms import go_explore

        rl_config = go_explore.Config(
            lr=0.001,
            epsilon=0.1,
            target_model_update_interval=2000,
            memory_warmup_size=1000,
            memory_capacity=10_000,
            downsampling_size=(16, 16),
            downsampling_val=4,
        )
        rl_config.hidden_block.set((64, 64))
        return rl_config

    def test_Grid(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        runner, tester = self.create_runner("Grid", rl_config)
        runner.rollout(max_steps=10_000)
        runner.train(max_steps=10_000)
        tester.eval(runner)
