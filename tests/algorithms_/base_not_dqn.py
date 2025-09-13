from typing import Tuple

from srl.base.rl.config import RLConfig
from tests.algorithms_.common_long_case import CommonLongCase
from tests.algorithms_.common_quick_case import CommonQuickCase


class QuickCase(CommonQuickCase):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        from srl.algorithms import not_dqn

        rl_config = not_dqn.Config()
        rl_config.batch_size = 2
        rl_config.memory.warmup_size = 2
        rl_config.input_block.image.set_dqn_block(filters=2)
        rl_config.input_block.value.set((1,))
        rl_config.base_units = 2
        return rl_config, {"env_list": ["Grid"]}


class LongCase(CommonLongCase):
    def _create_rl_config(self):
        self.check_test_skip()

        from srl.algorithms import not_dqn

        rl_config = not_dqn.Config(discount=0.9, lr=0.001)
        rl_config.base_units = 64
        return rl_config

    def test_Pendulum(self):
        rl_config = self._create_rl_config()
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train(max_steps=200 * 100)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Pendulum_mp(self):
        rl_config = self._create_rl_config()
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train_mp(max_train_count=200 * 100)
        assert runner.evaluate_compare_to_baseline_single_player()
