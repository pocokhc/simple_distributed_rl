from typing import Tuple

from srl.base.rl.config import RLConfig
from tests.algorithms_.common_long_case import CommonLongCase
from tests.algorithms_.common_quick_case import CommonQuickCase


class QuickCase(CommonQuickCase):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        from srl.algorithms import c51

        rl_config = c51.Config()
        rl_config.batch_size = 2
        rl_config.memory_warmup_size = 2

        return rl_config, {}


class LongCase(CommonLongCase):
    def _create_rl_config(self):
        self.check_test_skip()

        from srl.algorithms import c51

        return c51.Config()

    def test_Grid(self):
        rl_config = self._create_rl_config()
        rl_config.epsilon = 0.5
        rl_config.lr = 0.002
        rl_config.hidden_block.set((16, 16))
        rl_config.categorical_num_atoms = 11
        rl_config.categorical_v_min = -2
        rl_config.categorical_v_max = 2
        runner = self.create_test_runner("Grid", rl_config)
        runner.train(max_train_count=6000)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Pendulum(self):
        rl_config = self._create_rl_config()
        rl_config.categorical_v_min = -100
        rl_config.categorical_v_max = 100
        rl_config.batch_size = 64
        rl_config.lr = 0.001
        rl_config.hidden_block.set((32, 32, 32))
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 600)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Pendulum_mp(self):
        rl_config = self._create_rl_config()
        rl_config.categorical_v_min = -100
        rl_config.categorical_v_max = 100
        rl_config.batch_size = 64
        rl_config.lr = 0.001
        rl_config.hidden_block.set((32, 32, 32))
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train_mp(max_train_count=200 * 600)
        assert runner.evaluate_compare_to_baseline_single_player()
