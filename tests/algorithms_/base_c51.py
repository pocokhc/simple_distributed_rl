from typing import Tuple

import pytest

from srl.base.rl.config import RLConfig
from tests.algorithms_.common_base_case import CommonBaseCase
from tests.algorithms_.common_quick_case import CommonQuickCase


class QuickCase(CommonQuickCase):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:

        from srl.algorithms import c51

        rl_config = c51.Config()
        rl_config.batch_size = 2
        rl_config.memory_warmup_size = 2

        return rl_config, {}


class BaseCase(CommonBaseCase):
    def _create_rl_config(self):
        from srl.algorithms import c51

        return c51.Config()

    def test_Grid(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.epsilon = 0.5
        rl_config.lr = 0.002
        rl_config.hidden_block.set((16, 16))
        rl_config.categorical_num_atoms = 11
        rl_config.categorical_v_min = -2
        rl_config.categorical_v_max = 2
        runner, tester = self.create_runner("Grid", rl_config)
        runner.train(max_train_count=6000)
        tester.eval(runner)

    def test_Pendulum(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.categorical_v_min = -100
        rl_config.categorical_v_max = 100
        rl_config.batch_size = 64
        rl_config.lr = 0.001
        rl_config.hidden_block.set((32, 32, 32))
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 600)
        tester.eval(runner)

    def test_Pendulum_mp(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.categorical_v_min = -100
        rl_config.categorical_v_max = 100
        rl_config.batch_size = 64
        rl_config.lr = 0.001
        rl_config.hidden_block.set((32, 32, 32))
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train_mp(max_train_count=200 * 600)
        tester.eval(runner)
