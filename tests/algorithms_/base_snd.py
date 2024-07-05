from typing import Tuple

import pytest

from srl.base.rl.config import RLConfig
from tests.algorithms_.common_base_case import CommonBaseCase
from tests.algorithms_.common_quick_case import CommonQuickCase


class QuickCase(CommonQuickCase):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:

        from srl.algorithms import snd

        rl_config = snd.Config()
        rl_config.set_tensorflow()

        rl_config.batch_size = 2
        rl_config.memory_warmup_size = 2
        rl_config.target_model_update_interval = 1
        rl_config.input_image_block.set_dqn_block(filters=2)
        rl_config.hidden_block.set((2,))

        return rl_config, {}


class BaseCase(CommonBaseCase):
    def _create_rl_config(self):
        from srl.algorithms import snd

        rl_config = snd.Config()
        rl_config.hidden_block.set((64, 64))
        return rl_config

    def test_Pendulum(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train(max_steps=200 * 100)
        tester.eval(runner)

    def test_Pendulum_mp(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train_mp(max_train_count=200 * 100)
        tester.eval(runner)
