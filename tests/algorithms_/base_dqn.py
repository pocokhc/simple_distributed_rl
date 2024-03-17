from typing import Tuple

import pytest

from srl.base.rl.config import RLConfig
from tests.algorithms_.common_base_case import CommonBaseCase
from tests.algorithms_.common_quick_case import CommonQuickCase


class QuickCase(CommonQuickCase):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("tensorflow")

        from srl.algorithms import dqn

        rl_config = dqn.Config()
        rl_config.set_tensorflow()

        rl_config.set_atari_config()
        rl_config.batch_size = 2
        rl_config.memory.warmup_size = 2
        rl_config.target_model_update_interval = 1
        rl_config.enable_double_dqn = True
        rl_config.enable_rescale = True
        rl_config.input_image_block.set_dqn_block(filters=2)
        rl_config.hidden_block.set((2,))

        return rl_config, {}


class BaseCase(CommonBaseCase):
    def _create_rl_config(self):
        from srl.algorithms import dqn

        rl_config = dqn.Config(enable_double_dqn=False)
        rl_config.hidden_block.set_mlp((64, 64))
        if self.get_framework() == "tensorflow":
            rl_config.framework.set_tensorflow()
        elif self.get_framework() == "torch":
            rl_config.framework.set_torch()
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

    def test_Pendulum_DDQN(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.enable_double_dqn = True

        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train(max_steps=200 * 70)
        tester.eval(runner)

    def test_Pendulum_window(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.window_length = 4

        runner, tester = self.create_runner("Pendulum-v1", rl_config)
        runner.train(max_steps=200 * 80)
        tester.eval(runner)

        runner.model_summary()

    def test_OX(self):
        self.check_skip()
        rl_config = self._create_rl_config()
        rl_config.hidden_block.set_mlp((32, 32, 16))
        rl_config.epsilon.set_constant(0)

        runner, tester = self.create_runner("OX", rl_config)
        runner.set_seed(2)
        runner.train(max_train_count=10_000)

        runner.set_players([None, "random"])
        tester.eval(runner, baseline=[0.4, None])
        runner.set_players(["random", None])
        tester.eval(runner, baseline=[None, 0.4])
