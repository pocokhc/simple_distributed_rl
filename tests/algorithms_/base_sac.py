from typing import Tuple

import pytest

from srl.base.define import RLBaseActTypes
from srl.base.rl.config import RLConfig
from tests.algorithms_.common_long_case import CommonLongCase
from tests.algorithms_.common_quick_case import CommonQuickCase


class QuickCase(CommonQuickCase):
    @pytest.fixture(
        params=[
            dict(override_action_type=RLBaseActTypes.DISCRETE, entropy_bonus_exclude_q=False),
            dict(override_action_type=RLBaseActTypes.DISCRETE, entropy_bonus_exclude_q=True),
            dict(override_action_type=RLBaseActTypes.CONTINUOUS, enable_normal_squashed=False),
            dict(override_action_type=RLBaseActTypes.CONTINUOUS, enable_normal_squashed=True),
        ]
    )
    def rl_param(self, request):
        return request.param

    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        from srl.algorithms import sac

        rl_config = sac.Config()
        rl_config.memory.warmup_size = 2
        rl_config.batch_size = 2
        rl_config.entropy_bonus_exclude_q = rl_param.get("entropy_bonus_exclude_q", True)
        rl_config.enable_normal_squashed = rl_param.get("enable_normal_squashed", True)
        rl_config.override_action_type = rl_param["override_action_type"]

        return rl_config, {}


class LongCase(CommonLongCase):
    def _create_rl_config(self):
        self.check_test_skip()

        from srl.algorithms import sac

        rl_config = sac.Config()
        return rl_config

    def test_EasyGrid(self):
        rl_config = self._create_rl_config()

        rl_config.batch_size = 32
        rl_config.lr_policy = 0.0002
        rl_config.lr_q = 0.001
        rl_config.memory.capacity = 10000
        rl_config.memory.warmup_size = 1000
        rl_config.policy_hidden_block.set((32, 32, 32))
        rl_config.q_hidden_block.set((32, 32, 32))
        rl_config.entropy_bonus_exclude_q = True
        rl_config.entropy_alpha = 0.1
        rl_config.entropy_alpha_auto_scale = False

        runner = self.create_test_runner("EasyGrid", rl_config)
        runner.train(max_train_count=4000)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Pendulum(self):
        rl_config = self._create_rl_config()
        runner = self.create_test_runner("Pendulum-v1", rl_config)

        rl_config.batch_size = 32
        rl_config.lr_policy = 0.003
        rl_config.lr_q = 0.003
        rl_config.memory.capacity = 10000
        rl_config.memory.warmup_size = 1000
        rl_config.policy_hidden_block.set((64, 64, 64))
        rl_config.q_hidden_block.set((128, 128, 128))

        runner.train(max_train_count=200 * 30)
        assert runner.evaluate_compare_to_baseline_single_player()
