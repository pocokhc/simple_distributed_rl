from typing import Tuple

import pytest

from srl.base.define import RLBaseTypes
from srl.base.rl.config import RLConfig
from tests.algorithms_.common_long_case import CommonLongCase
from tests.algorithms_.common_quick_case import CommonQuickCase


class QuickCase(CommonQuickCase):
    @pytest.fixture(
        params=[
            [RLBaseTypes.DISCRETE, False],
            [RLBaseTypes.NP_ARRAY, False],
            [RLBaseTypes.NP_ARRAY, True],
        ]
    )
    def rl_param(self, request):
        return request.param

    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        from srl.algorithms import ppo_v

        rl_config = ppo_v.Config(
            override_action_type=rl_param[0],
            squashed_gaussian_policy=rl_param[1],
        )
        rl_config.set_model(4)
        rl_config.batch_size = 2
        rl_config.memory.warmup_size = 2

        return rl_config, {}


class LongCase(CommonLongCase):
    def _create_rl_config(self):
        self.check_test_skip()
        from srl.algorithms import ppo_v

        rl_config = ppo_v.Config(
            batch_size=64,
            discount=0.9,
        )
        rl_config.set_model(64)
        rl_config.memory.warmup_size = 1000
        rl_config.memory.compress = False
        return rl_config

    def test_EasyGrid(self):
        rl_config = self._create_rl_config()
        runner = self.create_test_runner("EasyGrid", rl_config)
        runner.train(max_train_count=5000)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Pendulum_continue(self):
        rl_config = self._create_rl_config()
        rl_config.loss_align_coeff = 0.2
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 200)
        assert runner.evaluate_compare_to_baseline_single_player()
