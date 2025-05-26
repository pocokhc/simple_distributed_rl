from typing import Tuple

import pytest

from srl.base.define import RLBaseTypes
from srl.base.rl.config import RLConfig
from tests.algorithms_.common_long_case import CommonLongCase
from tests.algorithms_.common_quick_case import CommonQuickCase


class QuickCase(CommonQuickCase):
    @pytest.fixture(
        params=[RLBaseTypes.DISCRETE, RLBaseTypes.ARRAY_CONTINUOUS],  # action
    )
    def rl_param(self, request):
        return request.param

    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        from srl.algorithms import efficient_zero_v2

        rl_config = efficient_zero_v2.Config()
        rl_config.res_blocks = 1
        rl_config.res_channels = 2
        rl_config.reward_units = 2
        rl_config.value_units = 2
        rl_config.policy_units = 2
        rl_config.projection_hid = 2
        rl_config.projection_out = 2
        rl_config.projection_head_hid = 2
        rl_config.projection_head_out = 2
        rl_config.batch_size = 2
        rl_config.memory.warmup_size = 2
        rl_config.num_simulations = 2
        rl_config.num_top_actions = 2
        rl_config.unroll_steps = 2

        rl_config.override_action_type = rl_param
        return rl_config, dict(use_layer_processor=True)

    def test_simple_input_image(self, rl_param, tmpdir):
        pytest.skip()


class LongCase(CommonLongCase):
    def _create_rl_config(self):
        self.check_test_skip()
        from srl.algorithms import efficient_zero_v2

        rl_config = efficient_zero_v2.Config()
        rl_config.set_small_params()

        return rl_config

    def test_EasyGrid(self):
        rl_config = self._create_rl_config()
        rl_config.num_simulations = 20
        rl_config.lr = 0.001
        rl_config.reward_loss_coeff = 1.0
        rl_config.consistency_loss_coeff = 1
        rl_config.unroll_steps = 2
        rl_config.memory.warmup_size = 100

        runner = self.create_test_runner("EasyGrid-layer", rl_config)
        runner.train_mp(max_train_count=2000)
        assert runner.evaluate_compare_to_baseline_single_player(episode=5)

    def test_EasyGridCont(self):
        rl_config = self._create_rl_config()
        rl_config.num_simulations = 20
        rl_config.lr = 0.001
        rl_config.reward_loss_coeff = 1.0
        rl_config.consistency_loss_coeff = 1
        rl_config.unroll_steps = 2
        rl_config.memory.warmup_size = 100

        rl_config.override_action_type = "ARRAY_CONTINUOUS"

        runner = self.create_test_runner("EasyGrid-layer", rl_config)
        runner.train_mp(max_train_count=10000)
        assert runner.evaluate_compare_to_baseline_single_player(episode=5)
