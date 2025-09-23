from typing import Tuple

import pytest

from srl.base.define import RLBaseTypes
from srl.base.rl.config import RLConfig
from tests.algorithms_.common_long_case import CommonLongCase
from tests.algorithms_.common_quick_case import CommonQuickCase


class QuickCase(CommonQuickCase):
    @pytest.fixture(
        params=[
            [RLBaseTypes.DISCRETE, "MC", "", ""],
            [RLBaseTypes.DISCRETE, "MC", "ave", "clip"],
            [RLBaseTypes.DISCRETE, "GAE", "std", "kl"],
            [RLBaseTypes.NP_ARRAY, "GAE", "normal", "kl"],
            [RLBaseTypes.NP_ARRAY, "MC", "advantage", "clip"],
        ]
    )
    def rl_param(self, request):
        return request.param

    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        from srl.algorithms import ppo

        rl_config = ppo.Config(
            override_action_type=rl_param[0],
            experience_collection_method=rl_param[1],
            baseline_type=rl_param[2],
            surrogate_type=rl_param[3],
        )
        rl_config.batch_size = 2
        rl_config.memory.warmup_size = 2
        rl_config.train_num = 2

        return rl_config, {}

    def test_simple_rollout_train(self, rl_param, tmpdir):
        pytest.skip("not support")

    def test_simple_mp_memory(self, rl_param, tmpdir):
        pytest.skip("not support")

    def test_simple_mp(self, rl_param, tmpdir):
        pytest.skip("not support")


class LongCase(CommonLongCase):
    def _create_rl_config(self):
        self.check_test_skip()
        from srl.algorithms import ppo

        rl_config = ppo.Config(
            batch_size=64,
            discount=0.9,
            gae_discount=0.9,
            surrogate_type="clip",
            baseline_type="normal",
            experience_collection_method="MC",
            enable_value_clip=False,
            enable_state_normalized=False,
        )
        rl_config.lr = 0.0005
        rl_config.hidden_block.set((64, 64))
        rl_config.value_block.set(())
        rl_config.policy_block.set(())
        rl_config.memory.warmup_size = 500
        rl_config.train_num = 50
        return rl_config

    def test_EasyGrid1(self):
        rl_config = self._create_rl_config()
        rl_config.experience_collection_method = "GAE"
        rl_config.baseline_type = ""
        rl_config.surrogate_type = "clip"
        rl_config.enable_value_clip = True
        rl_config.enable_state_normalized = False
        runner = self.create_test_runner("EasyGrid", rl_config)
        runner.train(max_train_count=5000)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Grid2(self):
        rl_config = self._create_rl_config()
        rl_config.experience_collection_method = "GAE"
        rl_config.baseline_type = "v"
        rl_config.surrogate_type = "clip"
        rl_config.enable_value_clip = False
        rl_config.enable_state_normalized = False
        runner = self.create_test_runner("Grid", rl_config)
        runner.train(max_train_count=5000)
        assert runner.evaluate_compare_to_baseline_single_player(baseline=0.2)

    def test_Grid3(self):
        rl_config = self._create_rl_config()
        rl_config.experience_collection_method = "MC"
        rl_config.baseline_type = "normal"
        rl_config.surrogate_type = "kl"
        rl_config.enable_value_clip = False
        rl_config.enable_state_normalized = False
        runner = self.create_test_runner("Grid", rl_config)
        runner.train(max_train_count=3000)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Pendulum_continue(self):
        rl_config = self._create_rl_config()
        rl_config.experience_collection_method = "MC"
        rl_config.baseline_type = "advantage"
        rl_config.surrogate_type = "clip"
        rl_config.enable_value_clip = False
        rl_config.lr = 0.0002
        rl_config.memory.warmup_size = 400
        rl_config.train_num = 50
        rl_config.hidden_block.set((128,))
        rl_config.value_block.set((128,))
        rl_config.policy_block.set((128,))
        rl_config.discount = 0.9
        rl_config.entropy_weight = 0.1
        rl_config.reward_shift = 1
        rl_config.reward_scale = 1.0 / 10.0
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 200)
        assert runner.evaluate_compare_to_baseline_single_player()
