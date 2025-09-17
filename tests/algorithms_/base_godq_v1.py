from typing import Tuple

import pytest

from srl.base.rl.config import RLConfig
from tests.algorithms_.common_long_case import CommonLongCase
from tests.algorithms_.common_quick_case import CommonQuickCase


class QuickCase(CommonQuickCase):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        from srl.algorithms import godq_v1

        rl_config = godq_v1.Config()
        rl_config.batch_size = 2
        rl_config.memory.warmup_size = 2
        rl_config.base_units = 8
        rl_config.input_block.cont_units = 8
        rl_config.input_block.discrete_units = 8
        return rl_config, {"env_list": ["Grid"]}


class LongCase(CommonLongCase):
    def _create_rl_config(self):
        self.check_test_skip()

        from srl.algorithms import godq_v1

        rl_config = godq_v1.Config()
        rl_config.base_units = 64
        rl_config.input_block.cont_units = 64
        rl_config.input_block.discrete_units = 32
        return rl_config

    @pytest.mark.parametrize("feat_type, archive", [["SimSiam", False], ["BYOL", True]])
    def test_Grid(self, feat_type, archive):
        rl_config = self._create_rl_config()
        rl_config.feat_type = feat_type
        rl_config.enable_archive = archive
        runner = self.create_test_runner("Grid", rl_config)
        runner.train(max_steps=3_000)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Pendulum(self):
        rl_config = self._create_rl_config()
        rl_config.discount = 0.9
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train(max_steps=200 * 50)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Pendulum_mp(self):
        rl_config = self._create_rl_config()
        rl_config.discount = 0.9
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train_mp(max_train_count=200 * 100)
        assert runner.evaluate_compare_to_baseline_single_player()
