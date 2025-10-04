from typing import Tuple

import pytest

from srl.base.rl.config import RLConfig
from tests.algorithms_.common_long_case import CommonLongCase
from tests.algorithms_.common_quick_case import CommonQuickCase


class QuickCase(CommonQuickCase):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        from srl.algorithms import godq_v1_lstm

        rl_config = godq_v1_lstm.Config()
        rl_config.batch_size = 2
        rl_config.batch_length = 2
        rl_config.memory.warmup_size = 2
        rl_config.base_units = 8
        rl_config.input_block.cont_units = 8
        rl_config.input_block.discrete_units = 8
        return rl_config, {"env_list": ["Grid"]}


class LongCase(CommonLongCase):
    def _create_rl_config(self):
        self.check_test_skip()

        from srl.algorithms import godq_v1_lstm

        rl_config = godq_v1_lstm.Config()
        rl_config.base_units = 128
        rl_config.input_block.cont_units = 128
        rl_config.input_block.discrete_units = 64
        return rl_config

    def test_Tiger(self):
        rl_config = self._create_rl_config()
        rl_config.base_units = 64
        rl_config.batch_length = 1
        rl_config.encode_discrete_type = "Discrete"
        rl_config.feat_type = ""
        rl_config.enable_int_q = False
        rl_config.enable_archive = False
        rl_config.reset_net_interval = 0
        runner = self.create_test_runner("Tiger", rl_config)
        runner.train(max_train_count=5000)
        assert runner.evaluate_compare_to_baseline_single_player(baseline=-0.1)

    @pytest.mark.parametrize("feat_type, archive", [["", False], ["BYOL", True]])
    def test_Grid(self, feat_type, archive):
        rl_config = self._create_rl_config()
        rl_config.feat_type = feat_type
        rl_config.enable_archive = archive
        runner = self.create_test_runner("Grid", rl_config)
        runner.train(max_steps=3_000)
        assert runner.evaluate_compare_to_baseline_single_player(baseline=0.3)

    def test_Pendulum(self):
        rl_config = self._create_rl_config()
        rl_config.discount = 0.9
        rl_config.lr = 0.001
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train(max_steps=200 * 50)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Pendulum_mp(self):
        rl_config = self._create_rl_config()
        rl_config.discount = 0.9
        rl_config.lr = 0.001
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train_mp(max_train_count=200 * 50)
        assert runner.evaluate_compare_to_baseline_single_player()
