from typing import Tuple

from srl.base.rl.config import RLConfig
from tests.algorithms_.common_long_case import CommonLongCase
from tests.algorithms_.common_quick_case import CommonQuickCase


class QuickCase(CommonQuickCase):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        from srl.algorithms import r2d2

        rl_config = r2d2.Config()
        rl_config.set_atari_config()

        rl_config.batch_size = 2
        rl_config.memory_warmup_size = 2
        rl_config.lstm_units = 2
        rl_config.burnin = 2
        rl_config.sequence_length = 2
        rl_config.target_model_update_interval = 1
        rl_config.enable_retrace = True

        return rl_config, {}


class LongCase(CommonLongCase):
    def _create_rl_config(self):
        self.check_test_skip()
        from srl.algorithms import r2d2

        rl_config = r2d2.Config(
            lstm_units=32,
            target_model_update_interval=100,
            enable_rescale=True,
            burnin=5,
            sequence_length=5,
            enable_retrace=False,
        )
        rl_config.hidden_block.set((16, 16))
        rl_config.set_replay_memory()
        return rl_config

    def test_Pendulum(self):
        rl_config = self._create_rl_config()
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 35)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Pendulum_mp(self):
        rl_config = self._create_rl_config()
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train_mp(max_train_count=200 * 20)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Pendulum_retrace(self):
        rl_config = self._create_rl_config()
        rl_config.enable_retrace = True
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 35)
        assert runner.evaluate_compare_to_baseline_single_player()
