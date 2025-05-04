from typing import Tuple

import pytest

from srl.base.rl.config import RLConfig
from tests.algorithms_.common_long_case import CommonLongCase
from tests.algorithms_.common_quick_case import CommonQuickCase


class QuickCase(CommonQuickCase):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        from srl.algorithms import agent57

        rl_config = agent57.Config()
        rl_config.set_tensorflow()

        rl_config.memory.warmup_size = 2
        rl_config.batch_size = 2
        rl_config.lstm_units = 4
        rl_config.input_image_block.set_dqn_block(filters=2)
        rl_config.hidden_block.set_dueling_network((2, 2))
        rl_config.target_model_update_interval = 1
        rl_config.burnin = 1
        rl_config.sequence_length = 2
        rl_config.input_ext_reward = True
        rl_config.input_int_reward = True
        rl_config.input_action = True

        return rl_config, {}

    def test_simple_dtype(self, rl_param, tmpdir):
        pytest.skip("LSTM + float16 is not supported")


class LongCase(CommonLongCase):
    def _create_rl_config(self):
        self.check_test_skip()

        from srl.algorithms import agent57

        rl_config = agent57.Config(
            lstm_units=64,
            target_model_update_interval=100,
            enable_rescale=True,
            batch_size=32,
            burnin=10,
            sequence_length=10,
            actor_num=2,
            input_ext_reward=False,
            input_int_reward=False,
            input_action=False,
            enable_intrinsic_reward=True,
        )
        rl_config.hidden_block.set((64, 64))
        rl_config.lr_ext = 0.001
        rl_config.lr_int = 0.001
        rl_config.memory.set_replay_buffer()

        return rl_config

    def test_Pendulum(self):
        rl_config = self._create_rl_config()
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 50)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Pendulum_mp(self):
        rl_config = self._create_rl_config()
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train_mp(max_train_count=200 * 50)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Pendulum_uvfa(self):
        rl_config = self._create_rl_config()
        rl_config.input_ext_reward = True
        rl_config.input_int_reward = True
        rl_config.input_action = True
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 70)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Pendulum_memory(self):
        rl_config = self._create_rl_config()
        rl_config.memory.set_proportional(beta_steps=200 * 30)
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 50)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Pendulum_dis_int(self):
        rl_config = self._create_rl_config()
        rl_config.enable_intrinsic_reward = False
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 50)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_OX(self):
        rl_config = self._create_rl_config()
        rl_config.burnin = 3
        rl_config.sequence_length = 3
        rl_config.hidden_block.set((64, 32, 16))

        runner = self.create_test_runner("OX", rl_config)
        runner.train(max_train_count=10_000)

        results = runner.evaluate_compare_to_baseline_multiplayer(
            baseline_params=[
                {"episode": 100, "players": [None, "random"], "baseline": [0.4, None]},
                {"episode": 100, "players": ["random", None], "baseline": [None, 0.4]},
            ]
        )
        assert all(results)
