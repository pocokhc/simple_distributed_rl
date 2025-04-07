from typing import Tuple

import pytest

from srl.base.rl.config import RLConfig
from tests.algorithms_.common_long_case import CommonLongCase
from tests.algorithms_.common_quick_case import CommonQuickCase


class QuickCase(CommonQuickCase):
    @pytest.fixture(params=[1, 2])
    def rl_param(self, request):
        return request.param

    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        from srl.algorithms import rainbow

        rl_config = rainbow.Config()
        rl_config.set_tensorflow()
        rl_config.set_atari_config()

        rl_config.batch_size = 2
        rl_config.memory.warmup_size = 2
        rl_config.multisteps = rl_param
        rl_config.target_model_update_interval = 1
        rl_config.enable_rescale = True

        return rl_config, {}


class LongCase(CommonLongCase):
    def _create_rl_config(self):
        self.check_test_skip()
        from srl.algorithms import rainbow

        rl_config = rainbow.Config()
        return rl_config

    def test_Pendulum(self):
        rl_config = self._create_rl_config()
        rl_config.hidden_block.set((64, 64))
        rl_config.multisteps = 3
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 70)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Pendulum_mp(self):
        rl_config = self._create_rl_config()
        rl_config.hidden_block.set((64, 64))
        rl_config.multisteps = 3
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train_mp(max_train_count=200 * 70)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Pendulum_noisy(self):
        rl_config = self._create_rl_config()
        rl_config.hidden_block.set((64, 64))
        rl_config.multisteps = 1
        rl_config.enable_noisy_dense = True
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 70)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Pendulum_no_multi(self):
        rl_config = self._create_rl_config()
        rl_config.hidden_block.set((64, 64))
        rl_config.multisteps = 1
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 70)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Pendulum_no_multi_mp(self):
        rl_config = self._create_rl_config()
        rl_config.hidden_block.set((64, 64))
        rl_config.multisteps = 1
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train_mp(max_train_count=200 * 70)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Pendulum_no_multi_noisy(self):
        rl_config = self._create_rl_config()
        rl_config.hidden_block.set((64, 64))
        rl_config.multisteps = 1
        rl_config.enable_noisy_dense = True
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 70)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_OX(self):
        # invalid action test
        rl_config = self._create_rl_config()
        rl_config.hidden_block.set((64, 32, 16))
        rl_config.epsilon = 0
        rl_config.multisteps = 3
        rl_config.memory.set_replay_buffer()

        runner = self.create_test_runner("OX", rl_config)
        runner.train(max_train_count=10000, players=[None, "random"])

        results = runner.evaluate_compare_to_baseline_multiplayer(
            baseline_params=[
                {"episode": 100, "players": [None, "random"], "baseline": [0.4, None]},
                {"episode": 100, "players": ["random", None], "baseline": [None, 0.4]},
            ]
        )
        assert all(results)

    def _create_pendulum_config(self):
        rl_config = self._create_rl_config()
        rl_config.epsilon = 0.1
        rl_config.discount = 0.9
        rl_config.lr = 0.001
        rl_config.batch_size = 32
        rl_config.hidden_block.set((64, 64))
        rl_config.enable_double_dqn = False
        rl_config.enable_noisy_dense = False
        rl_config.multisteps = 1
        rl_config.memory.set_replay_buffer()
        rl_config.enable_rescale = False
        rl_config.window_length = 1
        return rl_config

    def test_Pendulum_naive(self):
        rl_config = self._create_pendulum_config()
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 100)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Pendulum_window_multi(self):
        rl_config = self._create_pendulum_config()
        rl_config.window_length = 4
        rl_config.multisteps = 3
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 200)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Pendulum_window_length(self):
        pytest.skip("allで代用")
        rl_config = self._create_pendulum_config()
        rl_config.window_length = 4
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 70)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Pendulum_ddqn(self):
        pytest.skip("allで代用")
        rl_config = self._create_pendulum_config()
        rl_config.enable_double_dqn = True
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 80)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Pendulum_dueling(self):
        pytest.skip("allで代用")
        rl_config = self._create_pendulum_config()
        rl_config.hidden_block.set_dueling_network((64, 64))
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 70)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Pendulum_multistep(self):
        pytest.skip("allで代用")
        rl_config = self._create_pendulum_config()
        rl_config.multisteps = 10
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 80)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Pendulum_proportional(self):
        pytest.skip("allで代用")
        rl_config = self._create_pendulum_config()
        rl_config.memory.set_proportional(alpha=1.0, beta_initial=1.0)
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 120)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Pendulum_rankbase(self):
        rl_config = self._create_pendulum_config()
        rl_config.memory.set_rankbased(alpha=1.0, beta_initial=1.0)
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 120)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Pendulum_rankbaseLinear(self):
        rl_config = self._create_pendulum_config()
        rl_config.memory.set_rankbased_linear(alpha=1.0, beta_initial=1.0)
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 120)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Pendulum_all(self):
        rl_config = self._create_pendulum_config()
        rl_config.enable_double_dqn = True
        rl_config.hidden_block.set_dueling_network((64, 64))
        rl_config.enable_noisy_dense = True
        rl_config.multisteps = 5
        rl_config.memory.set_proportional(alpha=1.0, beta_initial=1.0)
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train(max_train_count=200 * 100)
        assert runner.evaluate_compare_to_baseline_single_player()
