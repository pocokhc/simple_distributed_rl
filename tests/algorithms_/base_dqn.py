from typing import Tuple

from srl.base.rl.config import RLConfig
from tests.algorithms_.common_long_case import CommonLongCase
from tests.algorithms_.common_quick_case import CommonQuickCase


class QuickCase(CommonQuickCase):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        from srl.algorithms import dqn

        rl_config = dqn.Config()
        rl_config.set_tensorflow()

        rl_config.set_atari_config()
        rl_config.batch_size = 2
        rl_config.memory.warmup_size = 2
        rl_config.target_model_update_interval = 1
        rl_config.enable_double_dqn = True
        rl_config.enable_rescale = True
        rl_config.input_block.image.set_dqn_block(filters=2)
        rl_config.input_block.value.set((1,))
        rl_config.hidden_block.set((2,))

        return rl_config, {}


class LongCase(CommonLongCase):
    def _create_rl_config(self):
        self.check_test_skip()

        from srl.algorithms import dqn

        rl_config = dqn.Config(enable_double_dqn=False)
        rl_config.hidden_block.set((64, 64))
        return rl_config

    def test_Pendulum(self):
        rl_config = self._create_rl_config()
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train(max_steps=200 * 100)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Pendulum_mp(self):
        rl_config = self._create_rl_config()
        runner = self.create_test_runner("Pendulum-v1", rl_config)
        runner.train_mp(max_train_count=200 * 100)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Pendulum_DDQN(self):
        rl_config = self._create_rl_config()
        rl_config.enable_double_dqn = True
        runner = self.create_test_runner("Pendulum-v1", rl_config)

        runner.train(max_steps=200 * 70)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Pendulum_window(self):
        rl_config = self._create_rl_config()
        rl_config.window_length = 4
        runner = self.create_test_runner("Pendulum-v1", rl_config)

        runner.train(max_steps=200 * 80)
        assert runner.evaluate_compare_to_baseline_single_player()

        runner.model_summary()

    def test_OX(self):
        rl_config = self._create_rl_config()
        rl_config.hidden_block.set((32, 32, 16))
        rl_config.epsilon = 0
        runner = self.create_test_runner("OX", rl_config)

        runner.set_seed(2)
        runner.train(max_train_count=10_000)

        results = runner.evaluate_compare_to_baseline_multiplayer(
            baseline_params=[
                {"episode": 100, "players": [None, "random"], "baseline": [0.4, None]},
                {"episode": 100, "players": ["random", None], "baseline": [None, 0.4]},
            ]
        )
        assert all(results)
