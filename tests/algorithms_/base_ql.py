import pytest

import srl
from srl.algorithms import ql
from tests.algorithms_.common_long_case import CommonLongCase


class LongCase(CommonLongCase):
    def test_Grid_policy(self):
        rl_config = ql.Config()
        rl_config.epsilon = 0.5
        rl_config.lr = 0.01
        runner = srl.Runner("Grid", rl_config)
        runner.set_seed(1)
        runner.train(max_train_count=100_000)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Grid_mp(self):
        rl_config = ql.Config()
        rl_config.epsilon = 0.5
        rl_config.lr = 0.01
        runner = srl.Runner("Grid", rl_config)
        runner.set_seed(2)
        runner.train_mp(max_train_count=100_000, queue_capacity=100_000)
        assert runner.evaluate_compare_to_baseline_single_player()

    @pytest.mark.parametrize("q_init", ["", "random", "normal"])
    def test_Grid(self, q_init):
        rl_config = ql.Config(q_init=q_init)
        rl_config.epsilon = 0.5
        runner = srl.Runner("Grid", rl_config)
        runner.set_seed(2)
        runner.train(max_train_count=100_000)
        assert runner.evaluate_compare_to_baseline_single_player()

    @pytest.mark.parametrize("is_mp", [False, True])
    def test_OX(self, is_mp):
        rl_config = ql.Config()
        rl_config.epsilon = 0.5
        rl_config.lr = 0.1
        runner = srl.Runner("OX", rl_config)
        runner.set_seed(1)
        if is_mp:
            runner.train_mp(max_train_count=100_000, queue_capacity=100_000)
        else:
            runner.train(max_train_count=100_000)

        results = runner.evaluate_compare_to_baseline_multiplayer()
        assert all(results)

    def test_Tiger(self):
        rl_config = ql.Config()
        rl_config.window_length = 10
        runner = srl.Runner("Tiger", rl_config)
        runner.set_seed(2)
        runner.train(max_train_count=500_000)
        assert runner.evaluate_compare_to_baseline_single_player()
