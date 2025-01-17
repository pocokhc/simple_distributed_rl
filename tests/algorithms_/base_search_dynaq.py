import pytest

import srl
from srl.algorithms import search_dynaq
from tests.algorithms_.common_long_case import CommonLongCase


class LongCase(CommonLongCase):
    @pytest.mark.parametrize("is_mp", [False, True])
    def test_Grid(self, is_mp):
        rl_config = search_dynaq.Config()
        runner = srl.Runner("Grid", rl_config)
        runner.set_seed(1)
        if is_mp:
            runner.train_mp(max_train_count=10_000, queue_capacity=100_000)
        else:
            runner.train(max_train_count=10_000)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_OneRoad(self):
        rl_config = search_dynaq.Config()
        runner = srl.Runner("OneRoad", rl_config)
        runner.set_seed(4)
        runner.train(max_train_count=2_000)
        assert runner.evaluate_compare_to_baseline_single_player()

    @pytest.mark.parametrize("is_mp", [False, True])
    def test_OX(self, is_mp):
        rl_config = search_dynaq.Config()
        runner = srl.Runner("OX", rl_config)
        runner.set_seed(1)
        if is_mp:
            runner.train_mp(max_train_count=10_000, queue_capacity=100_000)
        else:
            runner.train(max_train_count=10_000)
        runner.render_terminal()
        results = runner.evaluate_compare_to_baseline_multiplayer()
        assert all(results)
