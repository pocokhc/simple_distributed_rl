from typing import Tuple

import pytest

import srl
from srl.algorithms import go_dynaq
from srl.base.rl.config import RLConfig
from srl.test.rl import TestRL
from tests.algorithms_.common_quick_case import CommonQuickCase


class QuickCase(CommonQuickCase):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        return go_dynaq.Config(), {}


class BaseCase:
    @pytest.mark.parametrize("is_mp", [False, True])
    def test_Grid(self, is_mp):
        if is_mp:
            pytest.skip("TODO")
        tester = TestRL()
        rl_config = go_dynaq.Config()
        runner = srl.Runner("Grid", rl_config)
        runner.set_seed(1)
        if is_mp:
            runner.train_mp(max_train_count=10_000, queue_capacity=100_000)
        else:
            runner.train(max_train_count=10_000)
        tester.eval(runner)

    def test_OneRoad(self):
        tester = TestRL()
        rl_config = go_dynaq.Config()
        runner = srl.Runner("OneRoad", rl_config)
        runner.set_seed(4)
        runner.train(max_train_count=2_000)
        tester.eval(runner)
