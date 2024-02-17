from typing import Tuple

import pytest

import srl
from srl.algorithms import search_dynaq
from srl.base.rl.config import RLConfig
from srl.test import TestRL
from tests.algorithms_.common_base_class import CommonBaseSimpleTest


class Test_search_dynaq(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        return search_dynaq.Config(), {}


@pytest.mark.parametrize("is_mp", [False, True])
def test_Grid(is_mp):
    tester = TestRL()
    rl_config = search_dynaq.Config()
    runner = srl.Runner("Grid", rl_config)
    runner.set_seed(1)
    if is_mp:
        runner.train_mp(max_train_count=10_000, queue_capacity=100_000)
    else:
        runner.train(max_train_count=10_000)
    tester.eval(runner)


def test_OneRoad():
    tester = TestRL()
    rl_config = search_dynaq.Config()
    runner = srl.Runner("OneRoad", rl_config)
    runner.set_seed(4)
    runner.train(max_train_count=2_000)
    tester.eval(runner)


@pytest.mark.parametrize("is_mp", [False, True])
def test_OX(is_mp):
    tester = TestRL()
    rl_config = search_dynaq.Config()
    runner = srl.Runner("OX", rl_config)
    runner.set_seed(1)
    if is_mp:
        runner.train_mp(max_train_count=10_000, queue_capacity=100_000)
    else:
        runner.train(max_train_count=10_000)
    runner.render_terminal()
    tester.eval_2player(runner)
