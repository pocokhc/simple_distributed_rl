from typing import Tuple

import pytest

import srl
from srl.algorithms import ql
from srl.base.rl.config import RLConfig
from srl.test import TestRL
from tests.algorithms_.common_base_class import CommonBaseSimpleTest


class Test_ql(CommonBaseSimpleTest):
    @pytest.fixture(params=["", "random", "normal"])
    def rl_param(self, request):
        return request.param

    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        return ql.Config(q_init=rl_param), {}


def test_Grid_policy():
    tester = TestRL()
    rl_config = ql.Config()
    rl_config.epsilon.set_constant(0.5)
    rl_config.lr.set_constant(0.01)
    runner = srl.Runner("Grid", rl_config)
    runner.set_seed(1)
    runner.train(max_train_count=100_000)
    tester.eval(runner, episode=100)
    # tester.verify_grid_policy(runner)


def test_Grid_mp():
    tester = TestRL()
    rl_config = ql.Config()
    rl_config.epsilon.set_constant(0.5)
    rl_config.lr.set_constant(0.01)
    runner = srl.Runner("Grid", rl_config)
    runner.set_seed(2)
    runner.train_mp(max_train_count=100_000, queue_capacity=100_000)
    tester.eval(runner, episode=100)


@pytest.mark.parametrize("q_init", ["", "random", "normal"])
def test_Grid(q_init):
    tester = TestRL()
    rl_config = ql.Config(q_init=q_init)
    rl_config.epsilon.set_constant(0.5)
    runner = srl.Runner("Grid", rl_config)
    runner.set_seed(2)
    runner.train(max_train_count=100_000)
    tester.eval(runner, episode=100)


@pytest.mark.parametrize("is_mp", [False, True])
def test_OX(is_mp):
    tester = TestRL()
    rl_config = ql.Config()
    rl_config.epsilon.set_constant(0.5)
    rl_config.lr.set_constant(0.1)
    runner = srl.Runner("OX", rl_config)
    runner.set_seed(1)
    if is_mp:
        runner.train_mp(max_train_count=100_000, queue_capacity=100_000)
    else:
        runner.train(max_train_count=100_000)
    tester.eval_2player(runner)


def test_Tiger():
    tester = TestRL()
    rl_config = ql.Config()
    rl_config.window_length = 10
    runner = srl.Runner("Tiger", rl_config)
    runner.set_seed(2)
    runner.train(max_train_count=500_000)
    tester.eval(runner)
