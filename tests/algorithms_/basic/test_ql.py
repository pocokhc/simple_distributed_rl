import unittest

import pytest

from srl.algorithms import ql
from srl.runner.runner import Runner
from srl.test import TestRL
from srl.utils import common

common.logger_print()


class Test_ql(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        from srl.algorithms import ql

        self.rl_config = ql.Config()


def test_Grid_policy():
    tester = TestRL()
    rl_config = ql.Config()
    rl_config.epsilon.set_constant(0.5)
    rl_config.lr.set_constant(0.01)
    runner = Runner("Grid", rl_config)
    runner.set_seed(2)
    runner.train(max_train_count=100_000)
    tester.eval(runner, episode=100)
    tester.verify_grid_policy(runner)


def test_Grid_mp():
    tester = TestRL()
    rl_config = ql.Config()
    rl_config.epsilon.set_constant(0.5)
    rl_config.lr.set_constant(0.01)
    runner = Runner("Grid", rl_config)
    runner.set_seed(2)
    runner.train_mp(max_train_count=200_000)
    tester.eval(runner, episode=100)


@pytest.mark.parametrize("q_init", ["", "random", "normal"])
def test_Grid(q_init):
    tester = TestRL()
    rl_config = ql.Config(q_init=q_init)
    rl_config.epsilon.set_constant(0.5)
    runner = Runner("Grid", rl_config)
    runner.set_seed(2)
    runner.train(max_train_count=100_000)
    tester.eval(runner, episode=100)


def test_OX():
    tester = TestRL()
    rl_config = ql.Config()
    rl_config.epsilon.set_constant(0.5)
    rl_config.lr.set_constant(0.1)
    runner = Runner("OX", rl_config)
    runner.set_seed(1)
    runner.train(max_train_count=100_000)
    tester.eval_2player(runner)


def test_Tiger():
    tester = TestRL()
    rl_config = ql.Config()
    rl_config.window_length = 10
    runner = Runner("Tiger", rl_config)
    runner.set_seed(2)
    runner.train(max_train_count=500_000)
    tester.eval(runner)
