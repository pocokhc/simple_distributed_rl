import unittest

import pytest

from srl import runner
from srl.algorithms import ql
from srl.test import TestRL
from srl.utils import common

common.logger_print()


class Test_ql(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        from srl.algorithms import ql

        self.rl_config = ql.Config()


def test_Grid_policy():
    tester = TestRL()
    rl_config = ql.Config(
        epsilon=0.5,
        lr=0.01,
    )
    config = runner.Config("Grid", rl_config, seed=2)
    parameter, _, _ = tester.train_eval(config, 100_000, eval_episode=100)
    tester.verify_grid_policy(rl_config, parameter)


def test_Grid_mp():
    tester = TestRL()
    rl_config = ql.Config(
        epsilon=0.5,
        lr=0.01,
    )
    config = runner.Config("Grid", rl_config)
    tester.train_eval(config, 200_000, is_mp=True, eval_episode=100)


@pytest.mark.parametrize("q_init", ["", "random", "normal"])
def test_Grid(q_init):
    tester = TestRL()
    rl_config = ql.Config(
        epsilon=0.5,
        q_init=q_init,
    )
    config = runner.Config("Grid", rl_config, seed=2)
    tester.train_eval(config, 100_000, eval_episode=100)


def test_OX():
    tester = TestRL()
    rl_config = ql.Config(
        epsilon=0.5,
        lr=0.1,
    )
    config = runner.Config("OX", rl_config, seed=1)
    parameter, _, _ = tester.train(config, 100_000)
    tester.eval_2player(config, parameter)


def test_Tiger():
    tester = TestRL()
    rl_config = ql.Config()
    rl_config.window_length = 10
    config = runner.Config("Tiger", rl_config, seed=2)
    tester.train_eval(config, 500_000)
