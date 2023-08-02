import unittest

from srl.algorithms import ql_agent57
from srl.runner.runner import Runner
from srl.test import TestRL


class Test_ql_agent57(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        from srl.algorithms import ql_agent57

        self.rl_config = ql_agent57.Config()


def test_Grid():
    tester = TestRL()
    rl_config = ql_agent57.Config()
    rl_config.enable_actor = False
    rl_config.epsilon = 0.5
    runner = Runner("Grid", rl_config)
    runner.set_seed(3)
    runner.train(max_train_count=100_000)
    tester.eval(runner, episode=100)
    tester.verify_grid_policy(runner)


def test_Grid_window_length():
    tester = TestRL()
    rl_config = ql_agent57.Config()
    rl_config.enable_actor = False
    rl_config.epsilon = 0.5
    rl_config.window_length = 2
    runner = Runner("Grid", rl_config)
    runner.set_seed(3)
    runner.train(max_train_count=50_000)
    tester.eval(runner, episode=100)


def test_Grid_mp():
    tester = TestRL()
    rl_config = ql_agent57.Config()
    rl_config.enable_actor = False
    rl_config.epsilon = 0.5
    runner = Runner("Grid", rl_config)
    runner.train_mp(max_train_count=50_000)
    tester.eval(runner, episode=100)


def test_OneRoad():
    tester = TestRL()
    rl_config = ql_agent57.Config()
    runner = Runner("Grid", rl_config)
    runner.set_seed(2)
    runner.train(max_train_count=10_000)
    tester.eval(runner)
