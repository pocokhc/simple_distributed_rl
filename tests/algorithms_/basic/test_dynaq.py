import unittest

from srl.algorithms import dynaq
from srl.runner.runner import Runner
from srl.test import TestRL


class Test_dynaq(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        from srl.algorithms import dynaq

        self.rl_config = dynaq.Config()


def test_Grid():
    tester = TestRL()
    rl_config = dynaq.Config()
    runner = Runner("Grid", rl_config)
    runner.set_seed(5)
    runner.train(max_train_count=50_000)
    tester.eval(runner)
    tester.verify_grid_policy(runner)


def test_Grid_mp():
    tester = TestRL()
    rl_config = dynaq.Config()
    runner = Runner("Grid", rl_config)
    runner.set_seed(5)
    runner.train_mp(max_train_count=50_000)
    tester.eval(runner)
