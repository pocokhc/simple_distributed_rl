import unittest

from srl.algorithms import search_dynaq
from srl.runner.runner import Runner
from srl.test import TestRL


class Test_search_dynaq(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        from srl.algorithms import search_dynaq

        self.rl_config = search_dynaq.Config()


def test_Grid():
    tester = TestRL()
    rl_config = search_dynaq.Config()
    rl_config.ext_lr = 0.01
    runner = Runner("Grid", rl_config)
    runner.set_seed(1)
    runner.train(max_train_count=10_000)
    tester.eval(runner)
    tester.verify_grid_policy(runner)


def test_Grid_mp():
    tester = TestRL()
    rl_config = search_dynaq.Config()
    rl_config.ext_lr = 0.01
    runner = Runner("Grid", rl_config)
    runner.set_seed(1)
    runner.train_mp(max_train_count=10_000)
    tester.eval(runner)


def test_OneRoad():
    tester = TestRL()
    rl_config = search_dynaq.Config()
    runner = Runner("OneRoad", rl_config)
    runner.set_seed(4)
    runner.train(max_train_count=2_000)
    tester.eval(runner)
