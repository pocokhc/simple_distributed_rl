import unittest

from srl import runner
from srl.algorithms import search_dynaq
from srl.test import TestRL


class Test_search_dynaq(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        from srl.algorithms import search_dynaq

        self.rl_config = search_dynaq.Config()


def test_Grid():
    tester = TestRL()
    rl_config = search_dynaq.Config()
    rl_config.ext_lr = 0.01
    config = runner.Config("Grid", rl_config, seed=1)
    parameter, _, _ = tester.train_eval(config, 10_000, eval_episode=100)
    tester.verify_grid_policy(rl_config, parameter)


def test_Grid_mp():
    tester = TestRL()
    rl_config = search_dynaq.Config()
    rl_config.ext_lr = 0.01
    config = runner.Config("Grid", rl_config, seed=1)
    tester.train_eval(config, 10_000, is_mp=True, eval_episode=100)


def test_OneRoad():
    tester = TestRL()
    rl_config = search_dynaq.Config()
    config = runner.Config("OneRoad", rl_config, seed=4)
    tester.train_eval(config, 2_000)
