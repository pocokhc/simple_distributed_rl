from typing import Tuple

from srl.algorithms import dynaq
from srl.base.rl.config import RLConfig
from srl.runner.runner import Runner
from srl.test import TestRL
from tests.algorithms_.common_base_class import CommonBaseSimpleTest


class Test_dynaq(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        return dynaq.Config(), {}


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
