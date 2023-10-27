from typing import Tuple

import srl
from srl.algorithms import ql_agent57
from srl.base.rl.config import RLConfig
from srl.test import TestRL
from tests.algorithms_.common_base_class import CommonBaseSimpleTest


class Test_ql_agent57(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        return ql_agent57.Config(), {}


def test_Grid():
    tester = TestRL()
    rl_config = ql_agent57.Config()
    rl_config.enable_actor = False
    rl_config.epsilon.set_constant(0.5)
    runner = srl.Runner("Grid", rl_config)
    runner.set_seed(3)
    runner.train(max_train_count=100_000)
    tester.eval(runner, episode=100)


def test_Grid_window_length():
    tester = TestRL()
    rl_config = ql_agent57.Config()
    rl_config.enable_actor = False
    rl_config.epsilon.set_constant(0.5)
    rl_config.window_length = 2
    runner = srl.Runner("Grid", rl_config)
    runner.set_seed(3)
    runner.train(max_train_count=50_000)
    tester.eval(runner, episode=100)


def test_Grid_mp():
    tester = TestRL()
    rl_config = ql_agent57.Config()
    rl_config.enable_actor = False
    rl_config.epsilon.set_constant(0.5)
    runner = srl.Runner("Grid", rl_config)
    runner.train_mp(max_train_count=50_000)
    tester.eval(runner, episode=100)


def test_OneRoad():
    tester = TestRL()
    rl_config = ql_agent57.Config()
    runner = srl.Runner("Grid", rl_config)
    runner.set_seed(2)
    runner.train(max_train_count=10_000)
    tester.eval(runner)
