from typing import Tuple

import srl
from srl.algorithms import search_dynaq
from srl.base.rl.config import RLConfig
from srl.test import TestRL
from tests.algorithms_.common_base_class import CommonBaseSimpleTest


class Test_search_dynaq(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        return search_dynaq.Config(), {}


def test_Grid():
    tester = TestRL()
    rl_config = search_dynaq.Config()
    rl_config.lr_ext.set_constant(0.01)
    runner = srl.Runner("Grid", rl_config)
    runner.set_seed(1)
    runner.train(max_train_count=5_000)
    rl_config.search_mode = False
    runner.train_only(max_train_count=5_000)
    tester.eval(runner)


def test_Grid_mp():
    tester = TestRL()
    rl_config = search_dynaq.Config()
    rl_config.lr_ext.set_constant(0.01)
    runner = srl.Runner("Grid", rl_config)
    runner.set_seed(1)
    runner.train_mp(max_train_count=5_000)
    rl_config.search_mode = False
    runner.train_only(max_train_count=5_000)
    tester.eval(runner)


def test_OneRoad():
    tester = TestRL()
    rl_config = search_dynaq.Config()
    runner = srl.Runner("OneRoad", rl_config)
    runner.set_seed(4)
    runner.train(max_train_count=2_000)
    rl_config.search_mode = False
    tester.eval(runner)
