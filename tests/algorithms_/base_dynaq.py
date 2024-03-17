from typing import Tuple

import pytest

import srl
from srl.algorithms import dynaq
from srl.base.rl.config import RLConfig
from srl.test.rl import TestRL
from tests.algorithms_.common_quick_case import CommonQuickCase


class QuickCase(CommonQuickCase):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        return dynaq.Config(), {}


class BaseCase:
    def test_Grid(self):
        tester = TestRL()
        rl_config = dynaq.Config()
        runner = srl.Runner("Grid", rl_config)
        runner.set_seed(1)
        runner.train(max_train_count=50_000)
        tester.eval(runner)

    def test_Grid_mp(self):
        tester = TestRL()
        rl_config = dynaq.Config()
        runner = srl.Runner("Grid", rl_config)
        runner.set_seed(5)
        runner.train_mp(max_train_count=50_000)
        tester.eval(runner)
