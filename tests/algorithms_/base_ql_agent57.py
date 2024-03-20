from typing import Tuple

import srl
from srl.algorithms import ql_agent57
from srl.base.rl.config import RLConfig
from srl.test.rl import TestRL
from tests.algorithms_.common_quick_case import CommonQuickCase


class QuickCase(CommonQuickCase):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        return ql_agent57.Config(), {}


class BaseCase:
    def test_Grid(self):
        tester = TestRL()
        rl_config = ql_agent57.Config()
        rl_config.enable_actor = False
        rl_config.epsilon = 0.5
        runner = srl.Runner("Grid", rl_config)
        runner.set_seed(3)
        runner.train(max_train_count=100_000)
        tester.eval(runner, episode=100)

    def test_Grid_window_length(self):
        tester = TestRL()
        rl_config = ql_agent57.Config()
        rl_config.enable_actor = False
        rl_config.epsilon = 0.5
        rl_config.window_length = 2
        runner = srl.Runner("Grid", rl_config)
        runner.set_seed(3)
        runner.train(max_train_count=50_000)
        tester.eval(runner, episode=100)

    def test_Grid_mp(self):
        tester = TestRL()
        rl_config = ql_agent57.Config()
        rl_config.enable_actor = False
        rl_config.epsilon = 0.5
        runner = srl.Runner("Grid", rl_config)
        runner.train_mp(max_train_count=50_000)
        tester.eval(runner, episode=100)
