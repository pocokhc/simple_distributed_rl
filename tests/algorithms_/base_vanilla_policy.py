from typing import Tuple

import srl
from srl.algorithms import vanilla_policy
from srl.base.define import RLTypes
from srl.base.rl.config import RLConfig
from srl.test.rl import TestRL
from tests.algorithms_.common_quick_case import CommonQuickCase


class QuickCase_dis(CommonQuickCase):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        return vanilla_policy.Config(), {}


class QuickCase_con(CommonQuickCase):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        rl_config = vanilla_policy.Config()
        rl_config.override_action_type = RLTypes.CONTINUOUS
        return rl_config, {}


class BaseCase:
    def test_Grid_discrete(self):
        tester = TestRL()
        rl_config = vanilla_policy.Config()
        runner = srl.Runner("Grid", rl_config)
        runner.set_seed(1)
        runner.train(max_train_count=10_000)
        tester.eval(runner)

    def test_Grid_continuous(self):
        tester = TestRL()
        rl_config = vanilla_policy.Config()
        rl_config.override_action_type = RLTypes.CONTINUOUS
        runner = srl.Runner("Grid", rl_config)
        runner.set_seed(1)
        runner.train(max_train_count=500_000)
        tester.eval(runner)
