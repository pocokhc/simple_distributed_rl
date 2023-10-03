from typing import Tuple

from srl.algorithms import vanilla_policy
from srl.base.define import RLTypes
from srl.base.rl.config import RLConfig
from srl.runner.runner import Runner
from srl.test import TestRL
from tests.algorithms_.common_base_class import CommonBaseSimpleTest


class Test_vanilla_policy_discrete(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        return vanilla_policy.Config(), {}


class Test_vanilla_policy_continuous(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        rl_config = vanilla_policy.Config()
        rl_config.override_action_type = RLTypes.CONTINUOUS
        return rl_config, {}


def test_Grid_discrete():
    tester = TestRL()
    rl_config = vanilla_policy.Config()
    runner = Runner("Grid", rl_config)
    runner.set_seed(1)
    runner.train(max_train_count=10_000)
    tester.eval(runner)


def test_Grid_continuous():
    tester = TestRL()
    rl_config = vanilla_policy.Config()
    rl_config.override_action_type = RLTypes.CONTINUOUS
    runner = Runner("Grid", rl_config)
    runner.set_seed(1)
    runner.train(max_train_count=100_000)
    tester.eval(runner)
