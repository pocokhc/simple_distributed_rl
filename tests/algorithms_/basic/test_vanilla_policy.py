import unittest

from srl.algorithms import vanilla_policy
from srl.base.define import RLTypes
from srl.runner.runner import Runner
from srl.test import TestRL


class Test_vanilla_policy_discrete(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        from srl.algorithms import vanilla_policy

        self.rl_config = vanilla_policy.Config()


class Test_vanilla_policy_continuous(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        from srl.algorithms import vanilla_policy

        self.rl_config = vanilla_policy.Config()
        self.rl_config.override_action_type = RLTypes.CONTINUOUS


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
