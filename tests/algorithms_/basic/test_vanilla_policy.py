import unittest

from srl import runner
from srl.algorithms import vanilla_policy
from srl.base.define import RLTypes
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
    config = runner.Config("Grid", rl_config, seed=1)
    tester.train_eval(config, 10_000, eval_episode=100)


def test_Grid_continuous():
    tester = TestRL()
    rl_config = vanilla_policy.Config()
    rl_config.override_action_type = RLTypes.CONTINUOUS
    config = runner.Config("Grid", rl_config, seed=1)
    tester.train_eval(config, 100_000, eval_episode=100)
