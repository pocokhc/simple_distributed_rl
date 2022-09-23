import unittest

import srl.envs.grid  # noqa F401
from srl.algorithms import vanilla_policy_discrete
from srl.test import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()

    def test_simple_check(self):
        self.tester.simple_check(vanilla_policy_discrete.Config())

    def test_simple_check_mp(self):
        self.tester.simple_check_mp(vanilla_policy_discrete.Config())

    def test_verify_grid(self):
        rl_config = vanilla_policy_discrete.Config()
        self.tester.verify_singleplay("Grid", rl_config, 10_000)
        # self.tester.verify_grid_policy()


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_verify_grid", verbosity=2)
