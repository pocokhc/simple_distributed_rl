import unittest

from srl.algorithms import vanilla_policy
from srl.base.define import RLActionType
from srl.test import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()

    def test_Grid_discrete(self):
        rl_config = vanilla_policy.Config()
        self.tester.verify_1player("Grid", rl_config, 10_000)

    def test_Grid_continuous(self):
        rl_config = vanilla_policy.Config()
        rl_config.override_rl_action_type = RLActionType.CONTINUOUS
        self.tester.verify_1player("Grid", rl_config, 150_000)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_verify_grid_continuous", verbosity=2)
