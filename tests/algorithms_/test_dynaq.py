import unittest

import srl.envs.grid  # noqa F401
from srl.algorithms import dynaq
from srl.test import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()

    def test_Grid(self):
        rl_config = dynaq.Config()
        parameter = self.tester.verify_1player("Grid", rl_config, 50_000)
        self.tester.verify_grid_policy(rl_config, parameter)

    def test_Grid_mp(self):
        rl_config = dynaq.Config()
        parameter = self.tester.verify_1player("Grid", rl_config, 100_000, is_mp=True)
        self.tester.verify_grid_policy(rl_config, parameter)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_verify_grid_mp", verbosity=2)
