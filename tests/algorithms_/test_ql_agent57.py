import unittest

import srl.envs.grid  # noqa F401
import srl.envs.oneroad  # noqa F401
from srl.algorithms import ql_agent57
from srl.test import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()
        self.rl_config = ql_agent57.Config()

    def test_Grid(self):
        self.rl_config.enable_actor = False
        self.rl_config.epsilon = 0.5
        parameter = self.tester.verify_1player("Grid", self.rl_config, 50_000)
        self.tester.verify_grid_policy(self.rl_config, parameter)

    def test_Grid_window_length(self):
        self.rl_config.enable_actor = False
        self.rl_config.epsilon = 0.5
        self.rl_config.window_length = 2
        self.tester.verify_1player("Grid", self.rl_config, 30_000)

    def test_Grid_mp(self):
        self.rl_config.enable_actor = False
        self.rl_config.epsilon = 0.5
        parameter = self.tester.verify_1player("Grid", self.rl_config, 50_000, is_mp=True)
        self.tester.verify_grid_policy(self.rl_config, parameter)

    def test_OneRoad(self):
        self.tester.verify_1player("OneRoad", self.rl_config, 20_000)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_verify_oneroad", verbosity=2)
