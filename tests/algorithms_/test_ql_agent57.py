import unittest

from srl.algorithms import ql_agent57
from srl.test import TestRL
import srl.envs.grid  # noqa F401
import srl.envs.oneroad  # noqa F401


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()
        self.rl_config = ql_agent57.Config()

    def test_simple_check(self):
        self.tester.simple_check(ql_agent57.Config())

    def test_simple_check_mp(self):
        self.tester.simple_check_mp(ql_agent57.Config())

    def test_verify_grid(self):
        self.rl_config.enable_actor = False
        self.rl_config.epsilon = 0.5
        self.tester.verify_singleplay("Grid", self.rl_config, 50_000)
        self.tester.verify_grid_policy()

    def test_verify_grid_window_length(self):
        self.rl_config.enable_actor = False
        self.rl_config.epsilon = 0.5
        self.rl_config.window_length = 2
        self.tester.verify_singleplay("Grid", self.rl_config, 30_000)

    def test_verify_grid_mp(self):
        self.rl_config.enable_actor = False
        self.rl_config.epsilon = 0.5
        self.tester.verify_singleplay("Grid", self.rl_config, 50_000, is_mp=True)
        self.tester.verify_grid_policy()

    def test_verify_oneroad(self):
        self.tester.verify_singleplay("OneRoad", self.rl_config, 20_000)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_verify_grid_window_length", verbosity=2)
