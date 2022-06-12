import unittest

import srl
from srl.test import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()
        self.rl_config = srl.rl.ql_agent57.Config(multisteps=3)
        self.rl_config.memory_beta_initial = 1.0

    def test_sequence(self):
        self.tester.play_sequence(self.rl_config)

    def test_mp(self):
        self.tester.play_mp(self.rl_config)

    def test_verify_grid(self):
        self.rl_config.enable_actor = False
        self.rl_config.epsilon = 0.5
        self.tester.play_verify_singleplay("Grid", self.rl_config, 80_000)
        self.tester.verify_grid_policy()

    def test_verify_oneroad(self):
        self.tester.play_verify_singleplay("OneRoad", self.rl_config, 20_000)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_verify_grid", verbosity=2)
