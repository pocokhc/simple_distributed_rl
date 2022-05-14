import unittest

import srl
from srl.test import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()
        self.rl_config = srl.rl.ql.Config()

    def test_sequence(self):
        self.tester.play_sequence(self.rl_config)

    def test_mp(self):
        self.tester.play_mp(self.rl_config)

    def test_verify_grid(self):
        self.rl_config.epsilon = 0.5
        self.rl_config.lr = 0.01
        self.tester.play_verify_singleplay("Grid", self.rl_config, 50_000, 1000)
        self.tester.verify_grid_policy()

    def test_verify_ox(self):
        self.rl_config.epsilon = 0.5
        self.rl_config.lr = 0.1
        self.tester.play_verify_2play("OX", self.rl_config, 100_000, 1000)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_verify_grid", verbosity=2)
