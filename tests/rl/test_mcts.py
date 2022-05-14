import unittest

import srl
from srl.test import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()
        self.rl_config = srl.rl.mcts.Config()
        self.rl_config.simulation_times = 10

    def test_sequence(self):
        self.tester.play_sequence(self.rl_config)

    def test_mp(self):
        self.tester.play_mp(self.rl_config)

    def test_verify_grid(self):  # 50%
        self.tester.play_verify_singleplay("Grid", self.rl_config, 5000, 1000)

    def test_verify_StoneTaking(self):
        self.tester.play_verify_2play("StoneTaking", self.rl_config, 2000, 100)

    def test_verify_ox(self):  # 成功なし
        self.tester.play_verify_2play("OX", self.rl_config, 50000, 100)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_sequence", verbosity=2)
