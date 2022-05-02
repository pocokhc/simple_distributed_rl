import unittest

import srl
from srl.test import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()
        self.rl_config = srl.rl.mcts.Config()

    def test_sequence(self):
        self.tester.play_sequence(self.rl_config)

    def test_mp(self):
        self.tester.play_mp(self.rl_config)

    def test_verify_grid(self):
        self.tester.play_verify_singleplay("Grid", self.rl_config, 20000, 1000)

    # def test_verify_ox(self):
    #    self.tester.play_verify_2play("OX", self.rl_config, 20000, 1000)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_verify_ox", verbosity=2)
