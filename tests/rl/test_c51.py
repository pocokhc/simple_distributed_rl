import unittest

import srl
from srl.test import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()
        self.rl_config = srl.rl.c51.Config()

    def test_sequence(self):
        self.tester.play_sequence(self.rl_config)

    def test_mp(self):
        self.tester.play_mp(self.rl_config)

    def test_verify_Grid(self):
        self.rl_config.epsilon = 0.5
        self.rl_config.lr = 0.002
        self.rl_config.hidden_layer_sizes = (16, 16)
        self.rl_config.categorical_num_atoms = 11
        self.rl_config.categorical_v_min = -2
        self.rl_config.categorical_v_max = 2
        self.tester.play_verify_singleplay("Grid", self.rl_config, 6000, 100)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_verify_Grid", verbosity=2)
