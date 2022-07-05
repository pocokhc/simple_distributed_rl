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
        self.rl_config.hidden_block_kwargs = dict(hidden_layer_sizes=(16, 16))
        self.rl_config.categorical_num_atoms = 11
        self.rl_config.categorical_v_min = -2
        self.rl_config.categorical_v_max = 2
        self.tester.play_verify_singleplay("Grid", self.rl_config, 6000)

    def test_verify_Pendulum(self):
        self.rl_config.hidden_block_kwargs = dict(hidden_layer_sizes=(64, 64))
        self.tester.play_verify_singleplay("Pendulum-v1", self.rl_config, 200 * 60)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_verify_Pendulum", verbosity=2)
