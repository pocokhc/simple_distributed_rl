import unittest

import srl
from srl.test import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()
        self.rl_config = srl.rl.alphazero.Config()
        self.rl_config.simulation_times = 10

    def test_sequence(self):
        self.tester.play_sequence(self.rl_config)

    def test_mp(self):
        self.tester.play_mp(self.rl_config)

    # def test_verify_grid(self):
    #    self.rl_config.hidden_layer_sizes = (64, 64, 64)
    #    self.rl_config.simulation_times = 100
    #    self.rl_config.train_size = 50
    #    self.rl_config.early_steps = 1
    #    self.rl_config.gamma = 0.9
    #    self.rl_config.action_select_threshold = 5
    #    self.rl_config.epochs = 20
    #    self.tester.play_verify_singleplay("Grid", self.rl_config, 100, 20)

    # def test_verify_StoneTaking(self):
    #    self.tester.play_verify_2play("StoneTaking", self.rl_config, 2000, 100)

    def test_verify_ox(self):
        self.rl_config.hidden_layer_sizes = (64, 64, 64)
        self.rl_config.simulation_times = 100
        self.rl_config.train_size = 100
        self.rl_config.early_steps = 1
        self.rl_config.gamma = 1.0
        self.rl_config.action_select_threshold = 5
        self.rl_config.epochs = 10
        self.tester.play_verify_2play("OX", self.rl_config, 5000, 20)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_verify_ox", verbosity=2)
