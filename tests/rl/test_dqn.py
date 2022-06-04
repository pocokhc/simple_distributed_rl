import unittest

import srl
from srl.test import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()
        self.rl_config = srl.rl.dqn.Config()

    def test_sequence(self):
        self.tester.play_sequence(self.rl_config)

    def test_mp(self):
        self.tester.play_mp(self.rl_config)

    def test_verify_Grid(self):
        self.rl_config.epsilon = 0.5
        self.rl_config.gamma = 0.99
        self.rl_config.lr = 0.01
        self.rl_config.batch_size = 32
        self.rl_config.hidden_layer_sizes = (32, 32, 32)
        self.rl_config.enable_rescale = False
        self.tester.play_verify_singleplay("Grid", self.rl_config, 10000, 100)

    def test_verify_Pendulum(self):
        self.rl_config.hidden_layer_sizes = (64, 64)
        self.rl_config.enable_double_dqn = False
        self.tester.play_verify_singleplay("Pendulum-v1", self.rl_config, 200 * 100, 10)

    def test_verify_Pendulum_DDQN(self):
        self.rl_config.hidden_layer_sizes = (64, 64)
        self.tester.play_verify_singleplay("Pendulum-v1", self.rl_config, 200 * 70, 10)

    def test_verify_Pendulum_window(self):
        self.rl_config.window_length = 4
        self.rl_config.hidden_layer_sizes = (64, 64)
        self.tester.play_verify_singleplay("Pendulum-v1", self.rl_config, 200 * 70, 10)

    def test_verify_OX(self):
        self.rl_config.hidden_layer_sizes = (128,)
        self.rl_config.epsilon = 0.5
        self.tester.play_verify_2play("OX", self.rl_config, 10000, 1000)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_verify_Pendulum_window", verbosity=2)
