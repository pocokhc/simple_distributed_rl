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

    def test_Pendulum(self):
        self.rl_config.hidden_block_kwargs = dict(hidden_layer_sizes=(64, 64))
        self.rl_config.enable_double_dqn = False
        self.tester.play_verify_singleplay("Pendulum-v1", self.rl_config, 200 * 100)

    def test_Pendulum_mp(self):
        self.rl_config.hidden_block_kwargs = dict(hidden_layer_sizes=(64, 64))
        self.tester.play_verify_singleplay("Pendulum-v1", self.rl_config, 200 * 100, is_mp=True)

    def test_Pendulum_DDQN(self):
        self.rl_config.hidden_block_kwargs = dict(hidden_layer_sizes=(64, 64))
        self.tester.play_verify_singleplay("Pendulum-v1", self.rl_config, 200 * 70)

    def test_Pendulum_window(self):
        self.rl_config.window_length = 4
        self.rl_config.hidden_block_kwargs = dict(hidden_layer_sizes=(64, 64))
        self.tester.play_verify_singleplay("Pendulum-v1", self.rl_config, 200 * 70)

    def test_OX(self):
        self.rl_config.hidden_block_kwargs = dict(hidden_layer_sizes=(128,))
        self.rl_config.epsilon = 0.5
        self.tester.play_verify_2play("OX", self.rl_config, 10000)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_Grid", verbosity=2)
