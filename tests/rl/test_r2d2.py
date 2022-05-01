import unittest

import srl
from srl.test import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()
        self.rl_config = srl.rl.r2d2.Config(multisteps=5)

    def test_sequence(self):
        self.tester.play_sequence(self.rl_config)

    def test_mp(self):
        self.tester.play_mp(self.rl_config)

    def test_verify_Grid(self):
        self.rl_config.epsilon = 0.5
        self.rl_config.burnin = 10
        self.rl_config.multisteps = 3
        self.rl_config.lr = 0.001
        self.rl_config.hidden_layer_sizes = (16,)
        self.rl_config.lstm_units = 32
        self.rl_config.enable_dueling_network = False
        self.rl_config.memory_name = "RankBaseMemory"
        self.rl_config.memory_alpha = 0.8
        self.rl_config.memory_beta_initial = 1.0
        self.tester.play_verify_singleplay("Grid", self.rl_config, 2000, 100)

    def test_verify_Pendulum(self):
        self.rl_config.burnin = 10
        self.rl_config.multisteps = 1
        self.rl_config.hidden_layer_sizes = (16, 16)
        self.rl_config.lstm_units = 64
        self.rl_config.enable_dueling_network = True
        self.rl_config.memory_name = "RankBaseMemory"
        self.rl_config.memory_alpha = 0.8
        self.rl_config.memory_beta_initial = 1.0
        self.tester.play_verify_singleplay("Pendulum-v1", self.rl_config, 200 * 50, 10)

    # def test_verify_IGrid(self):
    #    # LSTMテスト, burninが最低4以上で学習できる
    #    self.rl_config.epsilon = 1.0
    #    self.rl_config.burnin = 10
    #    self.rl_config.multisteps = 1
    #    self.rl_config.dense_units = 16
    #    self.rl_config.lstm_units = 16
    #    self.rl_config.batch_size = 8
    #    self.rl_config.lr = 0.005
    #    self.rl_config.enable_dueling_network = False
    #    self.rl_config.memory_name = "ReplayMemory"
    #    self.tester.play_verify_singleplay("IGrid", self.rl_config, 2000, 100)

    # def test_verify_OX(self):
    #     # invalid action test
    #     self.rl_config.epsilon = 0.5
    #     self.rl_config.multisteps = 3
    #     self.rl_config.burnin = 3
    #     self.rl_config.dense_units = 32
    #     self.rl_config.lstm_units = 32
    #     self.rl_config.enable_dueling_network = False
    #     self.rl_config.memory_name = "ReplayMemory"
    #     self.tester.play_verify_2play("OX", self.rl_config, 10000, 100)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_verify_Pendulum", verbosity=2)
