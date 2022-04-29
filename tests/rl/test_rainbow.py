import unittest

from srl import rl
from srl.test import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()

    def test_play(self):
        rl_config = rl.rainbow.Config()
        self.tester.play_test(self, rl_config)

    def test_verify_IGrid(self):
        # window_length test, 4以上じゃないと学習できない
        rl_config = rl.rainbow.Config(
            window_length=4,
            multisteps=1,
            lr=0.005,
            dense_units=32,
            batch_size=8,
            epsilon=0.5,
            enable_dueling_network=False,
            memory_name="ReplayMemory",
        )
        self.tester.play_verify_singleplay(self, "IGrid", rl_config, 10000, 100)

    def test_verify_Pendulum(self):
        rl_config = rl.rainbow.Config(
            dense_units=128,
        )
        self.tester.play_verify_singleplay(self, "Pendulum-v1", rl_config, 200 * 100, 10)

    # TODO
    # def test_verify_Pong(self):
    #    rl_config = rl.rainbow.Config(
    #        window_length=4,
    #        multisteps=10,
    #    )
    #    self.tester.play_verify_singleplay(self, "ALE/Pong-v5", rl_config, 15000, 10, is_atari=True)

    def test_verify_OX(self):
        # invalid action test
        rl_config = rl.rainbow.Config(
            dense_units=64,
            multisteps=3,
            epsilon=0.5,
            enable_dueling_network=False,
        )
        self.tester.play_verify_2play(self, "OX", rl_config, 4000, 100)


class TestGrid(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()
        self.rl_config = rl.rainbow.Config(
            epsilon=0.5,
            gamma=0.9,
            lr=0.005,
            batch_size=8,
            dense_units=16,
            window_length=1,
            enable_double_dqn=False,
            enable_dueling_network=False,
            enable_noisy_dense=False,
            multisteps=1,
            memory_name="ReplayMemory",
        )

    def test_verify_Grid_naive(self):
        self.tester.play_verify_singleplay(self, "Grid", self.rl_config, 12000, 100)

    def test_verify_Grid_ddqn(self):
        self.rl_config.enable_double_dqn = True
        self.tester.play_verify_singleplay(self, "Grid", self.rl_config, 12000, 100)

    def test_verify_Grid_dueling(self):
        self.rl_config.dense_units = 16
        self.rl_config.batch_size = 16
        self.rl_config.lr = 0.001
        self.rl_config.enable_dueling_network = True
        self.tester.play_verify_singleplay(self, "Grid", self.rl_config, 7000, 100)

    def test_verify_Grid_noisy(self):
        self.rl_config.enable_noisy_dense = True
        self.tester.play_verify_singleplay(self, "Grid", self.rl_config, 7000, 100)

    def test_verify_Grid_multistep(self):
        self.rl_config.multisteps = 10
        self.tester.play_verify_singleplay(self, "Grid", self.rl_config, 8000, 100)

    def test_verify_Grid_rankbase(self):
        self.rl_config.memory_name = "RankBaseMemory"
        self.rl_config.memory_alpha = 1.0
        self.rl_config.memory_beta_initial = 1.0
        self.tester.play_verify_singleplay(self, "Grid", self.rl_config, 6000, 100)

    def test_verify_Grid_proportional(self):
        self.rl_config.memory_name = "ProportionalMemory"
        self.rl_config.memory_alpha = 0.6
        self.rl_config.memory_beta_initial = 1.0
        self.tester.play_verify_singleplay(self, "Grid", self.rl_config, 8000, 100)

    def test_verify_Grid_all(self):
        self.rl_config.enable_double_dqn = True
        self.rl_config.lr = 0.001
        self.rl_config.dense_units = 16
        self.rl_config.batch_size = 8
        self.rl_config.enable_dueling_network = True
        # self.rl_config.enable_noisy_dense = True
        self.rl_config.multisteps = 5
        self.rl_config.memory_name = "RankBaseMemory"
        self.rl_config.memory_alpha = 1.0
        self.rl_config.memory_beta_initial = 1.0
        self.tester.play_verify_singleplay(self, "Grid", self.rl_config, 5000, 100)


if __name__ == "__main__":
    # unittest.main(module=__name__, defaultTest="Test.test_play", verbosity=2)
    unittest.main(module=__name__, defaultTest="Test.test_verify_IGrid", verbosity=2)
    # unittest.main(module=__name__, defaultTest="TestGrid.test_verify_Grid_all", verbosity=2)
