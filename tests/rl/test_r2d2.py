import unittest

from srl import rl
from tests.rl.TestRL import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()

    def test_play(self):
        rl_config = rl.r2d2.Config(multisteps=5)
        self.tester.play_test(self, rl_config)

    def test_verify_Grid(self):
        rl_config = rl.r2d2.Config(
            epsilon=0.5,
            burnin=10,
            multisteps=3,
            gamma=0.9,
            lr=0.002,
            dense_units=16,
            lstm_units=16,
            enable_dueling_network=False,
            memory_name="RankBaseMemory",
            memory_alpha=0.8,
            memory_beta_initial=1.0,
        )
        self.tester.play_verify_singleplay(self, "Grid", rl_config, 2000, 100)

    def test_verify_Pendulum(self):
        rl_config = rl.r2d2.Config(
            burnin=10,
            multisteps=1,
            dense_units=64,
            lstm_units=64,
            enable_dueling_network=True,
            memory_name="RankBaseMemory",
            memory_alpha=0.8,
            memory_beta_initial=1.0,
        )
        self.tester.play_verify_singleplay(self, "Pendulum-v1", rl_config, 200 * 50, 10)

    def test_verify_IGrid(self):
        # LSTMテスト, burninが最低4以上で学習できる
        rl_config = rl.r2d2.Config(
            epsilon=1.0,
            burnin=4,
            multisteps=1,
            batch_size=8,
            lr=0.005,
            memory_name="ReplayMemory",
            enable_dueling_network=False,
            dense_units=32,
            lstm_units=32,
        )
        self.tester.play_verify_singleplay(self, "IGrid", rl_config, 2000, 100)

    def test_verify_OX(self):
        # invalid action test
        rl_config = rl.r2d2.Config(
            dense_units=32,
            lstm_units=32,
            burnin=3,
            multisteps=3,
            epsilon=0.5,
            memory_name="ReplayMemory",
            enable_dueling_network=False,
        )
        self.tester.play_verify_2play(self, "OX", rl_config, 6000, 100)


if __name__ == "__main__":
    # unittest.main(module=__name__, defaultTest="Test.test_play", verbosity=2)
    unittest.main(module=__name__, defaultTest="Test.test_verify_OX", verbosity=2)
