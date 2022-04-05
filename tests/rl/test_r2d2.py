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
            burnin=5,
            multisteps=5,
            gamma=0.9,
            lr=0.001,
            dense_units=16,
            lstm_units=16,
        )
        self.tester.play_verify(self, "Grid-v0", rl_config, 2000, 100)

    def test_verify_Pendulum(self):
        rl_config = rl.r2d2.Config(
            burnin=5,
            multisteps=1,
            dense_units=64,
            lstm_units=64,
            enable_dueling_network=True,
            batch_size=16,
        )
        self.tester.play_verify(self, "Pendulum-v1", rl_config, 5000, 100)

    def test_verify_IGrid(self):
        rl_config = rl.r2d2.Config(
            epsilon=0.5,
            burnin=4,
            multisteps=1,
            memory_name="ReplayMemory",
            enable_dueling_network=False,
            dense_units=16,
            lstm_units=16,
        )
        self.tester.play_verify(self, "IGrid-v0", rl_config, 5000, 100)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_verify_Pendulum", verbosity=2)
