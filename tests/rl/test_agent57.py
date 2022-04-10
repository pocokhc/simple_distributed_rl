import unittest

from srl import rl
from tests.rl.TestRL import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()

    def test_play(self):
        rl_config = rl.agent57.Config(multisteps=5)
        self.tester.play_test(self, rl_config)

    def test_verify_Grid(self):
        rl_config = rl.agent57.Config(
            burnin=5,
            multisteps=5,
            dense_units=16,
            lstm_units=32,
            dueling_dense_units=16,
        )
        self.tester.play_verify(self, "Grid-v0", rl_config, 2000, 10)

    def test_verify_Pendulum(self):
        rl_config = rl.agent57.Config(
            burnin=5,
            multisteps=5,
            dense_units=64,
            lstm_units=64,
            enable_dueling_network=False,
        )
        self.tester.play_verify(self, "Pendulum-v1", rl_config, 200 * 30, 10)

    def test_verify_IGrid(self):
        rl_config = rl.agent57.Config(
            burnin=4,
            multisteps=1,
            memory_name="ReplayMemory",
            enable_dueling_network=False,
            dense_units=16,
            lstm_units=64,
        )
        self.tester.play_verify(self, "IGrid-v0", rl_config, 5000, 10)


if __name__ == "__main__":
    # unittest.main(module=__name__, defaultTest="Test.test_play", verbosity=2)
    unittest.main(module=__name__, defaultTest="Test.test_verify_Grid", verbosity=2)
