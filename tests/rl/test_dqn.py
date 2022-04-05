import unittest

from srl import rl
from tests.rl.TestRL import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()

    def test_play(self):
        rl_config = rl.dqn.Config()
        self.tester.play_test(self, rl_config)

    def test_verify_Grid(self):
        rl_config = rl.dqn.Config(
            epsilon=0.5,
            gamma=0.9,
            lr=0.002,
            batch_size=8,
            hidden_layer_sizes=(16, 16),
        )
        self.tester.play_verify(self, "Grid-v0", rl_config, 5000, 100)
        self.tester.check_policy(self)

    def test_verify_Pendulum(self):
        rl_config = rl.dqn.Config(
            hidden_layer_sizes=(64, 64),
        )
        self.tester.play_verify(self, "Pendulum-v1", rl_config, 20000, 100)

    def test_verify_IGrid(self):
        rl_config = rl.dqn.Config(
            window_length=4,
            hidden_layer_sizes=(8, 8),
            epsilon=0.5,
        )
        self.tester.play_verify(self, "IGrid-v0", rl_config, 15000, 100)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_verify_IGrid", verbosity=2)
