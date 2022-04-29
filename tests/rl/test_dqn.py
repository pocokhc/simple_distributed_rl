import unittest

from srl import rl
from srl.test import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()

    def test_play(self):
        rl_config = rl.dqn.Config()
        self.tester.play_test(self, rl_config)

    def test_verify_Grid_DDQN(self):
        rl_config = rl.dqn.Config(
            epsilon=0.5,
            gamma=0.9,
            lr=0.002,
            batch_size=8,
            hidden_layer_sizes=(16, 16),
            enable_double_dqn=True,
        )
        self.tester.play_verify_singleplay(self, "Grid", rl_config, 6000, 100)
        # self.tester.verify_grid_policy(self)

    def test_verify_Grid_noDDQN(self):
        rl_config = rl.dqn.Config(
            epsilon=0.5,
            gamma=0.9,
            lr=0.002,
            batch_size=8,
            hidden_layer_sizes=(16, 16),
            enable_double_dqn=False,
        )
        self.tester.play_verify_singleplay(self, "Grid", rl_config, 6000, 100)

    def test_verify_2DGrid(self):
        rl_config = rl.dqn.Config(
            epsilon=0.5,
            gamma=0.9,
            lr=0.002,
            batch_size=8,
            hidden_layer_sizes=(16, 16),
            enable_double_dqn=True,
        )
        self.tester.play_verify_singleplay(self, "2DGrid", rl_config, 6000, 100)

    def test_verify_Pendulum(self):
        rl_config = rl.dqn.Config(
            hidden_layer_sizes=(64, 64),
        )
        self.tester.play_verify_singleplay(self, "Pendulum-v1", rl_config, 200 * 100, 10)

    def test_verify_IGrid(self):
        # window_length test, 4以上じゃないと学習できない
        rl_config = rl.dqn.Config(
            window_length=4,
            lr=0.001,
            batch_size=8,
            hidden_layer_sizes=(16, 16),
            epsilon=1.0,
        )
        self.tester.play_verify_singleplay(self, "IGrid", rl_config, 10000, 100)

    def test_verify_OX(self):
        rl_config = rl.dqn.Config(
            hidden_layer_sizes=(128,),
            epsilon=0.5,
        )
        self.tester.play_verify_2play(self, "OX", rl_config, 4000, 1000)


if __name__ == "__main__":
    # unittest.main(module=__name__, defaultTest="Test.test_play", verbosity=2)
    unittest.main(module=__name__, defaultTest="Test.test_verify_2DGrid", verbosity=2)
