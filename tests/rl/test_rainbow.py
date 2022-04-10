import unittest

from srl import rl
from tests.rl.TestRL import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()

    def test_play(self):
        rl_config = rl.rainbow.Config()
        self.tester.play_test(self, rl_config)

    def test_verify_IGrid(self):
        rl_config = rl.rainbow.Config(
            window_length=4,
            multisteps=1,
            dense_units=16,
            epsilon=0.5,
        )
        self.tester.play_verify(self, "IGrid-v0", rl_config, 15000, 100)

    def test_verify_Pong(self):
        rl_config = rl.rainbow.Config(
            window_length=4,
            multisteps=10,
        )
        self.tester.play_verify(self, "ALE/Pong-v5", rl_config, 15000, 100, is_atari=True)

    def test_verify_OX(self):
        rl_config = rl.rainbow.Config(
            dense_units=64,
            multisteps=3,
            epsilon=0.9,
            enable_dueling_network=False,
        )
        self.tester.play_verify(self, "OX-v0", rl_config, 4000, 100)


if __name__ == "__main__":
    # unittest.main(module=__name__, defaultTest="Test.test_play", verbosity=2)
    unittest.main(module=__name__, defaultTest="Test.test_verify_OX", verbosity=2)
