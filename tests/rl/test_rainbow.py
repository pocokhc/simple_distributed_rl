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
            multisteps=1,
            window_length=4,
            dense_units=16,
            epsilon=0.5,
        )
        self.tester.play_verify(self, "IGrid-v0", rl_config, 15000, 100)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_play", verbosity=2)
