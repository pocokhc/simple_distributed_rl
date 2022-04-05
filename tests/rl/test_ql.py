import unittest

from srl import rl
from tests.rl.TestRL import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()

    def test_play(self):
        rl_config = rl.ql.Config()
        self.tester.play_test(self, rl_config)

    def test_verify(self):
        rl_config = rl.ql.Config(epsilon=0.5, lr=0.01)
        self.tester.play_verify(self, "Grid-v0", rl_config, 50_000, 1000)
        self.tester.check_action_values(self)
        self.tester.check_policy(self)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_verify", verbosity=2)
