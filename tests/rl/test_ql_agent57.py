import unittest

from srl import rl
from tests.rl.TestRL import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()

    def test_play(self):
        rl_config = rl.ql_agent57.Config(multisteps=3)
        self.tester.play_test(self, rl_config)

    def test_verify_grid(self):
        rl_config = rl.ql_agent57.Config()
        self.tester.play_verify_singleplay(self, "Grid", rl_config, 100_000, 1000)
        self.tester.verify_grid_action_values(self)
        self.tester.verify_grid_policy(self)

    def test_verify_ox(self):
        rl_config = rl.ql_agent57.Config()
        self.tester.play_verify_2play(self, "OX", rl_config, 100_000, 1000)


if __name__ == "__main__":
    # unittest.main(module=__name__, defaultTest="Test.test_play", verbosity=2)
    unittest.main(module=__name__, defaultTest="Test.test_verify_grid", verbosity=2)
