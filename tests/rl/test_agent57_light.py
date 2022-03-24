import unittest

from srl import rl
from tests.rl.TestRL import TestRL


class Test(unittest.TestCase):
    def test_play(self):
        rl_config = rl.agent57_light.Config()

        tester = TestRL()
        tester.play_test(self, rl_config)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_play", verbosity=2)
