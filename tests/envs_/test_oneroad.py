import unittest

from srl.test import TestEnv
from envs import oneroad  # noqa F401


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestEnv()

    def test_play(self):
        self.tester.play_test("OneRoad")

    def test_play_hard(self):
        self.tester.play_test("OneRoad-hard")


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_play_hard", verbosity=2)
