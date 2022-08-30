import unittest

from envs import grid  # noqa F401
from srl.test import TestEnv


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestEnv()

    def test_play(self):
        self.tester.play_test("Grid")

    def test_play_2d(self):
        self.tester.play_test("2DGrid")

    def test_play_neon(self):
        self.tester.play_test("NeonGrid")


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_play", verbosity=2)
