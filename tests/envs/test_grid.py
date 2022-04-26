import unittest

from tests.envs.TestEnv import TestEnv  # noqa F402


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestEnv()

    def test_play(self):
        self.tester.play_test(self, "Grid")

    def test_play_2d(self):
        self.tester.play_test(self, "2DGrid")

    def test_play_neon(self):
        self.tester.play_test(self, "NeonGrid")


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_play", verbosity=2)
