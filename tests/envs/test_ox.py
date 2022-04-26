import unittest

from tests.envs.TestEnv import TestEnv  # noqa F402


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestEnv()

    def test_play(self):
        self.tester.play_test(self, "OX")

    def test_player(self):
        for player in [
            "cpu_lv1",
            "cpu_lv2",
            "cpu_lv3",
        ]:
            with self.subTest((player,)):
                self.tester.play_player(self, "OX", player)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_play", verbosity=2)
