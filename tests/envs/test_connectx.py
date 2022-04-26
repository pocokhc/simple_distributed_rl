import unittest

from tests.envs.TestEnv import TestEnv  # noqa F402


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestEnv()

    def test_play(self):
        self.tester.play_test(self, "ConnectX")

    def test_player(self):
        for player in [
            "negamax",
        ]:
            with self.subTest((player,)):
                self.tester.play_player(self, "ConnectX", player)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_player", verbosity=2)
