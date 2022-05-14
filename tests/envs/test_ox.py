import unittest

from srl.test import TestEnv


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestEnv()

    def test_play(self):
        self.tester.play_test("OX")

    def test_player(self):
        for player in [
            "cpu",
        ]:
            with self.subTest((player,)):
                self.tester.player_test("OX", player)

    # processor TODO


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_player", verbosity=2)
