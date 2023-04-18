from srl.envs import stone_taking  # noqa F401
from srl.test import TestEnv


def test_play():
    tester = TestEnv()
    tester.play_test("StoneTaking")
