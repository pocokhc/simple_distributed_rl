from srl.envs import oneroad
from srl.test.env import TestEnv  # noqa F401


def test_play():
    tester = TestEnv()
    tester.play_test("OneRoad")


def test_play_hard():
    tester = TestEnv()
    tester.play_test("OneRoad-hard")
