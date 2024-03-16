from srl.envs import oneroad  # noqa F401
from srl.test import TestEnv


def test_play():
    tester = TestEnv()
    tester.play_test("OneRoad")


def test_play_hard():
    tester = TestEnv()
    tester.play_test("OneRoad-hard")
