from srl.envs import tiger  # noqa F401
from srl.test.env import TestEnv


def test_play():
    tester = TestEnv()
    tester.play_test("Tiger")
