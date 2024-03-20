from srl.envs import igrid
from srl.test.env import TestEnv  # noqa F401


def test_play():
    tester = TestEnv()
    tester.play_test("IGrid")
