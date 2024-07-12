from srl.envs import igrid  # noqa F401
from srl.test.env import TestEnv  # noqa F401


def test_play():
    tester = TestEnv()
    tester.play_test("IGrid")
