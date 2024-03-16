from srl.envs import igrid  # noqa F401
from srl.test import TestEnv


def test_play():
    tester = TestEnv()
    tester.play_test("IGrid")
