from srl.envs import sample_env  # noqa F401
from srl.test import TestEnv


def test_env():
    tester = TestEnv()
    tester.play_test("SampleEnv")
