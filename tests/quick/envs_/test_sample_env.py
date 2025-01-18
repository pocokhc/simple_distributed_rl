from srl.envs import sample_env  # noqa F401
from srl.test.env import env_test


def test_simple_env():
    env_test("SampleEnv")
