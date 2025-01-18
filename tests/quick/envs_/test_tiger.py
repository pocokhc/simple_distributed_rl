from srl.envs import tiger  # noqa F401
from srl.test.env import env_test


def test_play():
    env_test("Tiger")
