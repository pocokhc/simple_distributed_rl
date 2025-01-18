from srl.envs import igrid  # noqa F401
from srl.test.env import env_test


def test_play():
    env_test("IGrid")
