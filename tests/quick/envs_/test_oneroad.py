from srl.envs import oneroad  # noqa F401
from srl.test.env import env_test


def test_play():
    env_test("OneRoad")


def test_play_hard():
    env_test("OneRoad-hard")
