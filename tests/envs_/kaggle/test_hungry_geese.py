import pytest

from srl.test import TestEnv

try:
    from srl.envs.kaggle import hungry_geese  # noqa F401
except ModuleNotFoundError as e:
    print(e)


def test_play():
    pytest.importorskip("kaggle_environments")

    tester = TestEnv()
    tester.play_test("hungry_geese")


@pytest.mark.parametrize("player", ["greedy"])
def test_player(player):
    pytest.importorskip("kaggle_environments")

    tester = TestEnv()
    tester.player_test("hungry_geese", player)
