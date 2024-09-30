import pytest

from srl.test.env import TestEnv


def test_play():
    pytest.skip("元が動かない")
    pytest.importorskip("kaggle_environments")
    from srl.envs.kaggle import hungry_geese  # noqa F401

    tester = TestEnv()
    tester.play_test("hungry_geese")


@pytest.mark.parametrize("player", ["greedy"])
def test_player(player):
    pytest.skip("元が動かない")
    pytest.importorskip("kaggle_environments")
    from srl.envs.kaggle import hungry_geese  # noqa F401

    tester = TestEnv()
    tester.player_test("hungry_geese", player)
