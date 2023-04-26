import numpy as np
import pytest

from srl.base.define import EnvObservationType
from srl.base.env.spaces.box import BoxSpace
from srl.test import TestEnv
from srl.test.processor import TestProcessor

try:
    from srl.envs.kaggle import connectx  # noqa F401
except ModuleNotFoundError as e:
    print(e)


def test_play():
    pytest.importorskip("kaggle_environments")

    tester = TestEnv()
    env = tester.play_test("connectx")

    for x in [0, 2, 3, 4, 5, 6]:
        env.reset()
        board = [0] * 42
        assert not env.done
        assert env.state == board

        env.step(x)
        board[x + (5 * 7)] = 1
        assert not env.done
        assert (env.step_rewards == [0, 0]).all()
        assert env.state == board

        env.step(1)
        board[1 + (5 * 7)] = 2
        assert not env.done
        assert (env.step_rewards == [0, 0]).all()
        assert env.state == board

        env.step(x)
        board[x + (4 * 7)] = 1
        assert not env.done
        assert (env.step_rewards == [0, 0]).all()
        assert env.state == board

        env.step(1)
        board[1 + (4 * 7)] = 2
        assert not env.done
        assert (env.step_rewards == [0, 0]).all()
        assert env.state == board

        env.step(x)
        board[x + (3 * 7)] = 1
        assert not env.done
        assert (env.step_rewards == [0, 0]).all()
        assert env.state == board

        env.step(1)
        board[1 + (3 * 7)] = 2
        assert not env.done
        assert (env.step_rewards == [0, 0]).all()
        assert env.state == board

        env.step(x)
        board[x + (2 * 7)] = 1
        assert env.done
        assert (env.step_rewards == [1, -1]).all()
        assert env.state == board


@pytest.mark.parametrize("player", ["negamax"])
def test_player(player):
    pytest.importorskip("kaggle_environments")

    tester = TestEnv()
    tester.player_test("connectx", player)


def test_processor():
    pytest.importorskip("kaggle_environments")

    tester = TestProcessor()
    processor = connectx.LayerProcessor()
    env_name = "connectx"
    columns = 7
    rows = 6

    in_state = [0] * 42
    out_state = np.zeros((2, rows, columns))

    tester.run(processor, env_name)
    tester.change_observation_info(
        processor,
        env_name,
        EnvObservationType.SHAPE3,
        BoxSpace((2, rows, columns), 0, 1),
    )
    tester.observation_decode(
        processor,
        env_name,
        in_observation=in_state,
        out_observation=out_state,
    )
