import numpy as np
import pytest

from srl.base.define import EnvObservationType
from srl.base.env.spaces.box import BoxSpace
from srl.envs import connectx  # noqa F401
from srl.test import TestEnv
from srl.test.processor import TestProcessor


def test_play():
    tester = TestEnv()
    env = tester.play_test("ConnectX")

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


@pytest.mark.parametrize(
    "player",
    [
        "alphabeta6",
        # "alphabeta7",
        # "alphabeta8",
        # "alphabeta9",
        # "alphabeta10",
    ],
)
def test_player(player):
    tester = TestEnv()
    tester.player_test("ConnectX", player)


def test_processor():
    tester = TestProcessor()
    processor = connectx.LayerProcessor()
    env_name = "ConnectX"
    columns = 7
    rows = 6

    in_state = [0] * 42
    out_state = np.zeros((3, columns, rows))

    tester.run(processor, env_name)
    tester.change_observation_info(
        processor,
        env_name,
        EnvObservationType.SHAPE3,
        BoxSpace((3, columns, rows), 0, 1),
    )
    tester.observation_decode(
        processor,
        env_name,
        in_observation=in_state,
        out_observation=out_state,
    )
