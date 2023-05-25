import numpy as np
import pytest

import srl
from srl.base.define import EnvObservationTypes
from srl.base.spaces.box import BoxSpace
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
    out_state = np.zeros((2, columns, rows))

    tester.run(processor, env_name)
    tester.change_observation_info(
        processor,
        env_name,
        EnvObservationTypes.SHAPE3,
        BoxSpace((2, columns, rows), 0, 1),
    )
    tester.observation_decode(
        processor,
        env_name,
        in_observation=in_state,
        out_observation=out_state,
    )


def test_kaggle_connectx():
    pytest.importorskip("kaggle_environments")
    import kaggle_environments

    from srl.algorithms import ql

    rl_config = ql.Config()

    env = srl.make_env("connectx")
    parameter = srl.make_parameter(rl_config, env)
    worker = srl.make_worker(rl_config, parameter)

    def agent(observation, configuration):
        env.direct_step(observation, configuration)
        if env.is_start_episode:
            worker.on_reset(env, env.next_player_index)
        action = worker.policy(env)
        return env.decode_action(action)

    kaggle_env = kaggle_environments.make("connectx", debug=True)
    steps = kaggle_env.run([agent, "random"])
    r1 = steps[-1][0]["reward"]
    r2 = steps[-1][1]["reward"]
    assert r1 is not None
    assert r2 is not None


def test_kaggle_connectx_fail():
    pytest.importorskip("kaggle_environments")
    import kaggle_environments

    from srl.algorithms import mcts

    rl_config = mcts.Config()

    env = srl.make_env("connectx")
    parameter = srl.make_parameter(rl_config, env)
    worker = srl.make_worker(rl_config, parameter, training=True)

    def agent(observation, configuration):
        env.direct_step(observation, configuration)
        if env.is_start_episode:
            worker.on_reset(env, env.next_player_index)
        action = worker.policy(env)
        return env.decode_action(action)

    kaggle_env = kaggle_environments.make("connectx", debug=True)
    steps = kaggle_env.run([agent, "random"])
    r1 = steps[-1][0]["reward"]
    r2 = steps[-1][1]["reward"]
    assert (r1 is None) or (r2 is None)
