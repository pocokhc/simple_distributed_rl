import numpy as np
import pytest

import srl
from srl.test.env import env_test, player_test


def test_play():
    env = env_test("connectx")

    for x in [0, 2, 3, 4, 5, 6]:
        env.reset()
        board = [0] * 42
        assert not env.done
        assert env.state == board

        env.step(x)
        board[x + (5 * 7)] = 1
        assert not env.done
        assert env.rewards[0] == 0
        assert env.rewards[1] == 0
        assert env.state == board

        env.step(1)
        board[1 + (5 * 7)] = 2
        assert not env.done
        assert env.rewards[0] == 0
        assert env.rewards[1] == 0
        assert env.state == board

        env.step(x)
        board[x + (4 * 7)] = 1
        assert not env.done
        assert env.rewards[0] == 0
        assert env.rewards[1] == 0
        assert env.state == board

        env.step(1)
        board[1 + (4 * 7)] = 2
        assert not env.done
        assert env.rewards[0] == 0
        assert env.rewards[1] == 0
        assert env.state == board

        env.step(x)
        board[x + (3 * 7)] = 1
        assert not env.done
        assert env.rewards[0] == 0
        assert env.rewards[1] == 0
        assert env.state == board

        env.step(1)
        board[1 + (3 * 7)] = 2
        assert not env.done
        assert env.rewards[0] == 0
        assert env.rewards[1] == 0
        assert env.state == board

        env.step(x)
        board[x + (2 * 7)] = 1
        assert env.done
        assert env.rewards[0] == 1
        assert env.rewards[1] == -1
        assert env.state == board


@pytest.mark.parametrize(
    "player",
    [
        "alphabeta2",
        "alphabeta3",
        "alphabeta4",
    ],
)
def test_player(player):
    player_test("connectx", player)


def test_processor():
    env = srl.make_env(srl.EnvConfig("connectx", {"obs_type": "layer"}))
    env.setup()
    env.reset()

    out_state = np.zeros((7, 6, 2))

    assert (out_state == env.state).all()


def test_kaggle_connectx():
    pytest.importorskip("kaggle_environments")
    import kaggle_environments

    from srl.algorithms import ql

    rl_config = ql.Config()

    env = srl.make_env("connectx")
    rl_config.setup(env)
    parameter = rl_config.make_parameter()
    worker = srl.make_worker(rl_config, env, parameter)
    env.setup()
    worker.setup()

    def agent(observation, configuration):
        is_start_episode, is_end_episode = env.direct_step(observation, configuration)
        if is_start_episode:
            worker.reset(env.next_player)
        action = worker.policy()
        print(action)
        return env.decode_action(action)

    kaggle_env = kaggle_environments.make("connectx", debug=True)
    steps = kaggle_env.run([agent, "random"])
    r1 = steps[-1][0]["reward"]
    r2 = steps[-1][1]["reward"]
    assert r1 is not None
    assert r2 is not None


def test_kaggle_connectx_success():
    pytest.importorskip("kaggle_environments")
    import kaggle_environments

    from srl.algorithms import mcts

    rl_config = mcts.Config()

    env = srl.make_env("connectx")
    rl_config.setup(env)
    parameter = rl_config.make_parameter()
    worker = srl.make_worker(rl_config, env, parameter)
    env.setup()
    worker.setup()

    def agent(observation, configuration):
        is_start_episode, is_end_episode = env.direct_step(observation, configuration)
        if is_start_episode:
            worker.reset(env.next_player)
        action = worker.policy()
        return env.decode_action(action)

    kaggle_env = kaggle_environments.make("connectx", debug=True)
    steps = kaggle_env.run([agent, "random"])
    r1 = steps[-1][0]["reward"]
    r2 = steps[-1][1]["reward"]
    assert r1 is not None
    assert r2 is not None
