import numpy as np

import srl
from srl.algorithms import ql, ql_agent57


def test_train():
    c = srl.RunContext(srl.EnvConfig("Grid"), ql.Config())
    c.play_mode = "train"
    c.max_train_count = 100000
    runner = srl.Runner(context=c)

    runner.play()
    rewards = np.mean(runner.evaluate(100))
    assert rewards > 0.6


def test_rollout():
    c = srl.RunContext(srl.EnvConfig("Grid"), ql_agent57.Config())
    c.play_mode = "rollout"
    c.max_episodes = 1000
    runner = srl.Runner(context=c)

    runner.play()
    assert runner.memory.length() > 0


def test_train_only():
    c = srl.RunContext(srl.EnvConfig("Grid"), ql_agent57.Config())
    c.play_mode = "train_only"
    c.max_train_count = 100000
    runner = srl.Runner(context=c)

    runner.rollout(max_episodes=1000)
    runner.play()
    rewards = np.mean(runner.evaluate(100))
    assert rewards > 0.6


def test_train_evaluate():
    c = srl.RunContext(srl.EnvConfig("Grid"), ql_agent57.Config())
    c.play_mode = "evaluate"
    c.max_episodes = 100
    runner = srl.Runner(context=c)

    state = runner.play()
    print(state.episode_rewards_list)
