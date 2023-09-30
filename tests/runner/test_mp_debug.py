import numpy as np

from srl.algorithms import ql, ql_agent57
from srl.runner.runner import Runner
from srl.utils import common

common.logger_print()


def test_train():
    runner = Runner("Grid", ql.Config())
    runner.train_mp_debug(actor_num=5, max_train_count=10_000)

    # eval
    rewards = runner.evaluate(max_episodes=100)
    rewards = np.mean(rewards)
    print(rewards)
    assert rewards > 0.5


def test_train2():
    rl_config = ql_agent57.Config()
    rl_config.memory.capacity = 1000
    rl_config.memory.set_replay_memory()
    runner = Runner("Grid", rl_config)
    runner.train(max_episodes=5000)

    memory_len = runner.remote_memory.length()
    rl_config.memory.capacity = 1100

    runner.train_mp_debug(max_episodes=10, return_remote_memory=True)
    memory2 = runner.remote_memory

    # eval
    rewards = runner.evaluate(max_episodes=100)
    rewards = np.mean(rewards)
    print(rewards)
    assert rewards > 0.5

    print(memory2.length(), memory_len)
    assert memory2.length() > memory_len
