import multiprocessing as mp
import time

import numpy as np

from srl import runner
from srl.algorithms import ql, ql_agent57
from srl.envs import grid, ox  # noqa F401
from srl.rl import memories
from srl.utils import common

common.logger_print()


def run_server():
    config = runner.Config("Grid", ql.Config(), base_dir="tmp_test")
    parameter, _, _ = runner.train_remote(
        config,
        max_train_count=10_000,
        eval=runner.EvalOption(),
        progress=runner.ProgressOption(),
        history=runner.HistoryOption(write_memory=True, write_file=True),
        checkpoint=runner.CheckpointOption(
            checkpoint_interval=1,
            eval=runner.EvalOption(),
        ),
    )

    # eval
    rewards = runner.evaluate(config, parameter, max_episodes=100)
    rewards = np.mean(rewards)
    print(rewards)
    assert rewards > 0.5


def run_client():
    from srl.runner.core_remote import run_actor

    run_actor("127.0.0.1", 50000)


def test_train():
    server = mp.Process(target=run_server)
    client = mp.Process(target=run_client)

    server.start()
    time.sleep(2)
    client.start()

    while True:
        time.sleep(1)  # polling time

        if not server.is_alive():
            break
        if not client.is_alive():
            break

    server.join(timeout=10)

    assert server.exitcode == 0
    assert client.exitcode == 0


def run_server2():
    config = runner.Config("Grid", ql_agent57.Config(memory=memories.ReplayMemoryConfig(capacity=1000)))
    parameter, memory, _ = runner.train(config, max_episodes=5000)

    memory_len = memory.length()
    config.rl_config.memory = memories.ReplayMemoryConfig(capacity=1100)

    parameter2, memory2, _ = runner.train_remote(
        config,
        max_train_count=10,
        progress=runner.ProgressOption(),
        parameter=parameter,
        remote_memory=memory,
        return_remote_memory=True,
    )

    # eval
    rewards = runner.evaluate(config, parameter2, max_episodes=100)
    rewards = np.mean(rewards)
    print(rewards)
    assert rewards > 0.5

    print(memory2.length(), memory_len)
    assert memory2.length() > memory_len


def test_train2():
    server = mp.Process(target=run_server2)
    client = mp.Process(target=run_client)

    server.start()
    time.sleep(60)
    client.start()
    time.sleep(30)

    while True:
        time.sleep(1)  # polling time

        if not server.is_alive():
            break
        if not client.is_alive():
            break

    server.join(timeout=10)

    assert server.exitcode == 0
    assert client.exitcode == 0
