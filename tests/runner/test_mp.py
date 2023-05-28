import numpy as np

from srl import runner
from srl.algorithms import ql, ql_agent57
from srl.envs import grid, ox
from srl.rl import memories  # noqa F401


def test_train():
    config = runner.Config("Grid", ql.Config(), base_dir="tmp_test", actor_num=2)
    parameter, _, _ = runner.train_mp(
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


def test_train2():
    config = runner.Config("Grid", ql_agent57.Config(memory=memories.ReplayMemoryConfig(capacity=1000)))
    parameter, memory, _ = runner.train(config, max_episodes=5000)

    memory_len = memory.length()
    config.rl_config.memory = memories.ReplayMemoryConfig(capacity=1100)

    parameter2, memory2, _ = runner.train_mp(
        config,
        max_train_count=10_000,
        eval=runner.EvalOption(),
        progress=runner.ProgressOption(),
        history=runner.HistoryOption(write_memory=True, write_file=True),
        checkpoint=runner.CheckpointOption(
            checkpoint_interval=1,
            eval=runner.EvalOption(),
        ),
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
