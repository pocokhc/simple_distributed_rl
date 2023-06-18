import io

import numpy as np

from srl import runner
from srl.algorithms import ql_agent57


def test_train():
    config = runner.Config("OX", ql_agent57.Config())
    parameter, memory = runner.train_simple(config, max_train_count=10)

    assert memory.length() > 1


def test_train_only():
    rl_config = ql_agent57.Config()
    config = runner.Config("Grid", rl_config)

    _, memory = runner.train_simple(
        config,
        max_steps=10_000,
        disable_trainer=True,
    )
    assert memory.length() > 1000
    rl_config.memory_warmup_size = 1000
    parameter, _ = runner.train_only_simple(
        config,
        remote_memory=memory,
        max_train_count=50_000,
    )
    rewards = runner.evaluate(config, parameter, max_episodes=100)
    reward = np.mean(rewards)
    assert reward > 0.5, f"reward: {reward}"
