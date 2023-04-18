import numpy as np
import pytest

import srl
from srl import runner
from srl.algorithms import ql, ql_agent57
from srl.envs import grid, ox  # noqa F401
from srl.utils.common import is_packages_installed


@pytest.mark.skipif(not is_packages_installed(["cv2", "matplotlib", "PIL", "pygame"]), reason="no module")
def test_basic():
    env_config = srl.EnvConfig("Grid")
    rl_config = ql.Config()
    config = runner.Config(env_config, rl_config)

    # train
    parameter, _, _ = runner.train(config, max_steps=10000)

    # eval
    rewards = runner.evaluate(config, parameter, max_episodes=100, print_progress=True)
    rewards = np.mean(rewards)
    print(rewards)
    assert rewards > 0.5

    # render terminal
    rewards = runner.render(config, parameter)
    rewards = np.mean(rewards)
    print(rewards)
    assert rewards > 0.5

    # render window
    rewards = runner.render(config, parameter, render_mode="window")
    rewards = np.mean(rewards)
    print(rewards)
    assert rewards > 0.5


def test_shuffle_player():
    env_config = srl.EnvConfig("OX")
    config = runner.Config(env_config, None)
    config.players = ["cpu", "random"]

    # shuffle した状態でも報酬は元の順序を継続する
    rewards = runner.evaluate(config, parameter=None, max_episodes=100, shuffle_player=True)
    rewards = np.mean(rewards, axis=0)
    assert rewards[0] > 0.7  # CPUがまず勝つ


def test_modelbase_train_flow():
    env_config = srl.EnvConfig("Grid")
    rl_config = ql_agent57.Config()
    config = runner.Config(env_config, rl_config)

    _, memory, _ = runner.train(config, max_steps=10_000, disable_trainer=True)

    rl_config.memory_warmup_size = 1000
    parameter, _, _ = runner.train_only(config, remote_memory=memory, max_train_count=100_000)

    rewards = runner.evaluate(config, parameter, max_episodes=100)
    reward = np.mean(rewards)
    assert reward > 0.5, f"reward: {reward}"
