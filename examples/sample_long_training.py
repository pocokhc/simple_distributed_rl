import numpy as np

import srl
from srl.algorithms import ql  # algorithm load


def _create_runner():
    # (Run "pip install gymnasium pygame" to use the gymnasium environment)
    env_config = srl.EnvConfig("FrozenLake-v1")
    rl_config = ql.Config()
    runner = srl.Runner(env_config, rl_config)

    # --- Set directory to save learning
    runner.set_checkpoint("_sample_long_training_checkpoint", is_load=True)

    return runner


def train(timeout=10):
    runner = _create_runner()
    runner.train(timeout=timeout)


def evaluate():
    runner = _create_runner()
    rewards = runner.evaluate(max_episodes=100)
    print(f"Average reward for 100 episodes: {np.mean(rewards, axis=0)}")


if __name__ == "__main__":
    train()
    evaluate()
