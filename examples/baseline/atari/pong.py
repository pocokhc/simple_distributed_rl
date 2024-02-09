import os

import numpy as np

import srl
from srl.algorithms import rainbow
from srl.utils import common

common.set_logger()

base_dir = os.path.dirname(__file__)
history_path = os.path.join(base_dir, "_pong/history")
TRAIN_COUNT = 500_000
ENV_PRE = "pong"


def _train(name, rl_config):
    env_config = srl.EnvConfig(
        "ALE/Pong-v5",
        kwargs=dict(frameskip=9, repeat_action_probability=0, full_action_space=False),
    )
    runner = srl.Runner(env_config, rl_config)
    runner.set_history_on_file(os.path.join(base_dir, f"_{ENV_PRE}_{name}"))
    runner.train(max_train_count=TRAIN_COUNT)

    rewards = runner.evaluate(max_episodes=10)
    print(f"reward 10 episode mean: {np.mean(rewards)}")


def train_rainbow():
    rl_config = rainbow.Config(
        batch_size=32,
        target_model_update_interval=1_000,
        discount=0.99,
        enable_reward_clip=False,
        enable_double_dqn=True,
        enable_rescale=False,
    )
    rl_config.memory.warmup_size = 1000
    rl_config.memory.capacity = 100_000
    rl_config.window_length = 4

    _train("Rainbow", rl_config)


def compare():
    names = [
        "Rainbow",
    ]
    histories = srl.Runner.load_histories([os.path.join(base_dir, f"_{ENV_PRE}_{n}") for n in names])
    histories.plot("train", "reward0", title=f"Train:{TRAIN_COUNT}")
    histories.plot("time", "reward0", title=f"Train:{TRAIN_COUNT}")


if __name__ == "__main__":
    train_rainbow()
