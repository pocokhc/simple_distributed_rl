import os

import mlflow
import numpy as np

import srl
from srl.envs.processors.atari_processor import AtariBreakoutProcessor
from srl.utils import common

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
common.logger_print()


def _train(rl_config, train_count=1_000_000):
    env_config = srl.EnvConfig(
        "ALE/Breakout-v5",
        kwargs=dict(frameskip=8, repeat_action_probability=0, full_action_space=False),
    )
    env_config.processors = [AtariBreakoutProcessor()]

    runner = srl.Runner(env_config, rl_config)
    runner.summary()
    runner.set_mlflow()

    runner.train(max_train_count=train_count)

    rewards = runner.evaluate()
    print(f"[{rl_config.name}] {train_count=}")
    print(f"{np.mean(rewards)}, {rewards}")


def train_dqn():
    from srl.algorithms import dqn

    rl_config = dqn.Config(
        batch_size=32,
        target_model_update_interval=2_000,
        discount=0.99,
        lr=0.0005,
        epsilon=0.1,
        enable_reward_clip=False,
        enable_double_dqn=True,
        enable_rescale=False,
        window_length=4,
    )
    rl_config.memory.warmup_size = 1000
    rl_config.memory.capacity = 10_000
    rl_config.memory.compress = False
    rl_config.input_block.image.set_dqn_block()
    rl_config.hidden_block.set((256,))

    _train(rl_config)


if __name__ == "__main__":
    train_dqn()
