import os

import mlflow
import numpy as np

import srl
from srl.rl.processors.atari_processor import AtariPongProcessor
from srl.utils import common

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
common.logger_print()


def _train(rl_config, train):
    env_config = srl.EnvConfig(
        "ALE/Pong-v5",
        kwargs=dict(frameskip=7, repeat_action_probability=0, full_action_space=False),
        processors=[AtariPongProcessor()],
    )

    runner = srl.Runner(env_config, rl_config)
    runner.model_summary()

    runner.set_mlflow()
    runner.train(max_train_count=train)

    rewards = runner.evaluate()
    print(f"[{rl_config.name}] {np.mean(rewards)}, {rewards}")


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
        memory_warmup_size=1000,
        memory_capacity=10_000,
        memory_compress=False,
        window_length=4,
    )
    rl_config.input_image_block.set_dqn_block()
    rl_config.hidden_block.set((512,))
    # Total params: 4,046,502

    _train(rl_config, train=100_000)


if __name__ == "__main__":
    train_dqn()
