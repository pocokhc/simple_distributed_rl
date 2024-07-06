import os

import mlflow
import numpy as np

import srl
from srl.base.define import SpaceTypes
from srl.rl.processors.atari_processor import AtariPongProcessor
from srl.runner.callbacks.mlflow_callback import MLFlowCallback
from srl.utils import common

mlflow.set_tracking_uri("mlruns")
common.logger_print()

ENV_NAME = "ALE/Pong-v5"
BASE_TRAIN = 200_000


def _train(rl_config, train):
    env_config = srl.EnvConfig(
        ENV_NAME,
        kwargs=dict(frameskip=7, repeat_action_probability=0, full_action_space=False),
    )
    env_config.processors = [AtariPongProcessor()]

    runner = srl.Runner(env_config, rl_config)
    runner.model_summary()

    runner.set_progress_options(interval_limit=60 * 5)
    runner.train_mp(max_train_count=train, callbacks=[MLFlowCallback()])

    rewards = runner.evaluate()
    print(f"[{rl_config.name}] evaluate episodes: {np.mean(rewards)}")
    # runner.replay_window()


def train_dqn():
    from srl.algorithms import dqn

    rl_config = dqn.Config(
        batch_size=32,
        target_model_update_interval=1_000,
        discount=0.99,
        lr=0.001,
        epsilon=0.1,
        enable_reward_clip=False,
        enable_double_dqn=True,
        enable_rescale=False,
        memory_warmup_size=1000,
        memory_capacity=100_000,
        memory_compress=False,
        window_length=4,
    )
    rl_config.input_image_block.set_dqn_block()
    rl_config.hidden_block.set((512,))
    rl_config.window_length = 4
    # Total params: 4,046,502

    _train(rl_config, BASE_TRAIN)


def train_rainbow():
    from srl.algorithms import rainbow

    rl_config = rainbow.Config(
        multisteps=1,
        batch_size=32,
        target_model_update_interval=1_000,
        discount=0.99,
        lr=0.001,
        enable_reward_clip=False,
        enable_double_dqn=True,
        enable_rescale=False,
        memory_warmup_size=1000,
        memory_capacity=100_000,
        memory_compress=False,
        window_length=4,
    )
    # Total params: 8,012,455

    _train(rl_config, BASE_TRAIN)


def train_sac():
    from srl.algorithms import sac

    rl_config = sac.Config(
        batch_size=32,
        discount=0.99,
        memory_warmup_size=1000,
        memory_capacity=100_000,
        memory_compress=False,
        window_length=4,
    )
    rl_config.q_hidden_block.set((512,))
    rl_config.policy_hidden_block.set((512,))
    rl_config.override_action_type = SpaceTypes.DISCRETE

    _train(rl_config, BASE_TRAIN)


def compare():
    import matplotlib.pyplot as plt

    metric_name = "eval_reward0"

    plt.figure(figsize=(12, 6))
    plt.xlabel("train")
    plt.ylabel(metric_name)
    for name in [
        "DQN:tensorflow",
        "Rainbow:tensorflow",
        "SAC",
    ]:
        steps, values = MLFlowCallback.get_metrics(ENV_NAME, name, metric_name)
        if steps is None:
            continue
        if len(values) > 20:
            values = common.moving_average(values, 10)
        plt.plot(steps, values, label=name)
    plt.grid()
    plt.legend()
    plt.title(f"Train:{BASE_TRAIN}")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "_pong.png"))
    plt.show()


if __name__ == "__main__":
    train_dqn()
    train_rainbow()
    train_sac()
    compare()
