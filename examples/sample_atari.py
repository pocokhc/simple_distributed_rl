import numpy as np

import srl
from srl import runner
from srl.base.define import EnvObservationType
from srl.base.rl.processors.image_processor import ImageProcessor
from srl.utils import common

# --- env & algorithm load
from srl.algorithms import dqn  # isort: skip
import gym  # isort: skip # noqa F401

common.set_logger()

_parameter_path = "_sample_atari_parameter.dat"


def _create_config():
    env_config = srl.EnvConfig(
        "ALE/Pong-v5",
        kwargs=dict(frameskip=1, repeat_action_probability=0, full_action_space=False),
        frameskip=7,
    )
    rl_config = dqn.Config()
    rl_config.set_atari_config()
    rl_config.capacity = 10_000
    rl_config.memory_warmup_size = 5_000
    rl_config.exploration_steps = 1000_000
    rl_config.processors = [
        ImageProcessor(
            image_type=EnvObservationType.GRAY_2ch,
            trimming=(30, 0, 210, 160),
            resize=(84, 84),
            enable_norm=True,
        )
    ]
    rl_config.use_rl_processor = False
    rl_config.window_length = 4
    return runner.Config(env_config, rl_config)


def train():
    config = _create_config()

    # (option) print tensorflow model
    config.model_summary(expand_nested=True)

    # --- train
    parameter, remote_memory, history = runner.train_mp(
        config,
        max_train_count=500_000,
        enable_evaluation=False,
    )
    parameter.save(_parameter_path)


def evaluate():
    config = _create_config()

    # --- setting load parameter
    config.rl_config.parameter_path = _parameter_path

    # --- evaluate
    rewards = runner.evaluate(config, max_episodes=10)
    print(f"reward 10 episode mean: {np.mean(rewards)}")

    # --- animation
    render = runner.animation(config)
    render.create_anime(draw_info=True).save("_atari.gif")


if __name__ == "__main__":
    train()
    evaluate()
