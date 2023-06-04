import numpy as np

import srl
from srl import runner
from srl.base.define import EnvObservationTypes
from srl.rl import memories
from srl.rl.models import dqn as dqn_model
from srl.rl.models import mlp
from srl.rl.processors.image_processor import ImageProcessor
from srl.utils import common

# --- env & algorithm load
from srl.algorithms import dqn  # isort: skip
import gym  # isort: skip # noqa F401

common.set_logger()

_parameter_path = "_sample_atari_parameter.dat"

WARMUP_COUNT = 5_000
TRAIN_COUNT = 500_000


def _create_config():
    env_config = srl.EnvConfig(
        "ALE/Pong-v5",
        kwargs=dict(frameskip=1, repeat_action_probability=0, full_action_space=False),
        frameskip=7,
    )
    rl_config = dqn.Config(
        batch_size=32,
        memory=memories.ProportionalMemoryConfig(capacity=10_000, beta_steps=TRAIN_COUNT),
        image_block_config=dqn_model.R2D3ImageBlockConfig(),
        hidden_block_config=mlp.MLPBlockConfig(layer_sizes=(512,)),
        target_model_update_interval=10_000,
        discount=0.99,
        lr=0.00025,
        initial_epsilon=1.0,
        final_epsilon=0.1,
        exploration_steps=TRAIN_COUNT,
        memory_warmup_size=WARMUP_COUNT,
        enable_reward_clip=False,
        enable_double_dqn=True,
        enable_rescale=False,
    )
    rl_config.processors = [
        ImageProcessor(
            image_type=EnvObservationTypes.GRAY_2ch,
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
    config.actor_num = 2
    parameter, remote_memory, history = runner.train_mp(
        config,
        max_train_count=TRAIN_COUNT,
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
