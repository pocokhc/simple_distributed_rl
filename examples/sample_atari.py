import numpy as np

import srl
from srl.algorithms import dqn  # algorithm load
from srl.base.define import EnvObservationTypes
from srl.rl.processors.image_processor import ImageProcessor
from srl.utils import common

common.set_logger()

_parameter_path = "_sample_atari_parameter.dat"

WARMUP_COUNT = 5_000
TRAIN_COUNT = 500_000


def _create_runner(load_parameter: bool):
    # --- Atari env
    # Run "pip install gymnasium pygame" and also see the URL below.
    # https://gymnasium.farama.org/environments/atari/
    env_config = srl.EnvConfig(
        "ALE/Pong-v5",
        kwargs=dict(frameskip=1, repeat_action_probability=0, full_action_space=False),
        frameskip=7,
    )
    rl_config = dqn.Config(
        batch_size=32,
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
    rl_config.memory.capacity = 10_000
    rl_config.memory.set_proportional_memory(beta_steps=TRAIN_COUNT)
    rl_config.image_block.set_r2d3_image()
    rl_config.hidden_block.set_mlp((512,))
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
    runner = srl.Runner(env_config, rl_config)

    # --- load parameter
    if load_parameter:
        runner.load_parameter(_parameter_path)

    return runner


def train():
    runner = _create_runner(load_parameter=False)

    # (option) print tensorflow model
    runner.model_summary(expand_nested=True)

    # --- train
    runner.train_mp(actor_num=2, max_train_count=TRAIN_COUNT)
    runner.save_parameter(_parameter_path)


def evaluate():
    runner = _create_runner(load_parameter=True)

    # --- evaluate
    rewards = runner.evaluate(max_episodes=10)
    print(f"reward 10 episode mean: {np.mean(rewards)}")

    # --- animation
    runner.animation_save_gif("_atari.gif")


if __name__ == "__main__":
    train()
    evaluate()
