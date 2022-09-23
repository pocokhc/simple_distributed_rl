import numpy as np
import srl
from srl import runner
from srl.base.define import EnvObservationType
from srl.base.rl.processors.image_processor import ImageProcessor
from srl.utils import common

common.set_logger()

# --- env & algorithm load
from srl.algorithms import dqn  # isort: skip
import gym  # isort: skip # noqa F401

_parameter_path = "_sample_atari_parameter.dat"


def _create_config():
    env_config = srl.EnvConfig(
        "ALE/Pong-v5",
        kwargs=dict(frameskip=1, repeat_action_probability=0, full_action_space=False),
        frameskip=3,
        max_episode_steps=1000,
    )
    rl_config = dqn.Config(
        capacity=50_000,
        memory_warmup_size=2_000,
        discount=0.997,
        lr=0.0002,
        batch_size=32,
        target_model_update_interval=2000,
        initial_epsilon=1.0,
        final_epsilon=0.1,
        exploration_steps=500_000,
    )
    rl_config.processors = [
        ImageProcessor(
            image_type=EnvObservationType.GRAY_2ch,
            trimming=(30, 10, 210, 150),
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
    config.model_summary()

    # --- train
    parameter, remote_memory, history = runner.mp_train(
        config,
        runner.MpConfig(actor_num=1, allocate_trainer="/GPU:0"),
        max_train_count=500_000,
        enable_evaluation=False,
    )
    parameter.save(_parameter_path)


def evaluate():
    config = _create_config()

    # --- setting load parameter (Loads the file if it exists)
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
