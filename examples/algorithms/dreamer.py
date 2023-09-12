import os

import numpy as np

import srl
from srl.algorithms import dreamer
from srl.utils import common

common.logger_print()

_parameter_path = os.path.join(os.path.dirname(__file__), "_dreamer_param.dat")


def create_runner():
    env_config = srl.EnvConfig("EasyGrid")
    env_config.max_episode_steps = 20

    rl_config = dreamer.Config(
        deter_size=30,
        stoch_size=20,
        reward_num_units=30,
        reward_layers=2,
        value_num_units=30,
        value_layers=3,
        action_num_units=30,
        action_layers=3,
        cnn_depth=32,
        batch_size=16,
        batch_length=21,
        free_nats=0.1,
        kl_scale=1.0,
        lr_model=0.001,
        lr_value=0.0005,
        lr_actor=0.0001,
        memory_warmup_size=1000,
        epsilon=1.0,
        value_estimation_method="dreamer",  # "simple" or "dreamer"
        horizon=20,
    )

    rl_config.use_render_image_for_observation = True
    runner = srl.Runner(env_config, rl_config)

    return runner, rl_config


def train_dynamics():
    runner, rl_config = create_runner()

    rl_config.enable_train_model = True
    rl_config.enable_train_actor = False
    rl_config.enable_train_value = False
    runner.train(max_train_count=10_000)
    runner.save_parameter(_parameter_path)


def train_value():
    runner, rl_config = create_runner()

    runner.load_parameter(_parameter_path)

    rl_config.enable_train_model = False
    rl_config.enable_train_actor = False
    rl_config.enable_train_value = True
    runner.train(max_train_count=1_000)
    runner.save_parameter(_parameter_path)


def train_actor():
    runner, rl_config = create_runner()

    runner.load_parameter(_parameter_path)

    rl_config.enable_train_model = False
    rl_config.enable_train_actor = True
    rl_config.enable_train_value = True
    runner.train(max_train_count=2_000)
    runner.save_parameter(_parameter_path)


def eval():
    runner, rl_config = create_runner()

    runner.load_parameter(_parameter_path)

    # --- evaluate
    rewards = runner.evaluate(max_episodes=10)
    print(f"Average reward for 10 episodes: {np.mean(rewards)}")

    # --- animation
    path = os.path.join(os.path.dirname(__file__), "_dreamer.gif")
    runner.animation_save_gif(path)


if __name__ == "__main__":
    train_dynamics()
    train_value()
    train_actor()
    eval()
