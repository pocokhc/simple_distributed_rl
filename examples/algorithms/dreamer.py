import os

import numpy as np

import srl
from srl import runner
from srl.utils import common

# --- load algorithm
from srl.algorithms import dreamer  # isort: skip


common.logger_print()

_parameter_path = os.path.join(os.path.dirname(__file__), "_dreamer_param.dat")


def create_config():
    env_config = srl.EnvConfig("EasyGrid")
    env_config.max_episode_steps = 20
    env_config.check_action = False
    env_config.check_val = False

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
        model_lr=0.001,
        value_lr=0.0005,
        actor_lr=0.0001,
        memory_warmup_size=1000,
        epsilon=1.0,
        value_estimation_method="dreamer",  # "simple" or "dreamer"
        horizon=20,
    )

    config = runner.Config(env_config, rl_config)
    rl_config.use_render_image_for_observation = True

    return config, rl_config


def train_dynamics():
    config, rl_config = create_config()

    rl_config.enable_train_model = True
    rl_config.enable_train_actor = False
    rl_config.enable_train_value = False
    parameter, _, _ = runner.train(config, max_train_count=10_000)
    parameter.save(_parameter_path)


def train_value():
    config, rl_config = create_config()

    parameter = config.make_parameter()
    parameter.load(_parameter_path)

    rl_config.enable_train_model = False
    rl_config.enable_train_actor = False
    rl_config.enable_train_value = True
    parameter, _, _ = runner.train(config, parameter=parameter, max_train_count=1_000)
    parameter.save(_parameter_path)


def train_actor():
    config, rl_config = create_config()

    parameter = config.make_parameter()
    parameter.load(_parameter_path)

    rl_config.enable_train_model = False
    rl_config.enable_train_actor = True
    rl_config.enable_train_value = True
    parameter, _, _ = runner.train(config, parameter=parameter, max_train_count=2_000)
    parameter.save(_parameter_path)


def eval():
    config, rl_config = create_config()

    parameter = config.make_parameter()
    parameter.load(_parameter_path)

    # --- evaluate
    rewards = runner.evaluate(config, parameter, max_episodes=10)
    print(f"Average reward for 10 episodes: {np.mean(rewards)}")

    # --- animation
    render = runner.animation(config, parameter)
    render.create_anime().save(os.path.join(os.path.dirname(__file__), "_dreamer.gif"))


if __name__ == "__main__":
    train_dynamics()
    train_value()
    train_actor()
    eval()
