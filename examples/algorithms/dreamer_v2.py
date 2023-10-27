import os

import numpy as np

import srl
from srl.algorithms import dreamer_v2
from srl.utils import common

common.logger_print()

_parameter_path = os.path.join(os.path.dirname(__file__), "_dreamer_v2_param.dat")


def create_runner():
    env_config = srl.EnvConfig("EasyGrid")
    env_config.max_episode_steps = 20

    rl_config = dreamer_v2.Config(
        deter_size=100,
        stoch_size=32,
        reward_layer_sizes=(30, 30),
        discount_layer_sizes=(50,),
        critic_layer_sizes=(50, 50),
        actor_layer_sizes=(50, 50),
        discount=0.9,
        batch_size=32,
        batch_length=21,
        free_nats=0.1,
        kl_scale=0.1,
        lr_model=0.002,
        lr_critic=0.0005,
        lr_actor=0.0005,
        epsilon=1.0,
        critic_estimation_method="dreamer_v2",
        experience_acquisition_method="episode",
        horizon=15,
        reinforce_rate=0.1,
        entropy_rate=0.001,
        reinforce_baseline="v",
    )
    rl_config.memory.warmup_size = 1000

    rl_config.use_render_image_for_observation = True
    runner = srl.Runner(env_config, rl_config)

    return runner, rl_config


def train_dynamics():
    runner, rl_config = create_runner()

    rl_config.enable_train_model = True
    rl_config.enable_train_critic = False
    rl_config.enable_train_actor = False
    runner.train(max_train_count=3_000)
    runner.save_parameter(_parameter_path)


def train_actor_critic():
    runner, rl_config = create_runner()

    runner.load_parameter(_parameter_path)

    rl_config.enable_train_model = False
    rl_config.enable_train_critic = True
    rl_config.enable_train_actor = True
    runner.train(max_train_count=2_000)
    runner.save_parameter(_parameter_path)


def eval():
    runner, rl_config = create_runner()

    runner.load_parameter(_parameter_path)

    # --- evaluate
    rewards = runner.evaluate(max_episodes=10)
    print(f"Average reward for 10 episodes: {np.mean(rewards)}")

    # --- animation
    path = os.path.join(os.path.dirname(__file__), "_dreamer_v2.gif")
    runner.animation_save_gif(path)


if __name__ == "__main__":
    train_dynamics()
    train_actor_critic()
    eval()
