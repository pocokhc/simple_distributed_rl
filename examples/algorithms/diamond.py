import os
from pathlib import Path

import srl
from srl.algorithms import diamond
from srl.utils import common


def create_runner():
    env_config = srl.EnvConfig("EasyGrid")
    rl_config = diamond.Config(observation_mode="render_image").set_small_params()

    runner = srl.Runner(env_config, rl_config)
    runner.set_progress(interval_limit=60)
    return runner


def train_diff():
    runner = create_runner()
    runner.rl_config.train_reward_end = False
    runner.rl_config.train_actor_critic = False

    runner.rollout(max_memory=10000)
    runner.train_only(max_train_count=15000)
    runner.save_parameter(str(Path(__file__).parent / "_diamond.dat"))


def train_reward_end():
    runner = create_runner()
    runner.load_parameter(str(Path(__file__).parent / "_diamond.dat"))
    runner.rl_config.train_diffusion = False
    runner.rl_config.train_actor_critic = False
    runner.rl_config.batch_size = 8

    runner.rollout(max_memory=10000)
    runner.train_only(max_train_count=1000)
    runner.save_parameter(str(Path(__file__).parent / "_diamond.dat"))


def train_actor_critic():
    runner = create_runner()
    runner.rl_config.train_diffusion = False
    runner.rl_config.batch_size = 8
    runner.load_parameter(str(Path(__file__).parent / "_diamond.dat"))

    runner.rollout(max_memory=10000)
    runner.train_only(max_train_count=5000)
    runner.save_parameter(str(Path(__file__).parent / "_diamond.dat"))


def eval():
    runner = create_runner()
    runner.load_parameter(str(Path(__file__).parent / "_diamond.dat"))

    print(runner.evaluate())

    path = os.path.join(str(Path(__file__).parent / "_diamond.gif"))
    runner.animation_save_gif(path)
    runner.replay_window()


if __name__ == "__main__":
    common.logger_print()

    train_diff()
    train_reward_end()
    train_actor_critic()
    eval()
