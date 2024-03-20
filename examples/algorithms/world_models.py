import os
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

import srl
from srl.algorithms import world_models
from srl.base.define import ObservationModes
from srl.utils import common

common.logger_print()

param_path = os.path.join(os.path.dirname(__file__), "_world_models_param.dat")
memory_path = os.path.join(os.path.dirname(__file__), "_world_models_memory.dat")


# WorldModelsのパラメータ
def _create_runner():
    env_config = srl.EnvConfig("Grid")
    rl_config = world_models.Config(
        z_size=1,
        sequence_length=10,
        rnn_units=8,
        num_mixture=3,
        batch_size=64,
        temperature=0.1,
    )
    rl_config.observation_mode = ObservationModes.RENDER_IMAGE

    runner = srl.Runner(env_config, rl_config)
    if os.path.isfile(param_path):
        runner.load_parameter(param_path)
    if os.path.isfile(memory_path):
        runner.load_memory(memory_path)

    return runner, rl_config


# サンプルの収集
def s1_collect_sample():
    runner, rl_config = _create_runner()
    rl_config.train_mode = 1
    runner.rollout(max_memory=10_000)
    runner.save_memory(memory_path)


# VAEの学習
def s2_train_vae():
    runner, rl_config = _create_runner()
    rl_config.train_mode = 1
    rl_config.lr = 0.001
    rl_config.kl_tolerance = 4.0
    runner.train_only(max_train_count=20_000)
    runner.save_parameter(param_path)


# MDN-RNNの学習
def s3_train_rnn():
    runner, rl_config = _create_runner()
    rl_config.train_mode = 2
    rl_config.lr = 0.001
    rl_config.memory_warmup_size = 100
    runner.train_only(max_train_count=50_000)
    runner.save_parameter(param_path)


# Controllerの学習
def s4_train_controller():
    runner, rl_config = _create_runner()
    rl_config.train_mode = 3
    rl_config.num_simulations = 20
    rl_config.num_individual = 4
    max_episodes = rl_config.num_simulations * rl_config.num_individual * 300
    runner.train(max_episodes=max_episodes)
    runner.save_parameter(param_path)


def evaluate():
    runner, rl_config = _create_runner()
    rewards = runner.evaluate(max_episodes=100)
    print(f"Average reward for 100 episodes: {np.mean(rewards)}")


def plot_vae():
    runner, rl_config = _create_runner()
    parameter = cast(world_models.Parameter, runner.make_parameter())

    imgs, _ = parameter.vae.sample(25)
    fig, ax = plt.subplots(nrows=5, ncols=5)
    for i in range(25):
        idx = divmod(i, 5)
        ax[idx].imshow(imgs[i])
        ax[idx].axis("off")
    plt.show()


def plot_episode():
    runner, rl_config = _create_runner()
    path = os.path.join(os.path.dirname(__file__), "_world_models.gif")
    runner.animation_save_gif(path)


if __name__ == "__main__":
    # 各学習
    s1_collect_sample()
    s2_train_vae()
    s3_train_rnn()
    s4_train_controller()

    evaluate()  # 評価
    plot_vae()  # VAEの可視化
    plot_episode()  # 1エピソードの可視化
