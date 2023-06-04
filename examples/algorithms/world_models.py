import os
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

import srl
from srl import runner

# --- env & algorithm load
from srl.envs import grid  # noqa F401
from srl.utils import common

from srl.algorithms import world_models  # isort: skip


common.logger_print()

param_path = os.path.join(os.path.dirname(__file__), "_world_models_param.dat")
memory_path = os.path.join(os.path.dirname(__file__), "_world_models_memory.dat")


# WorldModelsのパラメータ
def _create_config():
    env_config = srl.EnvConfig("Grid")
    rl_config = world_models.Config(
        z_size=1,
        sequence_length=10,
        rnn_units=8,
        num_mixture=3,
        batch_size=64,
        temperature=0.1,
    )
    rl_config.use_render_image_for_observation = True
    rl_config.parameter_path = param_path
    rl_config.remote_memory_path = memory_path
    return runner.Config(env_config, rl_config), rl_config


# サンプルの収集
def s1_collect_sample():
    config, rl_config = _create_config()
    rl_config.train_mode = 1
    _, memory, _ = runner.train(config, max_episodes=100, disable_trainer=True)
    memory.save(memory_path)


# VAEの学習
def s2_train_vae():
    config, rl_config = _create_config()
    rl_config.train_mode = 1
    rl_config.lr = 0.001
    rl_config.kl_tolerance = 4.0
    parameter, memory, history = runner.train_only(config, max_train_count=20_000)
    parameter.save(param_path)


# MDN-RNNの学習
def s3_train_rnn():
    config, rl_config = _create_config()
    rl_config.train_mode = 2
    rl_config.lr = 0.001
    rl_config.memory_warmup_size = 100
    parameter, memory, history = runner.train_only(config, max_train_count=50_000)
    parameter.save(param_path)


# Controllerの学習
def s4_train_controller():
    config, rl_config = _create_config()
    rl_config.train_mode = 3
    rl_config.num_simulations = 20
    rl_config.num_individual = 4
    max_episodes = rl_config.num_simulations * rl_config.num_individual * 300
    parameter, memory, history = runner.train(config, max_episodes=max_episodes)
    parameter.save(param_path)


def evaluate():
    config, rl_config = _create_config()
    rewards = runner.evaluate(config, max_episodes=100)
    print(f"Average reward for 100 episodes: {np.mean(rewards)}")


def plot_vae():
    config, rl_config = _create_config()
    parameter = cast(world_models.Parameter, config.make_parameter())

    imgs, _ = parameter.vae.sample(25)
    fig, ax = plt.subplots(nrows=5, ncols=5)
    for i in range(25):
        idx = divmod(i, 5)
        ax[idx].imshow(imgs[i])
        ax[idx].axis("off")
    plt.show()


def plot_episode():
    config, rl_config = _create_config()
    render = runner.animation(config)
    render.create_anime(draw_info=True).save(os.path.join(os.path.dirname(__file__), "_world_models.gif"))


if __name__ == "__main__":
    # 各学習
    s1_collect_sample()
    s2_train_vae()
    s3_train_rnn()
    s4_train_controller()

    # パラメータの表示
    config, rl_config = _create_config()
    config.model_summary()

    evaluate()  # 評価
    plot_vae()  # VAEの可視化
    plot_episode()  # 1エピソードの可視化
