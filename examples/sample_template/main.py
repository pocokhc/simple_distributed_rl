import os

import numpy as np

import srl
from srl.utils import common

common.logger_print()


cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
parameter_path = os.path.join(os.path.dirname(__file__), "_params.dat")


# --- train sample
def train():
    # yamlの設定を読み込んでrunnerを作成
    runner = srl.load(cfg_path)

    # loadした設定情報
    runner.summary(show_changed_only=True)

    # 学習して保存
    runner.play()
    runner.save_parameter(parameter_path)

    # 簡易評価
    print(runner.evaluate())


# --- evaluate sample
def evaluate():
    runner = srl.load(cfg_path)

    # loadした設定情報
    runner.summary(show_changed_only=True)
    runner.model_summary()

    runner.load_parameter(parameter_path)
    rewards = runner.evaluate(max_episodes=100)
    print(rewards)
    print(f"Average reward for 100 episodes: {np.mean(rewards, axis=0)}")


# --- render terminal sample
def render_terminal():
    runner = srl.load(cfg_path)
    runner.load_parameter(parameter_path)
    runner.render_terminal()


# --- render window sample
#  (Run "pip install pillow pygame" to use the render_window)
def render_window():
    runner = srl.load(cfg_path)
    runner.load_parameter(parameter_path)
    runner.render_window()


# --- animation sample
#  (Run "pip install opencv-python pillow pygame" to use the animation)
def animation():
    runner = srl.load(cfg_path)
    runner.load_parameter(parameter_path)
    runner.animation_save_gif("_FrozenLake.gif")


# --- replay window sample
#  (Run "pip install opencv-python pillow pygame" to use the replay_window)
def replay_window():
    runner = srl.load(cfg_path)
    runner.load_parameter(parameter_path)
    runner.replay_window()


if __name__ == "__main__":
    train()
    evaluate()
    # render_terminal()
    # render_window()
    # animation()
    # replay_window()
