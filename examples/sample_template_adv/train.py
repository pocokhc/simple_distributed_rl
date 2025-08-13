import os

import hydra
import mlflow
import numpy as np
from omegaconf import OmegaConf

import srl
from srl.utils import common

# mlflowの場所の設定
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))

tmp_cfg_path = os.path.join(os.path.dirname(__file__), "_config.yaml")
common.logger_print()


@hydra.main(version_base=None, config_path=".", config_name="config")
def train(cfg):
    OmegaConf.resolve(cfg)

    # dict形式からrunnerを作成
    runner = srl.load(cfg)
    runner.save(tmp_cfg_path)  # hydraで作られた設定を保存

    # loadした設定情報
    runner.summary(show_changed_only=True)

    # 学習し、結果をmlflowに保存
    runner.set_mlflow()
    runner.play()

    # 簡易評価
    print(runner.evaluate())


# --- evaluate sample
def evaluate():
    runner = srl.load(tmp_cfg_path)
    runner.load_parameter_from_mlflow()  # mlflowの最後の結果をload
    rewards = runner.evaluate(max_episodes=100)
    print(f"Average reward for 100 episodes: {np.mean(rewards, axis=0)}")


# --- render terminal sample
def render_terminal():
    runner = srl.load(tmp_cfg_path)
    runner.load_parameter_from_mlflow()
    runner.render_terminal()


# --- render window sample
#  (Run "pip install pillow pygame" to use the render_window)
def render_window():
    runner = srl.load(tmp_cfg_path)
    runner.load_parameter_from_mlflow()
    runner.render_window()


# --- animation sample
#  (Run "pip install opencv-python pillow pygame" to use the animation)
def animation():
    runner = srl.load(tmp_cfg_path)
    runner.load_parameter_from_mlflow()
    runner.animation_save_gif(os.path.join(os.path.dirname(__file__), f"_{runner.env_config.name}.gif"))


# --- replay window sample
#  (Run "pip install opencv-python pillow pygame" to use the replay_window)
def replay_window():
    runner = srl.load(tmp_cfg_path)
    runner.load_parameter_from_mlflow()
    runner.replay_window()


if __name__ == "__main__":
    train()
    evaluate()
    # render_terminal()
    # render_window()
    # animation()
    # replay_window()
