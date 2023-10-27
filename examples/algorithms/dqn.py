import os

import numpy as np
import srl
from srl.algorithms import dqn
from srl.utils import common

common.logger_print()


def main():
    env_config = srl.EnvConfig("Pendulum-v1")

    # atariのパラメータでは大げさすきるので Pendulum 用に縮小
    # あえて書いていますが、フレームワークのデフォルトは大体この値です。（詳細はコードを参照）
    rl_config = dqn.Config()
    rl_config.set_atari_config()
    rl_config.hidden_block.set_mlp((64, 64))  # 画像処理層がないので隠れ層を増やしてます
    rl_config.memory.capacity = 100_000  # 下げないとメモリ足りなくなる可能性あり
    rl_config.window_length = 1  # Pendulum は加速も状態に含まれているので不要
    rl_config.lr.set_constant(0.001)
    rl_config.epsilon.set_constant(0.1)
    rl_config.memory.warmup_size = 1000  # そこまで待たずに学習開始
    rl_config.target_model_update_interval = 1000  # 大きすぎると学習がゆっくり
    rl_config.enable_reward_clip = False  # 報酬clipしない

    # frameworkを指定したい場合 "tensorflow" or "torch"
    # rl_config.framework = "tensorflow"

    runner = srl.Runner(env_config, rl_config)
    runner.model_summary(expand_nested=True)

    # --- train
    runner.set_history()
    runner.train(max_episodes=200)
    history = runner.get_history()
    history.plot(ylabel_right=["trainer_loss"])

    # --- evaluate
    rewards = runner.evaluate(max_episodes=20)
    print(f"Average reward for 20 episodes: {np.mean(rewards)}")

    # --- animation
    path = os.path.join(os.path.dirname(__file__), "_dqn.gif")
    runner.animation_save_gif(path)


if __name__ == "__main__":
    main()
