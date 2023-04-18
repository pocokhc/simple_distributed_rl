import numpy as np

import srl
from srl import runner
from srl.rl.models.mlp_block_config import MLPBlockConfig

# --- env & algorithm load
import gym  # isort: skip # noqa F401
from srl.algorithms import dqn  # isort: skip


def main():
    env_config = srl.EnvConfig("Pendulum-v1")

    # atariのパラメータでは大げさすきるので Pendulum 用に縮小
    # あえて書いていますが、フレームワークのデフォルトは大体この値です。（詳細はコードを参照）
    rl_config = dqn.Config()
    rl_config.set_atari_config()
    rl_config.hidden_block_config = MLPBlockConfig(layer_sizes=(64, 64))  # 画像処理層がないので隠れ層を増やしてます
    rl_config.capacity = 100_000  # 下げないとメモリ足りなくなる可能性あり
    rl_config.window_length = 1  # Pendulum は加速も状態に含まれているので不要
    rl_config.lr = 0.001
    rl_config.exploration_steps = -1  # Annealing e-greedy は使わずに一定値を使用
    rl_config.epsilon = 0.1
    rl_config.memory_warmup_size = 1000  # そこまで待たずに学習開始
    rl_config.target_model_update_interval = 1000  # 大きすぎると学習がゆっくり
    rl_config.enable_reward_clip = False  # 報酬clipしない

    config = runner.Config(env_config, rl_config)
    config.model_summary(expand_nested=True)

    # --- train
    parameter, remote_memory, history = runner.train(config, max_episodes=200)
    history.plot(plot_right=["train_loss"])

    # --- evaluate
    rewards = runner.evaluate(config, parameter, max_episodes=20)
    print(f"Average reward for 20 episodes: {np.mean(rewards)}")

    # --- animation
    render = runner.animation(config, parameter)
    render.create_anime().save("_DQN_Pendulum.gif")


if __name__ == "__main__":
    from srl.utils import common
    common.logger_print()
    main()
