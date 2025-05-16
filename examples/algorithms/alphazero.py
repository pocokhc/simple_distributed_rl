import os

import numpy as np

import srl
from srl.algorithms import alphazero
from srl.utils import common

common.logger_print()


def main_ox():
    rl_config = alphazero.Config(
        num_simulations=50,
        sampling_steps=1,
        batch_size=32,
        lr=0.001,
        discount=1.0,
    )
    rl_config.memory.warmup_size = 100
    rl_config.input_image_block.set_alphazero_block(1, 32)
    rl_config.value_block.set((32,))
    rl_config.value_type = "rate"

    # 入力形式を WxHx2 の画像レイヤーで出力
    env_config = srl.EnvConfig("OX-layer")

    runner = srl.Runner(env_config, rl_config)

    # --- train(self play)
    runner.train_mp(players=[None, None], max_train_count=5_000)

    # --- evaluate
    for players in [
        [None, "random"],
        ["random", None],
        [None, "cpu"],
        ["cpu", None],
        ["random", "cpu"],
        ["cpu", "random"],
    ]:
        rewards = runner.evaluate(max_episodes=10, players=players, enable_progress=False)
        print(f"{players}: {np.mean(rewards, axis=0)}")

    # --- rendering
    path = os.path.join(os.path.dirname(__file__), "_alphazero_ox.gif")
    runner.animation_save_gif(path, players=[None, "cpu"])

    # --- 対戦
    # runner.render_terminal(players=[None, "human"])


def main_othello():
    rl_config = alphazero.Config(
        num_simulations=50,
        sampling_steps=1,
        batch_size=64,
    )
    rl_config.lr_scheduler.set_piecewise(
        [1000, 5000],
        [0.001, 0.0005, 0.0002],
    )
    rl_config.memory.capacity = 100_000
    rl_config.memory.warmup_size = 200
    rl_config.input_image_block.set_alphazero_block(3, 64)
    rl_config.value_block.set((64,))
    rl_config.policy_block.set(())

    # 入力形式を WxHx2 の画像レイヤーで出力
    env_config = srl.EnvConfig("Othello4x4-layer")

    runner = srl.Runner(env_config, rl_config)

    # --- train(self play)
    runner.train_mp(players=[None, None], max_train_count=10000)

    # --- evaluate
    for players in [
        [None, "random"],
        ["random", None],
        [None, "cpu"],
        ["cpu", None],
        ["random", "cpu"],
        ["cpu", "random"],
    ]:
        rewards = runner.evaluate(max_episodes=5, players=players, enable_progress=False)
        print(f"{players}: {np.mean(rewards, axis=0)}")

    # --- rendering
    path = os.path.join(os.path.dirname(__file__), "_alphazero.gif")
    runner.animation_save_gif(path, players=[None, "cpu"])

    # --- 対戦
    # runner.render_terminal(players=[None, "human"])

    runner.replay_window()


if __name__ == "__main__":
    # main_ox()
    main_othello()
