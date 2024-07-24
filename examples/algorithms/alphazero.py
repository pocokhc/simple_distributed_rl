import os

import numpy as np

import srl
from srl.algorithms import alphazero
from srl.utils import common

common.logger_print()


def main():
    rl_config = alphazero.Config(
        num_simulations=50,
        sampling_steps=1,
        batch_size=128,
    )
    rl_config.lr = rl_config.create_scheduler()
    rl_config.lr.add_constant(1000, 0.001)
    rl_config.lr.add_constant(4000, 0.0005)
    rl_config.lr.add_constant(1, 0.0002)
    rl_config.memory_capacity = 100_000
    rl_config.memory_warmup_size = 500
    rl_config.input_image_block.set_alphazero_block(9, 64)
    rl_config.value_block.set((128,))
    rl_config.policy_block.set((128,))

    """
    othelloは入力形式を変えています。
    変更後は WxHx2 の画像レイヤーで、WxHが盤面に相当します。
    """
    env_config = srl.EnvConfig("Othello4x4", {"obs_type": "layer"})

    runner = srl.Runner(env_config, rl_config)
    runner.model_summary()

    # --- train(self play)
    runner.train(players=[None, None], max_episodes=1000)

    # --- evaluate
    for players in [
        [None, "random"],
        ["random", None],
        [None, "cpu"],
        ["cpu", None],
        ["random", "cpu"],
        ["cpu", "random"],
    ]:
        rewards = runner.evaluate(max_episodes=10, players=players)
        print(f"Average reward for 100 episodes: {np.mean(rewards, axis=0)}, {players}")

    # --- rendering
    path = os.path.join(os.path.dirname(__file__), "_alphazero.gif")
    runner.animation_save_gif(path, players=[None, "cpu"])

    # --- 対戦
    runner.render_terminal(players=[None, "human"])


if __name__ == "__main__":
    main()
