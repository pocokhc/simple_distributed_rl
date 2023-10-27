import os

import numpy as np

import srl
from srl.algorithms import muzero
from srl.envs import grid
from srl.utils import common

common.logger_print()


def main():
    rl_config = muzero.Config(
        num_simulations=10,
        discount=0.9,
        batch_size=16,
        v_min=-2,
        v_max=2,
        unroll_steps=1,
        dynamics_blocks=1,
        enable_rescale=False,
        weight_decay=0,
    )
    rl_config.memory.warmup_size = 200
    rl_config.lr.set_linear(10_000, 0.002, 0.001)
    rl_config.memory.capacity = 100_000
    rl_config.input_image_block.set_alphazero_block(1, 16)

    rl_config.processors = [grid.LayerProcessor()]
    env_config = srl.EnvConfig("EasyGrid")
    runner = srl.Runner(env_config, rl_config)

    runner.model_summary()

    # --- 学習ループ
    runner.train(max_episodes=100)

    # --- evaluate
    rewards = runner.evaluate(max_episodes=10)
    print("mean", np.mean(rewards))

    # --- rendering
    path = os.path.join(os.path.dirname(__file__), "_muzero.gif")
    runner.animation_save_gif(path)


if __name__ == "__main__":
    main()
