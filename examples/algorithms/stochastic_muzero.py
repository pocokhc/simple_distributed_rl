import os

import numpy as np

import srl
from srl.algorithms import stochastic_muzero
from srl.envs import grid


def main():
    rl_config = stochastic_muzero.Config(
        num_simulations=10,
        discount=0.9,
        batch_size=16,
        memory_warmup_size=200,
        v_min=-2,
        v_max=2,
        unroll_steps=1,
        dynamics_blocks=1,
        enable_rescale=False,
        codebook_size=4,
    )
    rl_config.lr.set_linear(10_000, 0.01, 0.001)
    rl_config.memory.capacity = 100_000
    rl_config.input_image_block.set_alphazero_block(1, 16)

    rl_config.processors = [grid.LayerProcessor()]
    env_config = srl.EnvConfig("Grid")
    runner = srl.Runner(env_config, rl_config)

    runner.model_summary()

    # --- train
    runner.train(max_episodes=500)

    # --- evaluate
    rewards = runner.evaluate(max_episodes=10)
    print("mean", np.mean(rewards))

    # --- rendering
    path = os.path.join(os.path.dirname(__file__), "_stochastic_muzero.gif")
    runner.animation_save_gif(path)


if __name__ == "__main__":
    main()
