import os

import numpy as np

import srl
from srl.algorithms import stochastic_muzero


def main():
    rl_config = stochastic_muzero.Config(
        num_simulations=10,
        discount=0.9,
        batch_size=16,
        v_min=-2,
        v_max=2,
        unroll_steps=1,
        dynamics_blocks=1,
        enable_rescale=False,
        codebook_size=4,
    )
    rl_config.lr = 0.01
    rl_config.lr_scheduler.set_step(10_000, 0.001)
    rl_config.memory.capacity = 100_000
    rl_config.memory.warmup_size = 200
    rl_config.input_image_block.set_alphazero_block(1, 16)

    env_config = srl.EnvConfig("Grid-layer")
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
