import os

import numpy as np

import srl
from srl import runner
from srl.rl import memories
from srl.rl.models import alphazero as alphazero_model

# --- env & algorithm load
from srl.envs import grid  # isort: skip # noqa F401
from srl.algorithms import stochastic_muzero  # isort: skip


def main():
    rl_config = stochastic_muzero.Config(
        num_simulations=10,
        discount=0.9,
        batch_size=16,
        memory_warmup_size=200,
        memory=memories.ReplayMemoryConfig(100_000),
        lr_init=0.01,
        lr_decay_steps=10_000,
        v_min=-2,
        v_max=2,
        unroll_steps=1,
        input_image_block=alphazero_model.AlphaZeroImageBlockConfig(n_blocks=1, filters=16),
        dynamics_blocks=1,
        enable_rescale=False,
        codebook_size=4,
    )
    rl_config.processors = [grid.LayerProcessor()]
    env_config = srl.EnvConfig("Grid")
    config = runner.Config(env_config, rl_config)

    config.model_summary()

    # --- 学習ループ
    parameter, _, _ = runner.train(config, max_episodes=500)

    # --- evaluate
    rewards = runner.evaluate(config, parameter, max_episodes=10, print_progress=True)
    print("mean", np.mean(rewards))

    # --- rendering
    render = runner.animation(config, parameter, print_progress=True)
    render.create_anime(draw_info=True).save(os.path.join(os.path.dirname(__file__), "_stochastic_muzero.gif"))


if __name__ == "__main__":
    main()
