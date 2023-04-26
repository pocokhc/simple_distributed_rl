import os

import numpy as np

import srl
from srl import runner
from srl.rl.models import AlphaZeroImageBlockConfig
from srl.utils import common

# --- env & algorithm load
from srl.envs import grid  # isort: skip # noqa F401
from srl.algorithms import muzero  # isort: skip

common.logger_print()


def main():
    rl_config = muzero.Config(
        num_simulations=10,
        discount=0.9,
        batch_size=16,
        memory_warmup_size=200,
        memory_name="ReplayMemory",
        lr_init=0.002,
        lr_decay_steps=10_000,
        v_min=-2,
        v_max=2,
        unroll_steps=1,
        input_image_block=AlphaZeroImageBlockConfig(n_blocks=1, filters=16),
        dynamics_blocks=1,
        enable_rescale=False,
        weight_decay=0,
    )
    rl_config.processors = [grid.LayerProcessor()]
    env_config = srl.EnvConfig("EasyGrid")
    config = runner.Config(env_config, rl_config)

    config.model_summary()

    # --- 学習ループ
    parameter, memory, history = runner.train(config, max_episodes=100)

    # --- evaluate
    rewards = runner.evaluate(config, parameter, max_episodes=10, print_progress=True)
    print("mean", np.mean(rewards))

    # --- rendering
    render = runner.animation(config, parameter, print_progress=True)
    render.create_anime(draw_info=True).save(os.path.join(os.path.dirname(__file__), "_muzero.gif"))


if __name__ == "__main__":
    main()
