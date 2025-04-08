import os

import numpy as np

import srl
from srl.algorithms import go_explore
from srl.utils import common

common.logger_print()


def main():
    env_config = srl.EnvConfig("Grid")
    rl_config = go_explore.Config(
        lr=0.001,
        epsilon=0.1,
        target_model_update_interval=2000,
        memory_warmup_size=1000,
        memory_capacity=10_000,
        downsampling_size=(16, 16),
        downsampling_val=4,
    )
    rl_config.hidden_block.set((64, 64))

    runner = srl.Runner(env_config, rl_config)
    runner.set_progress(interval_limit=30, enable_eval=True)
    runner.summary()

    # phase1
    runner.rollout(max_steps=10_000)
    # phase2
    runner.train(max_train_count=10_000)

    # --- evaluate
    rewards = runner.evaluate()
    print(f"eval: {np.mean(rewards)}, {rewards}")

    # --- animation
    path = os.path.join(os.path.dirname(__file__), "_go_explore.gif")
    runner.animation_save_gif(path)


if __name__ == "__main__":
    main()
