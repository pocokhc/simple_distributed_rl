import os

import numpy as np
import srl
from srl.algorithms import go_explore
from srl.utils import common

common.logger_print()


def main():
    env_config = srl.EnvConfig("MountainCar-v0")
    rl_config = go_explore.Config(
        lr=0.001,
        epsilon=0.1,
        target_model_update_interval=2000,
        memory_warmup_size=1000,
        memory_capacity=50_000,
    )
    rl_config.hidden_block.set((128,))

    runner = srl.Runner(env_config, rl_config)
    runner.model_summary()

    # phase1
    runner.rollout(max_steps=500_000)
    # phase2
    runner.set_progress(interval_limit=60, enable_eval=True)
    runner.train(max_train_count=200_000)

    # --- evaluate
    rewards = runner.evaluate()
    print(f"eval: {np.mean(rewards)}, {rewards}")

    # --- animation
    path = os.path.join(os.path.dirname(__file__), "_go_explore.gif")
    runner.animation_save_gif(path)


if __name__ == "__main__":
    main()
