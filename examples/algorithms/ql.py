import os

import numpy as np

import srl
from srl.algorithms import ql


def main():
    env_config = srl.EnvConfig("Grid")
    rl_config = ql.Config()
    runner = srl.Runner(env_config, rl_config)

    # --- train
    runner.train(timeout=10)

    # --- evaluate
    rewards = runner.evaluate(max_episodes=100)
    print(f"Average reward for 100 episodes: {np.mean(rewards)}")

    # --- render
    runner.render_terminal()

    # --- animation
    path = os.path.join(os.path.dirname(__file__), "_ql.gif")
    runner.animation_save_gif(path)


if __name__ == "__main__":
    main()
