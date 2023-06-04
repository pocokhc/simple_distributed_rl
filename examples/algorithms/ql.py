import os

import numpy as np

import srl
from srl import runner

# --- env & algorithm load
from srl.envs import grid  # isort: skip # noqa F401
from srl.algorithms import ql  # isort: skip


def main():
    env_config = srl.EnvConfig("Grid")
    rl_config = ql.Config()

    # --- train
    config = runner.Config(env_config, rl_config)
    parameter, remote_memory, history = runner.train(config, timeout=10)

    # --- evaluate
    rewards = runner.evaluate(config, parameter, max_episodes=100)
    print(f"Average reward for 100 episodes: {np.mean(rewards)}")

    # --- render
    runner.render_terminal(config, parameter)

    # --- animation
    render = runner.animation(config, parameter)
    render.create_anime().save(os.path.join(os.path.dirname(__file__), "_ql.gif"))


if __name__ == "__main__":
    main()
