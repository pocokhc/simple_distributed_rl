import numpy as np

import srl
from srl.algorithms import vanilla_policy
from srl.base.define import RLTypes


def main_discrete():
    env_config = srl.EnvConfig("Grid")
    rl_config = vanilla_policy.Config()
    runner = srl.Runner(env_config, rl_config)

    # --- train
    runner.train(timeout=10)

    # --- evaluate
    rewards = runner.evaluate(max_episodes=100)
    print(f"Average reward for 100 episodes: {np.mean(rewards)}")

    # --- render
    runner.render_terminal()


def main_continuous():
    env_config = srl.EnvConfig("Grid")
    rl_config = vanilla_policy.Config()
    rl_config.override_action_type = RLTypes.CONTINUOUS
    runner = srl.Runner(env_config, rl_config)

    # --- train
    runner.train(timeout=30)

    # --- evaluate
    rewards = runner.evaluate(max_episodes=100)
    print(f"Average reward for 100 episodes: {np.mean(rewards)}")

    # --- render
    runner.render_terminal()


if __name__ == "__main__":
    main_discrete()
    main_continuous()
