import numpy as np

import srl
from srl import runner
from srl.utils import common

# --- env & algorithm load
import gym  # isort: skip # noqa F401
from srl.algorithms import sac  # isort: skip


common.logger_print()


def main():
    env_config = srl.EnvConfig("Pendulum-v1")
    rl_config = sac.Config()

    config = runner.Config(env_config, rl_config, seed=1)
    config.model_summary(expand_nested=True)

    # --- train
    parameter, _, _ = runner.train(config, max_episodes=25)

    # --- evaluate
    rewards = runner.evaluate(config, parameter, max_episodes=20)
    print(f"Average reward for 20 episodes: {np.mean(rewards)}")


if __name__ == "__main__":
    main()
