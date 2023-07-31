import numpy as np

import srl
from srl.algorithms import sac
from srl.utils import common

common.logger_print()


def main():
    env_config = srl.EnvConfig("Pendulum-v1")
    rl_config = sac.Config()

    runner = srl.Runner(env_config, rl_config)
    runner.model_summary(expand_nested=True)

    # --- train
    runner.train(max_episodes=25)

    # --- evaluate
    rewards = runner.evaluate(max_episodes=20)
    print(f"Average reward for 20 episodes: {np.mean(rewards)}")


if __name__ == "__main__":
    main()
