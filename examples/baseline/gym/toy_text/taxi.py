import numpy as np

import srl
from srl.algorithms import ql, search_dynaq


def main_ql():
    runner = srl.Runner("Taxi-v3", ql.Config())

    runner.train(max_train_count=500_000)

    rewards = runner.evaluate(max_episodes=1000)
    print(f"evaluate episodes: {np.mean(rewards)} > 7")


def main_search_dynaq():
    rl_config = search_dynaq.Config()
    runner = srl.Runner("Taxi-v3", rl_config)

    rl_config.search_mode = True
    runner.train(max_train_count=100_000)

    rl_config.search_mode = False
    runner.train_only(max_train_count=1_000_000)

    rewards = runner.evaluate(max_episodes=1000)
    print(f"evaluate episodes: {np.mean(rewards)} > 7")


if __name__ == "__main__":
    main_ql()
    main_search_dynaq()
