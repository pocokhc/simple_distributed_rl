import numpy as np

import srl
from srl.algorithms import ql, search_dynaq

BASE_TRAIN = 10_000


def main_ql():
    runner = srl.Runner("Grid", ql.Config())
    runner.train(max_train_count=BASE_TRAIN)
    rewards = runner.evaluate(max_episodes=1000)
    print(f"evaluate episodes: {np.mean(rewards)} > 0.6")


def main_search_dynaq():
    runner = srl.Runner("Grid", search_dynaq.Config())
    runner.train(max_train_count=int(BASE_TRAIN / 10))
    rewards = runner.evaluate(max_episodes=1000)
    print(f"evaluate episodes: {np.mean(rewards)} > 0.6")


if __name__ == "__main__":
    main_ql()
    main_search_dynaq()
