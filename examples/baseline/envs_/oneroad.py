import os

import numpy as np

import srl
from srl.algorithms import ql, search_dynaq

base_dir = os.path.dirname(__file__)
BASE_TRAIN = 10_000_000
ENV_PRE = "OneRoad"


def main_ql():
    runner = srl.Runner("OneRoad-hard", ql.Config())

    runner.set_history_on_file(
        os.path.join(base_dir, f"_{ENV_PRE}_QL"),
        enable_eval=True,
        eval_episode=100,
    )
    runner.train(max_train_count=BASE_TRAIN)
    rewards = runner.evaluate(max_episodes=1000)
    print(f"{np.mean(rewards)} >= 0.0")


def main_search_dynaq():
    rl_config = search_dynaq.Config()

    runner = srl.Runner("OneRoad-hard", rl_config)
    runner.set_history_on_file(
        os.path.join(base_dir, f"_{ENV_PRE}_SearchDynaQ"),
        enable_eval=True,
        eval_episode=100,
    )
    runner.train(max_train_count=int(BASE_TRAIN))
    rewards = runner.evaluate(max_episodes=1000)
    print(f"{np.mean(rewards)} > 0.9")


def compare():
    histories = srl.Runner.load_histories(
        [
            os.path.join(base_dir, f"_{ENV_PRE}_QL"),
            os.path.join(base_dir, f"_{ENV_PRE}_SearchDynaQ"),
        ]
    )
    histories.plot("train", "eval_reward0", title=f"Train:{BASE_TRAIN}")
    histories.plot("time", "eval_reward0", title=f"Train:{BASE_TRAIN}")


if __name__ == "__main__":
    main_ql()
    main_search_dynaq()
    compare()
