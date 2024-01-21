import os

import numpy as np

import srl
from srl.algorithms import ql, ql_agent57, search_dynaq

base_dir = os.path.dirname(__file__)
BASE_LR = 0.05
BASE_TRAIN = 1_000_000
ENV_PRE = "taxi"


def main_ql():
    runner = srl.Runner("Taxi-v3", ql.Config(lr=BASE_LR))

    runner.set_history_on_file(
        os.path.join(base_dir, f"_{ENV_PRE}_QL"),
        enable_eval=True,
        eval_episode=100,
    )
    runner.train(max_train_count=BASE_TRAIN)
    rewards = runner.evaluate(max_episodes=1000)
    print(f"evaluate episodes: {np.mean(rewards)} > 7")


def main_ql_agent57():
    runner = srl.Runner("Taxi-v3", ql_agent57.Config(lr_ext=BASE_LR, lr_int=BASE_LR))

    runner.set_history_on_file(
        os.path.join(base_dir, f"_{ENV_PRE}_QL_agent57"),
        enable_eval=True,
        eval_episode=100,
    )
    runner.train(max_train_count=BASE_TRAIN)
    rewards = runner.evaluate(max_episodes=1000)
    print(f"evaluate episodes: {np.mean(rewards)} > 7")


def main_search_dynaq():
    rl_config = search_dynaq.Config(q_ext_lr=BASE_LR, q_int_lr=BASE_LR)
    runner = srl.Runner("Taxi-v3", rl_config)
    runner.set_history_on_file(
        os.path.join(base_dir, f"_{ENV_PRE}_SearchDynaQ"),
        enable_eval=True,
        eval_episode=100,
    )
    runner.train(max_train_count=int(BASE_TRAIN / 50))

    rewards = runner.evaluate(max_episodes=1000)
    print(f"evaluate episodes: {np.mean(rewards)} > 7")


def compare():
    histories = srl.Runner.load_histories(
        [
            os.path.join(base_dir, f"_{ENV_PRE}_QL"),
            os.path.join(base_dir, f"_{ENV_PRE}_QL_agent57"),
            os.path.join(base_dir, f"_{ENV_PRE}_SearchDynaQ"),
        ]
    )
    histories.plot("train", "eval_reward0", title=f"Train:{BASE_TRAIN}, lr={BASE_LR}")
    histories.plot("time", "eval_reward0", title=f"Train:{BASE_TRAIN}, lr={BASE_LR}")


if __name__ == "__main__":
    main_ql()
    main_ql_agent57()
    main_search_dynaq()
    compare()
