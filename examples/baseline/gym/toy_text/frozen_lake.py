import os

import numpy as np

import srl
from srl.algorithms import ql, ql_agent57, search_dynaq

base_dir = os.path.dirname(__file__)
BASE_LR = 0.005
BASE_TRAIN = 2_000_000


def main_ql():
    runner = srl.Runner("FrozenLake-v1", ql.Config(lr=BASE_LR))

    runner.set_history_on_file(
        os.path.join(base_dir, "_fl_QL"),
        enable_eval=True,
        eval_episode=100,
    )
    runner.train(max_train_count=BASE_TRAIN)
    rewards = runner.evaluate(max_episodes=1000)
    print(f"{np.mean(rewards)} > 0.4")


def main_ql_agent57():
    runner = srl.Runner("FrozenLake-v1", ql_agent57.Config(lr_ext=BASE_LR, lr_int=BASE_LR * 10))

    runner.set_history_on_file(
        os.path.join(base_dir, "_fl_QL_Agent57"),
        enable_eval=True,
        eval_episode=100,
    )
    runner.train(max_train_count=BASE_TRAIN)
    rewards = runner.evaluate(max_episodes=1000)
    print(f"{np.mean(rewards)} > 0.4")


def main_search_dynaq():
    rl_config = search_dynaq.Config(lr_ext=BASE_LR, lr_int=BASE_LR * 10)

    runner = srl.Runner("FrozenLake-v1", rl_config)
    runner.set_history_on_file(
        os.path.join(base_dir, "_fl_SearchDynaQ"),
        enable_eval=True,
        eval_episode=100,
    )

    # 1st
    rl_config.search_mode = True
    runner.train(max_train_count=int(BASE_TRAIN / 100))

    # 2nd
    rl_config.search_mode = False
    runner.train_only(max_train_count=BASE_TRAIN)

    rewards = runner.evaluate(max_episodes=1000)
    print(f"{np.mean(rewards)} > 0.6")


def compare():
    _env = "fl"
    histories = srl.Runner.load_histories(
        [
            os.path.join(base_dir, f"_{_env}_QL"),
            os.path.join(base_dir, f"_{_env}_QL_Agent57"),
            os.path.join(base_dir, f"_{_env}_SearchDynaQ"),
        ]
    )
    histories.plot("train", "eval_reward0", title=f"Train:{BASE_TRAIN}, lr={BASE_LR}")
    histories.plot("time", "eval_reward0", title=f"Train:{BASE_TRAIN}, lr={BASE_LR}")


if __name__ == "__main__":
    main_ql()
    main_ql_agent57()
    main_search_dynaq()
    compare()
