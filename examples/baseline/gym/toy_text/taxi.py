import os

import numpy as np

import srl
from srl.algorithms import ql, ql_agent57, search_dynaq

base_dir = os.path.dirname(__file__)
BASE_LR = 0.05
TRAIN_COUNT = 1_000_000
ENV_PRE = "taxi"


def _train(name, rl_config):
    runner = srl.Runner("Taxi-v3", rl_config)
    runner.set_history_on_file(
        os.path.join(base_dir, f"_{ENV_PRE}_{name}"),
        enable_eval=True,
        eval_episode=10,
    )
    runner.train(max_train_count=TRAIN_COUNT)
    rewards = runner.evaluate(max_episodes=1000)
    print(f"{np.mean(rewards)} > 7")


def main_ql():
    _train("QL", ql.Config(lr=BASE_LR))


def main_ql_agent57():
    _train("QL_agent57", ql_agent57.Config(lr_ext=BASE_LR, lr_int=BASE_LR))


def main_search_dynaq():
    _train("SearchDynaQ", search_dynaq.Config(q_ext_lr=BASE_LR, q_int_lr=BASE_LR))


def compare():
    histories = srl.Runner.load_histories(
        [
            os.path.join(base_dir, f"_{ENV_PRE}_QL"),
            os.path.join(base_dir, f"_{ENV_PRE}_QL_agent57"),
            os.path.join(base_dir, f"_{ENV_PRE}_SearchDynaQ"),
        ]
    )
    histories.plot("train", "eval_reward0", title=f"Train:{TRAIN_COUNT}, lr={BASE_LR}")
    histories.plot("time", "eval_reward0", title=f"Train:{TRAIN_COUNT}, lr={BASE_LR}")


if __name__ == "__main__":
    main_ql()
    main_ql_agent57()
    main_search_dynaq()
    compare()
