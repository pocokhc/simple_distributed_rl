import os

import numpy as np

import srl
from srl.algorithms import ql, search_dynaq

base_dir = os.path.dirname(__file__)
BASE_LR = 0.1
TRAIN_COUNT = 200_000
ENV_PRE = "MG_DO_6x6"


def _train(name, rl_config):
    runner = srl.Runner("MiniGrid-Dynamic-Obstacles-6x6-v0", rl_config)
    runner.set_history_on_file(
        os.path.join(base_dir, f"_{ENV_PRE}_{name}"),
        enable_eval=True,
        eval_episode=100,
    )
    runner.train(max_train_count=TRAIN_COUNT)
    rewards = runner.evaluate(max_episodes=1000)
    print(f"{np.mean(rewards)} > 0.5")


def main_ql():
    _train("QL", ql.Config(lr=BASE_LR))


def main_search_dynaq():
    _train("SearchDynaQ", search_dynaq.Config(q_ext_lr=BASE_LR, q_int_lr=BASE_LR))


def compare():
    names = [
        "QL",
        "SearchDynaQ",
    ]
    histories = srl.Runner.load_histories([os.path.join(base_dir, f"_{ENV_PRE}_{n}") for n in names])
    histories.plot("train", "eval_reward0", title=f"Train:{TRAIN_COUNT}, lr={BASE_LR}")
    histories.plot("time", "eval_reward0", title=f"Train:{TRAIN_COUNT}, lr={BASE_LR}")


if __name__ == "__main__":
    main_ql()
    main_search_dynaq()
    compare()
