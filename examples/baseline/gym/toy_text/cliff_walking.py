import os

import numpy as np

import srl
from srl.algorithms import ql, search_dynaq

base_dir = os.path.dirname(__file__)
BASE_LR = 0.02
TRAIN_TIMEOUT = 10
ENV_PRE = "cw"


def _train(name, rl_config):
    runner = srl.Runner("CliffWalking-v0", rl_config)
    runner.set_history_on_file(
        os.path.join(base_dir, f"_{ENV_PRE}_{name}"),
        enable_eval=True,
        eval_episode=1,
    )
    runner.train(timeout=TRAIN_TIMEOUT)
    rewards = runner.evaluate(max_episodes=10)
    print(f"evaluate episodes: {np.mean(rewards)} > -20")


def main_ql():
    _train("QL", ql.Config(lr=BASE_LR))


def main_search_dynaq():
    _train("SearchDynaQ", search_dynaq.Config(q_ext_lr=BASE_LR, q_int_lr=BASE_LR))


def compare():
    histories = srl.Runner.load_histories(
        [
            os.path.join(base_dir, f"_{ENV_PRE}_QL"),
            os.path.join(base_dir, f"_{ENV_PRE}_SearchDynaQ"),
        ]
    )
    histories.plot("train", "eval_reward0", title=f"TrainTime:{TRAIN_TIMEOUT}s, lr={BASE_LR}")
    histories.plot("time", "eval_reward0", title=f"TrainTime:{TRAIN_TIMEOUT}s, lr={BASE_LR}")


if __name__ == "__main__":
    main_ql()
    main_search_dynaq()
    compare()
