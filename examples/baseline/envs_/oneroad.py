import os

import numpy as np

import srl
from srl.algorithms import go_dynaq, ql, search_dynaq

base_dir = os.path.dirname(__file__)
TRAIN_COUNT = 1_000_000
ENV_PRE = "OneRoad"


def _train(name, rl_config):
    runner = srl.Runner("OneRoad-hard", rl_config)
    runner.set_history_on_file(
        os.path.join(base_dir, f"_{ENV_PRE}_{name}"),
        enable_eval=True,
        eval_episode=100,
    )
    runner.train(max_train_count=TRAIN_COUNT)
    rewards = runner.evaluate(max_episodes=1000)
    print(f"{np.mean(rewards)} >= 0.9")


def main_ql():
    _train("QL", ql.Config())


def main_search_dynaq():
    _train("SearchDynaQ", search_dynaq.Config())


def main_go_dynaq():
    _train("GoDynaQ", go_dynaq.Config())


def compare():
    histories = srl.Runner.load_histories(
        [
            os.path.join(base_dir, f"_{ENV_PRE}_QL"),
            os.path.join(base_dir, f"_{ENV_PRE}_SearchDynaQ"),
            os.path.join(base_dir, f"_{ENV_PRE}_GoDynaQ"),
        ]
    )
    histories.plot("train", "eval_reward0", title=f"Train:{TRAIN_COUNT}")
    histories.plot("time", "eval_reward0", title=f"Train:{TRAIN_COUNT}")


if __name__ == "__main__":
    main_ql()
    main_search_dynaq()
    main_go_dynaq()
    compare()
