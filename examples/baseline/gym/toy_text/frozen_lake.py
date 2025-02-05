import os

import mlflow
import numpy as np

import srl
from srl.utils import common

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
common.logger_print()

ENV_NAME = "FrozenLake-v1"
BASE_LR = 0.1
BASE_TRAIN = 1_000_000


def _train(rl_config, train):
    runner = srl.Runner(ENV_NAME, rl_config)

    runner.set_mlflow()
    runner.train(max_train_count=train)

    rewards = runner.evaluate(max_episodes=1000)
    print(f"{rl_config.name}:{np.mean(rewards)} > 0.4")


def main_ql():
    from srl.algorithms import ql

    _train(ql.Config(lr=BASE_LR), BASE_TRAIN)


def main_policy():
    from srl.algorithms import vanilla_policy

    _train(vanilla_policy.Config(lr=BASE_LR), BASE_TRAIN)


def main_ql_agent57():
    from srl.algorithms import ql_agent57

    _train(ql_agent57.Config(lr_ext=BASE_LR, lr_int=BASE_LR), BASE_TRAIN)


def main_search_dynaq():
    from srl.algorithms import search_dynaq

    _train(search_dynaq.Config(q_ext_lr=BASE_LR, q_int_lr=BASE_LR), BASE_TRAIN)


def main_go_dynaq():
    from srl.algorithms import go_dynaq

    _train(go_dynaq.Config(q_ext_lr=BASE_LR, q_int_lr=BASE_LR), BASE_TRAIN)


def compare():
    import matplotlib.pyplot as plt

    from srl.runner.callbacks.mlflow_callback import MLFlowCallback

    metric_name = "eval_reward0"

    plt.figure(figsize=(12, 6))
    plt.xlabel("train")
    plt.ylabel(metric_name)
    for name in [
        "QL",
        "VanillaPolicy",
        "QL_Agent57",
        "SearchDynaQ",
        "GoDynaQ",
    ]:
        history = MLFlowCallback.get_metric(ENV_NAME, name, metric_name)
        if history is None:
            continue
        times = np.array([h.timestamp for h in history])
        times -= times[0]
        steps = [h.step for h in history]
        vals = [h.value for h in history]
        plt.plot(times, common.ema(vals), label=name)
    plt.grid()
    plt.legend()
    plt.title(f"Train:{BASE_TRAIN}, lr={BASE_LR}")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "_frozen_lake.png"))
    plt.show()


if __name__ == "__main__":
    main_ql()
    main_policy()
    main_ql_agent57()
    main_search_dynaq()
    main_go_dynaq()
    compare()
