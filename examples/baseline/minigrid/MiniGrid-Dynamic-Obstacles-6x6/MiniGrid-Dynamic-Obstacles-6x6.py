import os

import mlflow
import numpy as np

import srl
from srl.runner.callbacks.mlflow_callback import MLFlowCallback
from srl.utils import common

mlflow.set_tracking_uri("mlruns")
common.logger_print()

ENV_NAME = "MiniGrid-Dynamic-Obstacles-6x6-v0"
BASE_LR = 0.1
BASE_TRAIN = 300_000


def _train(rl_config, train):
    runner = srl.Runner(ENV_NAME, rl_config)
    runner.train(max_train_count=train, callbacks=[MLFlowCallback(interval_eval=1)])
    rewards = runner.evaluate(max_episodes=100)
    print(f"{np.mean(rewards)} > 0.5")


def main_ql():
    from srl.algorithms import ql

    _train(ql.Config(lr=BASE_LR), BASE_TRAIN)


def main_policy():
    from srl.algorithms import vanilla_policy

    _train(vanilla_policy.Config(lr=BASE_LR), BASE_TRAIN)


def main_search_dynaq():
    from srl.algorithms import search_dynaq

    _train(
        search_dynaq.Config(
            q_ext_lr=BASE_LR,
            q_int_lr=BASE_LR,
            iteration_interval=50_000,
        ),
        BASE_TRAIN,
    )


def compare():
    import matplotlib.pyplot as plt

    metric_name = "eval_reward0"

    plt.figure(figsize=(12, 6))
    plt.xlabel("train")
    plt.ylabel(metric_name)
    for name in [
        "QL",
        "VanillaPolicy",
        "SearchDynaQ",
    ]:
        steps, values = MLFlowCallback.get_metrics(ENV_NAME, name, metric_name)
        if steps is None:
            continue
        if len(values) > 20:
            values = common.moving_average(values, 10)
        plt.plot(steps, values, label=name)
    plt.grid()
    plt.legend()
    plt.title(f"Train:{BASE_TRAIN}, lr={BASE_LR}")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "_MiniGrid-Dynamic-Obstacles-6x6.png"))
    plt.show()


if __name__ == "__main__":
    main_ql()
    main_policy()
    main_search_dynaq()
    compare()
