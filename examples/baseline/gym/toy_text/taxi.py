import os

import mlflow
import numpy as np

import srl
from srl.runner.callbacks.mlflow_callback import MLFlowCallback
from srl.utils import common

mlflow.set_tracking_uri("mlruns")
common.logger_print()

ENV_NAME = "Taxi-v3"
BASE_LR = 0.05
BASE_TRAIN = 200_000


def _train(rl_config, train):
    runner = srl.Runner(ENV_NAME, rl_config)
    runner.train(max_train_count=train, callbacks=[MLFlowCallback(interval_eval=1)])
    rewards = runner.evaluate(max_episodes=1000)
    print(f"{rl_config.name}:{np.mean(rewards)} > 7")


def main_ql():
    from srl.algorithms import ql

    _train(ql.Config(lr=BASE_LR, epsilon=0.5), BASE_TRAIN)


def main_policy():
    from srl.algorithms import vanilla_policy

    _train(vanilla_policy.Config(lr=BASE_LR), BASE_TRAIN)


def main_ql_agent57():
    from srl.algorithms import ql_agent57

    _train(ql_agent57.Config(lr_ext=BASE_LR, lr_int=BASE_LR, epsilon=0.5), BASE_TRAIN)


def main_search_dynaq():
    from srl.algorithms import search_dynaq

    _train(search_dynaq.Config(q_ext_lr=BASE_LR, q_int_lr=BASE_LR), BASE_TRAIN)


def compare():
    import matplotlib.pyplot as plt

    metric_name = "eval_reward0"

    plt.clf()
    plt.close()
    plt.figure(figsize=(12, 6))
    plt.xlabel("train")
    plt.ylabel(metric_name)
    for name in [
        "QL",
        "QL_Agent57",
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
    plt.savefig(os.path.join(os.path.dirname(__file__), "_taxi.png"))
    plt.show()


def get_metrics(run_id, metric_name):
    client = mlflow.tracking.MlflowClient()
    metric_history = client.get_metric_history(run_id, metric_name)
    values = [m.value for m in metric_history]
    steps = [m.step for m in metric_history]
    return steps, values


if __name__ == "__main__":
    main_ql()
    main_policy()
    main_ql_agent57()
    main_search_dynaq()
    compare()
