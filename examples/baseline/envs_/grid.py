import mlflow
import numpy as np

import srl
from srl.algorithms import ql, search_dynaq
from srl.runner.callbacks.mlflow_callback import MLFlowCallback

BASE_TRAIN = 10_000

mlflow.set_tracking_uri("mlruns")


def main_ql():

    runner = srl.Runner("Grid", ql.Config())
    runner.train(max_train_count=BASE_TRAIN, callbacks=[MLFlowCallback()])
    rewards = runner.evaluate(max_episodes=1000)
    print(f"evaluate episodes: {np.mean(rewards)} > 0.6")


def main_search_dynaq():
    runner = srl.Runner("Grid", search_dynaq.Config())
    runner.train(max_train_count=int(BASE_TRAIN))
    rewards = runner.evaluate(max_episodes=1000)
    print(f"evaluate episodes: {np.mean(rewards)} > 0.6")


if __name__ == "__main__":
    main_ql()
    main_search_dynaq()
