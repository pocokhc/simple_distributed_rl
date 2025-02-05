import os

import mlflow
import numpy as np

import srl
from srl.utils import common

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
common.logger_print()

ENV_NAME = "OX"
BASE_TRAIN = 1_000_000


def _train(rl_config, train):
    runner = srl.Runner(ENV_NAME, rl_config)

    runner.set_mlflow(eval_players=[None, "cpu"])

    runner.train(max_train_count=train, players=[None, None])

    for players in [
        [None, "random"],
        ["random", None],
        [None, "cpu"],
        ["cpu", None],
    ]:
        rewards = runner.evaluate(max_episodes=100, players=players)
        print(f"Average reward for 100 episodes: {np.mean(rewards, axis=0)}, {players}")


def main_ql():
    from srl.algorithms import ql

    _train(ql.Config(), BASE_TRAIN)


def main_mcts():
    from srl.algorithms import mcts

    _train(mcts.Config(), BASE_TRAIN)


def main_search_dynaq():
    from srl.algorithms import search_dynaq

    _train(search_dynaq.Config(), BASE_TRAIN)


def main_go_dynaq():
    from srl.algorithms import go_dynaq

    _train(go_dynaq.Config(), BASE_TRAIN)


def compare():
    import matplotlib.pyplot as plt

    from srl.runner.callbacks.mlflow_callback import MLFlowCallback

    metric_name = "eval_reward0"

    plt.figure(figsize=(12, 6))
    plt.xlabel("train")
    plt.ylabel(metric_name)
    for name in [
        "QL",
        "MCTS",
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
    plt.title(f"Train:{BASE_TRAIN}")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "_ox.png"))
    plt.show()


if __name__ == "__main__":
    main_ql()
    main_mcts()
    main_search_dynaq()
    main_go_dynaq()
    compare()
