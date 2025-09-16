import os

# MiniGrid Documentation https://minigrid.farama.org/
import minigrid  # noqa: F401
import mlflow
import numpy as np

import srl
from srl.utils import common

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
common.logger_print()


ENV_NAME = "MiniGrid-Dynamic-Obstacles-6x6-v0"
BASE_LR = 0.1
BASE_TRAIN = 300_000


def _train(rl_config, train):
    runner = srl.Runner(ENV_NAME, rl_config)

    runner.set_mlflow()
    runner.train(max_train_count=train)

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
        "SearchDynaQ",
        "GoDynaQ",
    ]:
        run_id = MLFlowCallback.get_run_id(ENV_NAME, rl_name=name)
        history = MLFlowCallback.get_metric(run_id, metric_name)
        if history is None:
            continue
        times = np.array([h.timestamp for h in history])
        times -= times[0]
        steps = [h.step for h in history]
        vals = [h.value for h in history]
        plt.plot(steps, common.rolling(vals), label=name)
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
    main_go_dynaq()
    compare()
