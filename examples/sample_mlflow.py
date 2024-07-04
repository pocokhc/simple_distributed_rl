import mlflow

import srl
from srl.algorithms import ql, vanilla_policy
from srl.runner.callbacks.mlflow_callback import MLFlowCallback

# mlflow ui --backend-store-uri mlruns
mlflow.set_tracking_uri("mlruns")


def train_ql():
    env_config = srl.EnvConfig("Grid")
    rl_config = ql.Config()
    runner = srl.Runner(env_config, rl_config)
    runner.train(timeout=60, callbacks=[MLFlowCallback()])


def train_vanilla_policy():
    env_config = srl.EnvConfig("Grid")
    rl_config = vanilla_policy.Config()
    runner = srl.Runner(env_config, rl_config)
    runner.train(timeout=60, callbacks=[MLFlowCallback()])


if __name__ == "__main__":
    train_ql()
    train_vanilla_policy()
