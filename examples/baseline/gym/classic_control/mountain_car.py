import os

import mlflow
import numpy as np

import srl
from srl.utils import common

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
common.logger_print()


def run(rl_config, train):
    runner = srl.Runner("MountainCar-v0", rl_config)
    runner.model_summary()

    runner.set_mlflow()
    runner.train(max_train_count=train)

    rewards = runner.evaluate()
    print(f"[{rl_config.name}] {np.mean(rewards)}, {rewards}")


def main_ql():
    from srl.algorithms import ql

    rl_config = ql.Config()
    run(rl_config, train=1_000_000)


def main_dqn():
    from srl.algorithms import dqn

    rl_config = dqn.Config(
        lr=0.001,
        epsilon=0.5,
        target_model_update_interval=2000,
        memory=dqn.ReplayBufferConfig(
            capacity=10_000,
            warmup_size=1000,
            compress=False,
        ),
    )
    rl_config.hidden_block.set((128,))
    run(rl_config, train=200_000)


if __name__ == "__main__":
    main_ql()
    main_dqn()
