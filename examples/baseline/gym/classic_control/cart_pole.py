import os

import mlflow
import numpy as np

import srl
from srl.utils import common

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
common.logger_print()


def run(rl_config, train):
    runner = srl.Runner("CartPole-v1", rl_config)

    runner.set_mlflow()
    runner.train(max_train_count=train)

    rewards = runner.evaluate()
    print(f"[{rl_config.name}] {np.mean(rewards)}, {rewards}")


def main_dqn():
    from srl.algorithms import dqn

    rl_config = dqn.Config(
        lr=0.001,
        target_model_update_interval=1000,
        memory=dqn.PriorityReplayBufferConfig(
            capacity=10_000,
            warmup_size=1000,
            compress=False,
        ),
    )
    rl_config.hidden_block.set((64, 64))
    run(rl_config, train=100_000)


if __name__ == "__main__":
    main_dqn()
