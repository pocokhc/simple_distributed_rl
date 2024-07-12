import os

import flappy_bird_gymnasium  # pip install flappy-bird-gymnasium
import mlflow
import numpy as np

import srl
from srl.utils import common

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
common.logger_print()


def _train(rl_config, train):
    runner = srl.Runner("FlappyBird-v0", rl_config)
    runner.model_summary()

    runner.set_mlflow()
    runner.train_mp(max_train_count=train)

    rewards = runner.evaluate()
    print(f"[{rl_config.name}] {np.mean(rewards)}, {rewards}")


def main_dqn():
    from srl.algorithms import dqn

    rl_config = dqn.Config(
        lr=0.0005,
        target_model_update_interval=2000,
        memory_warmup_size=1000,
        memory_capacity=10_000,
        memory_compress=False,
    )
    rl_config.hidden_block.set((256, 256))
    _train(rl_config, train=500_000)


if __name__ == "__main__":
    main_dqn()
