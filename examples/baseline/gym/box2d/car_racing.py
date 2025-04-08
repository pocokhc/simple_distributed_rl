import os

import mlflow
import numpy as np

import srl
from srl.utils import common

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
common.logger_print()


def _run(rl_config, train, continuous: bool):
    env_config = srl.EnvConfig("CarRacing-v2", {"continuous": continuous})
    runner = srl.Runner(env_config, rl_config)
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
        window_length=4,
        memory=dqn.ReplayBufferConfig(
            capacity=10_000,
            warmup_size=1000,
            compress=False,
        ),
    )
    rl_config.hidden_block.set((512,))
    _run(rl_config, train=100_000, continuous=False)


if __name__ == "__main__":
    main_dqn()
