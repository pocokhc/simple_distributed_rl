import os

import numpy as np

import srl
from srl.algorithms import ql


def test_train(tmpdir):
    rl_config = ql.Config()
    runner = srl.Runner("Grid", rl_config)

    runner.set_mlflow("test_Grid")
    runner.train(timeout=3)

    rewards = runner.evaluate(max_episodes=100)
    assert np.mean(rewards) > 0.6

    # --- reset
    rl_config = ql.Config()
    runner = srl.Runner("Grid", rl_config)
    rewards = runner.evaluate(max_episodes=100)
    print(rewards)
    assert np.mean(rewards) < 0.5

    runner.load_parameter_from_mlflow("test_Grid")
    runner.save_parameter(os.path.join(tmpdir, "_tmp.dat"))

    # --- reset
    rl_config = ql.Config()
    runner = srl.Runner("Grid", rl_config)
    rewards = runner.evaluate(max_episodes=100)
    assert np.mean(rewards) < 0.5

    runner.load_parameter(os.path.join(tmpdir, "_tmp.dat"))
    rewards = runner.evaluate(max_episodes=100)
    assert np.mean(rewards) > 0.6


def test_method():
    from srl.runner.callbacks.mlflow_callback import MLFlowCallback

    rl_config = ql.Config()
    runner = srl.Runner("Grid", rl_config)

    runner.set_mlflow("test_Grid2")
    runner.train(max_episodes=100)

    # --- metric
    metrics = MLFlowCallback.get_metric("Grid", rl_config.get_name(), "reward0")
    assert metrics is not None
    for metric in metrics:
        print(f"{metric.key=}")
        print(f"{metric.value=}")
        print(f"{metric.step=}")
        print(f"{metric.timestamp=}")

    # get
    exp_id = MLFlowCallback.get_experiment_id("test_Grid2")
    print(f"{exp_id=}")
    assert exp_id is not None

    run_id = MLFlowCallback.get_run_id("test_Grid2", rl_config.get_name())
    print(f"{run_id=}")
    assert run_id is not None
