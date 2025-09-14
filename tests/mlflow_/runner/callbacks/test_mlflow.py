import os

import numpy as np
import pytest

import srl
from srl.algorithms import ql


def test_train(tmpdir):
    pytest.importorskip("mlflow")

    import mlflow

    mldir = os.path.join(tmpdir, "mlruns")
    print(mldir)
    mlflow.set_tracking_uri("file:///" + mldir)

    rl_config = ql.Config()
    runner = srl.Runner("Grid", rl_config)

    runner.set_mlflow(experiment_name="test_Grid", checkpoint_interval=1)
    runner.train(max_train_count=100_000)

    rewards = runner.evaluate(max_episodes=100)
    print(rewards)
    assert np.mean(rewards) > 0.6

    # --- reset
    rl_config = ql.Config()
    runner = srl.Runner("Grid", rl_config)
    rewards = runner.evaluate(max_episodes=100)
    print(rewards)
    assert np.mean(rewards) < 0.5

    runner.load_parameter_from_mlflow(experiment_name="test_Grid")
    rewards = runner.evaluate(max_episodes=100)
    print(rewards)
    assert np.mean(rewards) > 0.6
    runner.save_parameter(os.path.join(tmpdir, "_tmp.dat"))

    # --- reset
    rl_config = ql.Config()
    runner = srl.Runner("Grid", rl_config)
    rewards = runner.evaluate(max_episodes=100)
    assert np.mean(rewards) < 0.5

    runner.load_parameter(os.path.join(tmpdir, "_tmp.dat"))
    rewards = runner.evaluate(max_episodes=100)
    print(rewards)
    assert np.mean(rewards) > 0.6

    # --- html,  _tkinter.TclError が出るので一旦保留
    # runner.make_html_all_parameters_in_mlflow(experiment_name="test_Grid")


def test_method(tmpdir):
    pytest.importorskip("mlflow")
    import mlflow

    mldir = os.path.join(tmpdir, "mlruns")
    print(mldir)
    mlflow.set_tracking_uri("file:///" + mldir)

    from srl.runner.callbacks.mlflow_callback import MLFlowCallback

    rl_config = ql.Config()
    runner = srl.Runner("Grid", rl_config)

    runner.set_mlflow("test_Grid2")
    runner.train(max_episodes=100)

    # --- metric
    run_id = MLFlowCallback.get_run_id("test_Grid2", rl_name=rl_config.get_name())
    print(f"{run_id=}")
    assert run_id is not None
    metrics = MLFlowCallback.get_metric(run_id, "reward0")
    assert metrics is not None
    for metric in metrics:
        print(f"{metric.key=}")
        print(f"{metric.value=}")
        print(f"{metric.step=}")
        print(f"{metric.timestamp=}")


def test_method2(tmpdir):
    pytest.importorskip("mlflow")
    import mlflow

    mldir = os.path.join(tmpdir, "mlruns")
    print(mldir)
    mlflow.set_tracking_uri("file:///" + mldir)

    from srl.runner.callbacks.mlflow_callback import MLFlowCallback

    rl_config = ql.Config()
    runner = srl.Runner("Grid", rl_config)

    runner.set_mlflow("test_Grid2", run_name="AAA")
    runner.train(max_episodes=100)

    # --- metric
    run_id = MLFlowCallback.get_run_id("test_Grid2", run_name="AAA")
    print(f"{run_id=}")
    assert run_id is not None
    metrics = MLFlowCallback.get_metric(run_id, "reward0")
    assert metrics is not None
    for metric in metrics:
        print(f"{metric.key=}")
        print(f"{metric.value=}")
        print(f"{metric.step=}")
        print(f"{metric.timestamp=}")
