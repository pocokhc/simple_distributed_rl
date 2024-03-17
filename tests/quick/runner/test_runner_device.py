import os

import pytest

import srl
from srl.algorithms import dqn, ql
from srl.base.run.context import RunNameTypes
from srl.utils import common


# -----------------------------------
# get_device
# -----------------------------------
@pytest.mark.parametrize(
    "run_name, device, true_device",
    [
        [RunNameTypes.main, "AUTo", "AUTO"],
        [RunNameTypes.main, "gpu:1", "GPU:1"],
        [RunNameTypes.trainer, "", "AUTO"],
        [RunNameTypes.eval, "", "CPU"],
    ],
)
def test_get_device(run_name, device, true_device):
    runner = srl.Runner("Grid", ql.Config())
    runner.config.device = device
    get_device = runner.get_device(run_name, 1)
    assert get_device == true_device


@pytest.mark.parametrize(
    "device, actor_id, true_device",
    [
        ["AUTo", 0, "CPU"],
        ["GPu", 1, "GPU"],
        [["", ""], 1, "CPU"],
        [["", "AUTO"], 1, "CPU"],
        [["", "GPu"], 1, "GPU"],
        [["CPU:0", "Cpu:1"], 1, "CPU:1"],
    ],
)
def test_get_device_actor(device, actor_id, true_device):
    runner = srl.Runner("Grid", ql.Config())
    runner.config.device_actors = device
    get_device = runner.get_device(RunNameTypes.actor, actor_id)
    assert get_device == true_device


# -----------------------------------
# setup
# -----------------------------------
def _setup_device(
    framework,
    device,
    true_tf="/CPU",
    true_torch="cpu",
):
    if framework == "tensorflow":
        pytest.importorskip("tensorflow")
        rl_config = dqn.Config()
        rl_config.framework.set_tensorflow()
        runner = srl.Runner("Grid", rl_config)
    elif framework == "torch":
        pytest.importorskip("torch")
        rl_config = dqn.Config()
        rl_config.framework.set_torch()
        runner = srl.Runner("Grid", rl_config)
    else:
        runner = srl.Runner("Grid", ql.Config())

    used_device_tf, used_device_torch = runner.setup_device(framework, device)
    used_device_tf, used_device_torch = runner.setup_device(framework, device)
    assert used_device_tf == true_tf
    assert used_device_torch == true_torch

    runner.train(max_train_count=10)


@pytest.mark.parametrize("device", ["AUTO", "CPU", "GPU", "CPU:0", "GPU:0"])
def test_setup_device_no_framework(device):
    _setup_device(
        framework="",
        device=device,
        true_tf="/CPU",
        true_torch="cpu",
    )
