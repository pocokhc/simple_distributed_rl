import pytest

import srl
from srl.algorithms import dqn, ql
from srl.base.system import device as device_module
from srl.base.system.device import setup_device
from srl.utils.common import is_available_gpu_tf, is_available_gpu_torch


# -----------------------------------
# setup
# -----------------------------------
def _setup_device(
    framework,
    device,
    true_tf="/CPU",
    true_torch="cpu",
):
    device_module.__setup_device = False
    device_module.__framework = ""
    device_module.__used_device_tf = "/CPU"
    device_module.__used_device_torch = "cpu"

    if framework == "tensorflow":
        pytest.importorskip("tensorflow")
        rl_config = dqn.Config()
        rl_config.set_tensorflow()
        runner = srl.Runner("Grid", rl_config)
    elif framework == "torch":
        pytest.importorskip("torch")
        rl_config = dqn.Config()
        rl_config.set_torch()
        runner = srl.Runner("Grid", rl_config)
    else:
        runner = srl.Runner("Grid", ql.Config())

    used_device_tf, used_device_torch = setup_device(framework, device)
    used_device_tf, used_device_torch = setup_device(framework, device)
    if framework == "tensorflow":
        assert used_device_tf == true_tf
    if framework == "torch":
        assert used_device_torch == true_torch

    if true_torch != "cuda:1":
        runner.train(max_train_count=2)


# -----------------------------------
# tensorflow
# -----------------------------------
@pytest.mark.parametrize(
    "device, true_tf, true_torch",
    [
        ["AUTO", "/GPU", "cpu"],
        ["CPU", "/CPU", "cpu"],
        ["CPU:1", "/CPU:1", "cpu:1"],
        ["GPU", "/GPU", "cpu"],
        ["GPU:1", "/GPU:1", "cpu"],
    ],
)
def test_setup_device_tf_gpu(device, true_tf, true_torch):
    if not is_available_gpu_tf():
        pytest.skip()
    _setup_device(
        framework="tensorflow",
        device=device,
        true_tf=true_tf,
        true_torch=true_torch,
    )


# -----------------------------------
# torch
# -----------------------------------
@pytest.mark.parametrize(
    "device, true_tf, true_torch",
    [
        ["AUTO", "/CPU", "cuda"],
        ["CPU", "/CPU", "cpu"],
        ["CPU:1", "/CPU:1", "cpu:1"],
        ["GPU", "/CPU", "cuda"],
        ["GPU:1", "/CPU", "cuda:1"],
    ],
)
def test_setup_device_torch_gpu(device, true_tf, true_torch):
    if not is_available_gpu_torch():
        pytest.skip()
    _setup_device(
        framework="torch",
        device=device,
        true_tf=true_tf,
        true_torch=true_torch,
    )
