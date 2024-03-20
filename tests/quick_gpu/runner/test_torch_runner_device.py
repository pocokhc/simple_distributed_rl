import os

import pytest

from srl.utils import common
from tests.quick_gpu.runner.test_runner_device import _setup_device


@pytest.mark.parametrize(
    "device, true_tf, true_torch",
    [
        ["AUTO", "/CPU", "cpu"],
        ["CPU", "/CPU", "cpu"],
        ["CPU:1", "/CPU:1", "cpu:1"],
        ["GPU", "/CPU", "cpu"],
        ["GPU:1", "/CPU", "cpu"],
    ],
)
def test_setup_device_torch_cpu(device, true_tf, true_torch):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-2"  # change CPU
    _setup_device(
        framework="torch",
        device=device,
        true_tf=true_tf,
        true_torch=true_torch,
    )


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
    if not common.is_available_gpu_torch():
        pytest.skip()
    _setup_device(
        framework="torch",
        device=device,
        true_tf=true_tf,
        true_torch=true_torch,
    )
