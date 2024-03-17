import os

import pytest

import srl
from srl.algorithms import dqn, ql
from srl.base.run.context import RunNameTypes
from srl.utils import common


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
def test_setup_device_tf_cpu(device, true_tf, true_torch):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-2"  # change CPU
    _setup_device(
        framework="tensorflow",
        device=device,
        true_tf=true_tf,
        true_torch=true_torch,
    )


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
    if not common.is_available_gpu_tf():
        pytest.skip()
    _setup_device(
        framework="tensorflow",
        device=device,
        true_tf=true_tf,
        true_torch=true_torch,
    )
