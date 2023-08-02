import os

import numpy as np
import pytest

from srl.algorithms import ql
from srl.runner.runner import Runner
from srl.utils import common

common.logger_print()


def test_psutil():
    pytest.importorskip("psutil")

    runner = Runner("Grid", ql.Config())

    runner.context.max_steps = 10
    runner.context.init(runner)
    assert runner.context.used_psutil

    memory_percent, cpu_percent = runner.read_psutil()
    assert not np.isnan(memory_percent)
    assert not np.isnan(cpu_percent)

    runner.close_nvidia()
    runner.close_nvidia()


def test_pynvml():
    pytest.importorskip("pynvml")
    if common.is_available_gpu_tf():
        pass
    elif common.is_available_gpu_torch():
        pass
    else:
        pytest.skip()

    runner = Runner("Grid", ql.Config())

    runner._init_process()
    assert runner.context.used_nvidia

    gpus = runner.read_nvml()
    for device_id, gpu, memory in gpus:
        print(device_id, gpu, memory)

    runner.close_nvidia()
    runner.close_nvidia()
