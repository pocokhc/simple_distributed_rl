import numpy as np
import pytest

import srl
from srl.algorithms import ql
from srl.utils import common


def test_psutil():
    pytest.importorskip("psutil")

    runner = srl.Runner("Grid", ql.Config())

    for _ in range(2):
        runner.context.max_steps = 10
        runner._init_process()
        assert runner.config.used_psutil

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

    runner = srl.Runner("Grid", ql.Config())

    runner._init_process()
    assert runner.config.used_nvidia

    gpus = runner.read_nvml()
    for device_id, gpu, memory in gpus:
        print(device_id, gpu, memory)

    runner.close_nvidia()
    runner.close_nvidia()
