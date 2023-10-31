import numpy as np
import pytest

import srl
from srl.algorithms import ql
from srl.utils import common


def test_psutil():
    pytest.importorskip("psutil")

    runner = srl.Runner("Grid", ql.Config())

    memory_percent, cpu_percent = runner.read_psutil()
    assert np.isnan(memory_percent)

    for _ in range(2):
        runner.context.max_steps = 10
        runner.setup_psutil()
        assert runner._psutil_process is not None
        assert runner._psutil_process

        memory_percent, cpu_percent = runner.read_psutil()
        print(memory_percent, cpu_percent)
        assert not np.isnan(memory_percent)
        assert not np.isnan(cpu_percent)


def test_pynvml():
    pytest.importorskip("pynvml")
    if common.is_available_gpu_tf():
        pass
    elif common.is_available_gpu_torch():
        pass
    else:
        pytest.skip()

    runner = srl.Runner("Grid", ql.Config())
    gpus = runner.read_nvml()
    assert len(gpus) == 0

    for _ in range(2):
        runner = srl.Runner("Grid", ql.Config())
        runner.setup_nvidia()

        gpus = runner.read_nvml()
        assert len(gpus) > 0
        for device_id, gpu, memory in gpus:
            print(device_id, gpu, memory)

    runner.close_nvidia()
    runner.close_nvidia()
