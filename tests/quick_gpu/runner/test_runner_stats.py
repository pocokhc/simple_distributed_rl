import pytest

import srl
from srl.algorithms import ql
from srl.utils import common


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
        runner._setup_nvidia()

        gpus = runner.read_nvml()
        assert len(gpus) > 0
        for device_id, gpu, memory in gpus:
            print(device_id, gpu, memory)

    runner.close_nvidia()
    runner.close_nvidia()
