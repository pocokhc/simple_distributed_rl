import pytest

from srl.base.system.pynvml_ import close_nvidia, read_nvml
from srl.utils import common


def test_pynvml():
    pytest.importorskip("pynvml")
    if common.is_available_gpu_tf():
        pass
    elif common.is_available_gpu_torch():
        pass
    else:
        pytest.skip()

    for _ in range(5):
        gpus = read_nvml()
        assert len(gpus) > 0
        for device_id, gpu, memory in gpus:
            print(device_id, gpu, memory)

    close_nvidia()
    close_nvidia()
