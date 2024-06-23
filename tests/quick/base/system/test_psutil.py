import numpy as np
import pytest

from srl.base.system.psutil_ import read_cpu, read_memory


def test_read_memory():
    pytest.importorskip("psutil")

    for _ in range(5):
        memory_percent = read_memory()
        print(memory_percent)
        assert not np.isnan(memory_percent)


def test_read_cpu():
    pytest.importorskip("psutil")

    for _ in range(5):
        cpu_percent = read_cpu()
        print(cpu_percent)
        assert not np.isnan(cpu_percent)
