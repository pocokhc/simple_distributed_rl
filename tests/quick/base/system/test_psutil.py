import numpy as np
import pytest

from srl.base.system import psutil_


def test_read_system_memory_percent():
    pytest.importorskip("psutil")

    for _ in range(5):
        memory_percent = psutil_.read_system_memory_percent()
        print(memory_percent)
        assert not np.isnan(memory_percent)


def test_read_process_memory_rss():
    pytest.importorskip("psutil")

    for _ in range(5):
        memory_percent = psutil_.read_process_memory_rss()
        print(memory_percent)
        assert not np.isnan(memory_percent)


def test_read_cpu():
    pytest.importorskip("psutil")

    for _ in range(5):
        cpu_percent = psutil_.read_cpu()
        print(cpu_percent)
        assert not np.isnan(cpu_percent)
