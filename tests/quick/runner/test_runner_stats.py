import numpy as np
import pytest

import srl
from srl.algorithms import ql


def test_psutil():
    pytest.importorskip("psutil")

    runner = srl.Runner("Grid", ql.Config())

    for _ in range(2):
        runner.context.max_steps = 10

        memory_percent, cpu_percent = runner.read_psutil()
        print(memory_percent, cpu_percent)
        assert not np.isnan(memory_percent)
        assert not np.isnan(cpu_percent)
