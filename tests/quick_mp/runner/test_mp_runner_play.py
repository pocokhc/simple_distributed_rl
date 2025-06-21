import pytest

import srl
from srl.algorithms import ql


@pytest.mark.parametrize("enable_mp_memory", [False, True])
def test_train_multi_runner(enable_mp_memory):
    runner1 = srl.Runner("Grid", ql.Config())
    runner2 = srl.Runner("OX", ql.Config())
    runner3 = srl.Runner("OX", ql.Config())

    runner1.train_mp(max_train_count=10, enable_mp_memory=enable_mp_memory)
    runner1.train(max_train_count=10)

    runner3.train(max_train_count=10)
    runner3.train_mp(max_train_count=10, enable_mp_memory=enable_mp_memory)
    runner2.train(max_train_count=10)
    runner1.train(max_train_count=10)
