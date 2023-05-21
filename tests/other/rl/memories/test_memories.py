import collections
import math

import numpy as np
import pytest

from srl.rl.memories import config
from srl.rl.memories.proportional_memory import ProportionalMemory
from srl.rl.memories.rankbase_memory import RankBaseMemory
from srl.rl.memories.rankbase_memory_linear import RankBaseMemoryLinear

capacity = 10


@pytest.mark.parametrize(
    "memory_config, use_priority, check_dup",
    [
        (config.ReplayMemoryConfig(capacity), False, True),
        (config.ProportionalMemoryConfig(capacity, 0.8, 1, 10, has_duplicate=False), True, True),
        (config.RankBaseMemoryConfig(capacity, 0.8, 1, 10), True, True),
        (config.RankBaseMemoryLinearConfig(capacity, 0.8, 1, 10), True, True),
        (
            config.BestEpisodeMemoryConfig(
                memory=config.ProportionalMemoryConfig(capacity, 0.8, 1, 10),
            ),
            False,
            False,
        ),
        (config.DemoMemoryConfig(memory=config.ProportionalMemoryConfig(capacity, 0.8, 1, 10)), False, False),
    ],
)
def test_memory(memory_config, use_priority, check_dup):
    capacity = memory_config.get_capacity()
    assert capacity == 10
    memory = memory_config.create_memory()

    # add
    for i in range(100):
        memory.add((i, i, i, i), 0)
        memory.on_step(i, (i == 99))
    assert len(memory) == capacity

    # 中身を1～10にする
    for i in range(10):
        i += 1
        memory.add((i, i, i, i), i)
        assert len(memory) == capacity

    # --- 複数回やって比率をだす
    counter = []
    for i in range(10000):
        (indices, batchs, weights) = memory.sample(5, 1)
        assert len(batchs) == 5
        assert len(weights) == 5

        if check_dup:
            # 重複がないこと
            assert len(list(set(batchs))) == 5, list(set(batchs))

        # batchの中身をカウント
        for batch in batchs:
            counter.append(batch[0])

        # update priority
        memory.update(indices, batchs, [b[3] for b in batchs])
        assert len(memory) == capacity

        # save/load
        d = memory.backup()
        memory.restore(d)
        d2 = memory.backup()
        assert d == d2

    counter = collections.Counter(counter)

    if check_dup:
        # keyは1～10まであること
        keys = sorted(counter.keys())
        assert keys == [i + 1 for i in range(capacity)]
    else:
        keys = []

    if use_priority:
        # priorityが高いほど数が増えている
        vals = [counter[key] for key in keys]
        for i in range(capacity - 1):
            assert vals[i] < vals[i + 1]


# -------------------------------
# test IS
# -------------------------------
@pytest.mark.parametrize("alpha", [0, 0.2, 0.5, 0.8, 1.0])
def test_IS_Proportional(alpha):
    epsilon = 0.0001
    memory = ProportionalMemory(capacity=10, alpha=alpha, beta_initial=1, epsilon=epsilon, has_duplicate=False)

    # --- true data
    td_errors = [1, 2, 4, 3]

    # (|delta| + e)^a
    true_priorities = [(t + epsilon) ** alpha for t in td_errors]

    # --- check
    _check_weights(memory, td_errors, true_priorities)


@pytest.mark.parametrize("alpha", [0, 0.2, 0.5, 0.8, 1.0])
def test_IS_RankBase(alpha):
    memory = RankBaseMemory(capacity=10, alpha=alpha, beta_initial=1)

    # --- true data
    td_errors = [1, 2, 4, 3]
    rank = [4, 3, 1, 2]

    # 1 / rank(|delta|)^a
    true_priorities = [(1 / r) ** alpha for r in rank]

    # --- check
    _check_weights(memory, td_errors, true_priorities)


@pytest.mark.parametrize("alpha", [0.8])
def test_IS_RankBaseLinear(alpha):
    memory = RankBaseMemoryLinear(capacity=10, alpha=alpha, beta_initial=1)

    # --- true data
    td_errors = [1, 2, 4, 3]
    rank = [4, 3, 1, 2]

    # rank(|delta|)^a
    true_priorities = [1 + (4 - r) * alpha for r in rank]

    # --- check
    _check_weights(memory, td_errors, true_priorities)


def _check_weights(memory, td_errors, true_priorities):
    N = len(true_priorities)
    # print(sum(true_priorities))
    # print(true_priorities)

    # p^a / sum(p^a)
    sum_probs = sum(true_priorities)
    true_probs = [p / sum_probs for p in true_priorities]
    # print(true_probs)

    # (1/N)*(1/p)
    true_weights = np.array([(N * p) ** -1 for p in true_probs])
    # print(true_weights)

    # 最大値で正規化
    maxw = np.max(true_weights)
    true_weights /= maxw

    for i, td_error in enumerate(td_errors):
        memory.add((i, i, i, i), td_error=td_error)
    indices, batchs, weights = memory.sample(N, 1)

    # 順番が変わっているので batch より元のindexを取得し比較
    for i, b in enumerate(batchs):
        idx = b[0]
        assert math.isclose(weights[i], true_weights[idx], rel_tol=1e-7), f"{weights[i]} != {true_weights[idx]}"
