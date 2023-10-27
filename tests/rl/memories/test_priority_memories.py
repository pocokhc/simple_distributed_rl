import collections
import math

import numpy as np
import pytest

from srl.rl.memories.priority_memories.imemory import IPriorityMemory
from srl.rl.memories.priority_memories.proportional_memory import ProportionalMemory
from srl.rl.memories.priority_memories.rankbase_memory import RankBaseMemory
from srl.rl.memories.priority_memories.rankbase_memory_linear import RankBaseMemoryLinear
from srl.rl.memories.priority_memories.replay_memory import ReplayMemory

capacity = 10


@pytest.mark.parametrize(
    "memory, use_priority, check_dup",
    [
        (ReplayMemory(capacity), False, True),
        (ProportionalMemory(capacity, 0.8, 1, 10, has_duplicate=False), True, True),
        (RankBaseMemory(capacity, 0.8, 1, 10), True, True),
        (RankBaseMemoryLinear(capacity, 0.8, 1, 10), True, True),
    ],
)
def test_priority_memory(memory: IPriorityMemory, use_priority: bool, check_dup: bool):
    # add
    for i in range(100):
        memory.add((i, i, i, i), 0)
    assert memory.length() == capacity

    # 中身を1～10にする
    for i in range(10):
        i += 1
        memory.add((i, i, i, i), i)
        assert memory.length() == capacity

    # --- 複数回やって比率をだす
    counter = []
    for i in range(10000):
        (indices, batchs, weights) = memory.sample(5, step=1)
        assert len(batchs) == 5
        assert len(weights) == 5

        if check_dup:
            # 重複がないこと
            assert len(list(set(batchs))) == 5, list(set(batchs))

        # batchの中身をカウント
        for batch in batchs:
            counter.append(batch[0])

        # update priority
        memory.update(indices, batchs, np.array([b[3] for b in batchs]))
        assert memory.length() == capacity

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
    priorities = [1, 2, 4, 3]

    # (|delta| + e)^a
    true_priorities = [(t + epsilon) ** alpha for t in priorities]

    # --- check
    _check_weights(memory, priorities, true_priorities)


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


def _check_weights(memory: IPriorityMemory, priorities, true_priorities):
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

    for i, priority in enumerate(priorities):
        assert priority >= 0
        memory.add((i, i, i, i), priority=priority)
    indices, batchs, weights = memory.sample(N, step=1)

    # 順番が変わっているので batch より元のindexを取得し比較
    for i, b in enumerate(batchs):
        idx = b[0]
        assert math.isclose(weights[i], true_weights[idx], rel_tol=1e-7), f"{weights[i]} != {true_weights[idx]}"
