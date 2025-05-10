import collections
import math

import numpy as np
import pytest

from srl.rl.memories.priority_memories.cpp_module import load_or_build_module
from srl.rl.memories.priority_memories.imemory import IPriorityMemory
from srl.rl.memories.priority_memories.proportional_memory import ProportionalMemory
from srl.rl.memories.priority_memories.rankbased_memory import RankBasedMemory
from srl.rl.memories.priority_memories.rankbased_memory_linear import RankBasedMemoryLinear
from srl.rl.memories.priority_memories.replay_buffer import ReplayBuffer

capacity = 10


@pytest.mark.parametrize(
    "memory_type, use_priority, check_dup",
    [
        ("ReplayBuffer", False, True),
        ("ProportionalMemory", True, True),
        ("ProportionalMemory", True, False),
        ("ProportionalMemory_cpp", True, True),
        ("ProportionalMemory_cpp", True, False),
        ("RankBasedMemory", True, True),
        ("RankBasedMemoryLinear", True, True),
    ],
)
def test_priority_memory(memory_type: str, use_priority: bool, check_dup: bool):
    if memory_type == "ReplayBuffer":
        memory = ReplayBuffer(capacity)
    elif memory_type == "ProportionalMemory":
        memory = ProportionalMemory(capacity, 0.8, 1, 10, has_duplicate=not check_dup)
    elif memory_type == "ProportionalMemory_cpp":
        pytest.importorskip("pybind11")
        proportional_memory_cpp = load_or_build_module("proportional_memory")
        memory = proportional_memory_cpp.ProportionalMemory(capacity, 0.8, 1, 10, has_duplicate=not check_dup)
    elif memory_type == "RankBasedMemory":
        memory = RankBasedMemory(capacity, 0.8, 1, 10)
    elif memory_type == "RankBasedMemoryLinear":
        memory = RankBasedMemoryLinear(capacity, 0.8, 1, 10)
    else:
        raise ValueError(f"Unknown memory type: {memory_type}")

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
        (batches, weights, update_args) = memory.sample(5, step=1)
        assert len(batches) == 5
        assert len(weights) == 5

        if check_dup:
            # 重複がないこと
            assert len(list(set(batches))) == 5, list(set(batches))

        # batchの中身をカウント
        for batch in batches:
            counter.append(batch[0])

        # update priority
        memory.update(update_args, np.array([b[3] for b in batches]))
        assert memory.length() == capacity

        # backup/restore
        l1 = memory.length()
        memory.restore(memory.backup())
        assert l1 == memory.length()

    counter = collections.Counter(counter)

    # keyは1～10まであること
    keys = sorted(counter.keys())
    if check_dup:
        assert keys == [i + 1 for i in range(capacity)]

    if use_priority:
        # priorityが高いほど数が増えている
        vals = [counter[key] for key in keys]
        for i in range(capacity - 1):
            assert vals[i] < vals[i + 1]


# -------------------------------
# test IS
# -------------------------------
@pytest.mark.parametrize("alpha", [0, 0.2, 0.5, 0.8, 1.0])
@pytest.mark.parametrize("memory_type", ["Proportional", "Proportional_cpp"])
def test_IS_Proportional(alpha, memory_type):
    epsilon = 0.0001
    if memory_type == "Proportional":
        memory = ProportionalMemory(capacity=10, alpha=alpha, beta_initial=1, epsilon=epsilon, has_duplicate=False)
    elif memory_type == "Proportional_cpp":
        pytest.importorskip("pybind11")
        proportional_memory_cpp = load_or_build_module("proportional_memory")
        memory = proportional_memory_cpp.ProportionalMemory(capacity=10, alpha=alpha, beta_initial=1, epsilon=epsilon, has_duplicate=False)
    else:
        raise ValueError(f"Unknown memory type: {memory_type}")

    # --- true data
    priorities = [1, 2, 4, 3]

    # (|delta| + e)^a
    true_priorities = [(t + epsilon) ** alpha for t in priorities]

    # --- check
    _check_weights(memory, priorities, true_priorities)


@pytest.mark.parametrize("alpha", [0, 0.2, 0.5, 0.8, 1.0])
def test_IS_RankBased(alpha):
    memory = RankBasedMemory(capacity=10, alpha=alpha, beta_initial=1)

    # --- true data
    td_errors = [1, 2, 4, 3]
    rank = [4, 3, 1, 2]

    # 1 / rank(|delta|)^a
    true_priorities = [(1 / r) ** alpha for r in rank]

    # --- check
    _check_weights(memory, td_errors, true_priorities)


@pytest.mark.parametrize("alpha", [0.8])
def test_IS_RankBasedLinear(alpha):
    memory = RankBasedMemoryLinear(capacity=10, alpha=alpha, beta_initial=1)

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
    batches, weights, update_args = memory.sample(N, step=1)

    # 順番が変わっているので batch より元のindexを取得し比較
    for i, b in enumerate(batches):
        idx = b[0]
        assert math.isclose(weights[i], true_weights[idx], rel_tol=1e-7), f"{weights[i]} != {true_weights[idx]}"
