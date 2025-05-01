import os

import numpy as np
import pytest


def test_single_use_buffer():
    from srl.rl.memories.single_use_buffer import SingleUseBuffer

    memory = SingleUseBuffer()
    assert memory.length() == 0

    memory.add((1, "A", [2, 2, 2]))
    memory.add((2, "B", [3, 3, 3]))
    assert memory.length() == 2

    memory.call_restore(memory.call_backup())
    assert memory.length() == 2

    memory.call_restore(memory.call_backup())
    assert memory.length() == 2

    batches = memory.sample()
    assert memory.length() == 0
    assert len(batches) == 2
    assert batches[0][0] == 1


def _play_memory_sub(
    memory,
    capacity: int,
    warmup_size: int,
    batch_size: int,
    is_priority: bool,
):
    assert memory.length() == 0
    assert warmup_size <= capacity

    # --- warmup前のsampleはNone
    batches = memory.sample()
    assert batches is None

    # --- warmup
    assert capacity < 100
    for i in range(100):
        memory.add((i, i, i, i))
    assert memory.length() == capacity

    memory.call_restore(memory.call_backup())
    assert memory.length() == capacity
    assert memory.length() > warmup_size

    # --- loop
    for i in range(100):
        if not is_priority:
            batches = memory.sample()
            assert len(batches) == batch_size
            assert memory.length() == capacity
        else:
            # sampleとupdateがずれても問題ないようにする
            batches1, weights1, update_args1 = memory.sample(i)
            assert len(batches1) == batch_size
            assert len(weights1) == batch_size
            assert isinstance(weights1, np.ndarray)
            assert weights1.dtype == np.float32

            batches2, weights2, update_args2 = memory.sample(i)
            assert len(batches2) == batch_size
            assert len(weights2) == batch_size

            memory.update(update_args1, np.array([b[3] for b in batches1]), i)

            batches3, weights3, update_args3 = memory.sample(i)
            assert len(batches3) == batch_size
            assert len(weights3) == batch_size

            memory.update(update_args2, np.array([b[3] for b in batches2]), i)
            memory.update(update_args3, np.array([b[3] for b in batches3]), i)
            assert memory.length() == capacity


@pytest.mark.parametrize("compress", [False, True])
def test_replay_buffer(compress):
    from srl.rl.memories.replay_buffer import ReplayBuffer, ReplayBufferConfig

    capacity = 10
    warmup_size = 5
    batch_size = 5

    memory = ReplayBuffer(ReplayBufferConfig(capacity, warmup_size, compress), batch_size)
    _play_memory_sub(memory, capacity, warmup_size, batch_size, is_priority=False)


@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize(
    "memory_type",
    [
        "ReplayBuffer",
        "Proportional",
        "Proportional_cpp",
        "RankBased",
        "RankBasedLinear",
    ],
)
def test_priority_memory(compress, memory_type):
    from srl.rl.memories.priority_replay_buffer import PriorityReplayBuffer, PriorityReplayBufferConfig

    capacity = 10
    warmup_size = 5
    batch_size = 5

    cfg = PriorityReplayBufferConfig(capacity, warmup_size, compress)
    if memory_type == "ReplayBuffer":
        cfg.set_replay_buffer()
    elif memory_type == "Proportional":
        cfg.set_proportional()
    elif memory_type == "Proportional_cpp":
        if os.environ.get("TEST_TYPE", "") == "low":
            pytest.skip("TEST_TYPE is test")
        cfg.set_proportional_cpp(force_build=True)
    elif memory_type == "RankBased":
        cfg.set_rankbased()
    elif memory_type == "RankBasedLinear":
        cfg.set_rankbased_linear()
    else:
        raise ValueError(f"Unknown memory type: {memory_type}")

    memory = PriorityReplayBuffer(cfg, batch_size)
    _play_memory_sub(memory, capacity, warmup_size, batch_size, is_priority=True)
