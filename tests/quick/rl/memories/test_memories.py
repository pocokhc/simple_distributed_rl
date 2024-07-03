from dataclasses import dataclass

import numpy as np

from srl.base.rl.config import DummyRLConfig
from srl.base.rl.memory import RLMemory
from srl.rl.memories.experience_replay_buffer import ExperienceReplayBuffer, RLConfigComponentExperienceReplayBuffer
from srl.rl.memories.priority_experience_replay import (
    PriorityExperienceReplay,
    RLConfigComponentPriorityExperienceReplay,
)
from srl.rl.memories.sequence_memory import SequenceMemory


def test_sequence_memory():
    memory = SequenceMemory(DummyRLConfig())
    assert memory.length() == 0

    memory.add((1, "A", [2, 2, 2]))
    memory.add((2, "B", [3, 3, 3]))
    assert memory.length() == 2

    memory.restore(memory.backup(compress=False))
    assert memory.length() == 2

    memory.restore(memory.backup(compress=True))
    assert memory.length() == 2

    batchs = memory.sample()
    assert memory.length() == 0
    assert len(batchs) == 2
    assert batchs[0][0] == 1


def _play_memory_sub(
    memory: RLMemory,
    capacity: int,
    warmup_size: int,
    batch_size: int,
    is_priority: bool,
):
    assert memory.length() == 0
    assert warmup_size <= capacity

    # --- warmup
    for i in range(100):
        memory.add((i, i, i, i))
    assert memory.length() == capacity

    # --- サイズ以上をsampleした場合の動作は未定義
    # with pytest.raises(ValueError) as e:
    #    memory.sample(0, batch_size=200)

    memory.restore(memory.backup(compress=True))
    assert memory.length() == capacity

    # --- loop
    for i in range(100):
        if not is_priority:
            batchs = memory.sample()
            assert len(batchs) == 5
            assert memory.length() == capacity
        else:
            batchs, weights, update_args = memory.sample(i)
            assert len(batchs) == 5
            assert len(weights) == 5

            memory.update(update_args, np.array([b[3] for b in batchs]))
            assert memory.length() == capacity


def test_experience_replay_buffer():
    capacity = 10
    warmup_size = 5
    batch_size = 5

    @dataclass
    class C(DummyRLConfig, RLConfigComponentExperienceReplayBuffer):
        pass

    conf = C()
    conf.memory_capacity = capacity
    conf.memory_warmup_size = warmup_size
    conf.batch_size = batch_size

    memory = ExperienceReplayBuffer(conf)
    _play_memory_sub(memory, capacity, warmup_size, batch_size, is_priority=False)


def _play_priority_memories(conf: RLConfigComponentPriorityExperienceReplay):
    capacity = 10
    warmup_size = 5
    batch_size = 5

    conf.memory_capacity = capacity
    conf.memory_warmup_size = warmup_size
    conf.batch_size = batch_size

    memory = PriorityExperienceReplay(conf)
    _play_memory_sub(memory, capacity, warmup_size, batch_size, is_priority=True)


@dataclass
class _C_PER(DummyRLConfig, RLConfigComponentPriorityExperienceReplay):
    pass


def test_replay_memory():
    conf = _C_PER()
    conf.set_replay_memory()
    _play_priority_memories(conf)


def test_proportional_memory():
    conf = _C_PER()
    conf.set_proportional_memory()
    _play_priority_memories(conf)


def test_rankbase_memory():
    conf = _C_PER()
    conf.set_rankbase_memory()
    _play_priority_memories(conf)


def test_rankbase_memory_linear():
    conf = _C_PER()
    conf.set_rankbase_memory_linear()
    _play_priority_memories(conf)
