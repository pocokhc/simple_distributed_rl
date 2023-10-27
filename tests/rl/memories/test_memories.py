import numpy as np

from srl.base.rl.base import RLMemory
from srl.rl.memories.experience_replay_buffer import ExperienceReplayBuffer, ExperienceReplayBufferConfig
from srl.rl.memories.priority_experience_replay import PriorityExperienceReplay, PriorityExperienceReplayConfig
from srl.rl.memories.sequence_memory import SequenceMemory


def test_sequence_memory():
    memory = SequenceMemory(None)
    assert memory.length() == 0
    assert memory.is_warmup_needed()

    memory.add((1, "A", [2, 2, 2]))
    memory.add((2, "B", [3, 3, 3]))
    assert memory.length() == 2
    assert not memory.is_warmup_needed()

    memory.restore(memory.backup(compress=False))
    assert memory.length() == 2
    assert not memory.is_warmup_needed()

    memory.restore(memory.backup(compress=True))
    assert memory.length() == 2
    assert not memory.is_warmup_needed()

    batchs = memory.sample()
    assert memory.length() == 0
    assert memory.is_warmup_needed()
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
    assert memory.is_warmup_needed()
    for i in range(100):
        memory.add((i, i, i, i))
    assert memory.length() == capacity
    assert not memory.is_warmup_needed()

    # --- サイズ以上をsampleした場合の動作は未定義
    # with pytest.raises(ValueError) as e:
    #    memory.sample(0, batch_size=200)

    memory.restore(memory.backup(compress=True))
    assert memory.length() == capacity
    assert not memory.is_warmup_needed()

    # --- loop
    for i in range(100):
        if not is_priority:
            batchs = memory.sample(batch_size, i)
            assert len(batchs) == 5
            assert memory.length() == capacity
        else:
            (indices, batchs, weights) = memory.sample(batch_size, i)
            assert len(batchs) == 5
            assert len(weights) == 5

            memory.update((indices, batchs, np.array([b[3] for b in batchs])))
            assert memory.length() == capacity
            assert not memory.is_warmup_needed()


def test_experience_replay_buffer():
    capacity = 10
    warmup_size = 5
    batch_size = 5

    conf = ExperienceReplayBufferConfig()
    conf.memory.capacity = capacity
    conf.memory.warmup_size = warmup_size
    conf.batch_size = batch_size

    memory = ExperienceReplayBuffer(conf)
    _play_memory_sub(memory, capacity, warmup_size, batch_size, is_priority=False)


def _play_priority_memories(conf: PriorityExperienceReplayConfig):
    capacity = 10
    warmup_size = 5
    batch_size = 5

    conf.memory.capacity = capacity
    conf.memory.warmup_size = warmup_size
    conf.batch_size = batch_size

    memory = PriorityExperienceReplay(conf)
    _play_memory_sub(memory, capacity, warmup_size, batch_size, is_priority=True)


def test_replay_memory():
    conf = PriorityExperienceReplayConfig()
    conf.memory.set_replay_memory()
    _play_priority_memories(conf)


def test_proportional_memory():
    conf = PriorityExperienceReplayConfig()
    conf.memory.set_proportional_memory()
    _play_priority_memories(conf)


def test_rankbase_memory():
    conf = PriorityExperienceReplayConfig()
    conf.memory.set_rankbase_memory()
    _play_priority_memories(conf)


def test_rankbase_memory_linear():
    conf = PriorityExperienceReplayConfig()
    conf.memory.set_rankbase_memory_linear()
    _play_priority_memories(conf)
