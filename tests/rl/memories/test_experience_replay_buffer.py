import numpy as np

from srl.rl.memories.priority_experience_replay import PriorityExperienceReplayConfig
from srl.rl.memories.priority_memories.best_episode_memory import BestEpisodeMemory
from srl.rl.memories.priority_memories.demo_memory import DemoMemory
from srl.rl.memories.priority_memories.imemory import IPriorityMemory
from srl.rl.memories.priority_memories.proportional_memory import ProportionalMemory
from srl.rl.memories.priority_memories.rankbase_memory import RankBaseMemory
from srl.rl.memories.priority_memories.rankbase_memory_linear import RankBaseMemoryLinear
from srl.rl.memories.priority_memories.replay_memory import ReplayMemory


def _play_memory(memory: IPriorityMemory, capacity: int):
    assert len(memory) == 0

    # add
    for i in range(100):
        memory.add((i, i, i, i))
        memory.on_step(i, (i == 99))
    assert len(memory) == capacity

    # loop
    for i in range(100):
        # sample
        (indices, batchs, weights) = memory.sample(5, i)
        assert len(batchs) == 5
        assert len(weights) == 5

        # update
        memory.update(indices, batchs, np.array([b[3] for b in batchs]))
        assert len(memory) == capacity


def test_replay_memory():
    capacity = 10

    conf = PriorityExperienceReplayConfig()
    conf.memory.capacity = capacity
    conf.memory.set_replay_memory()

    memory = conf.memory.create_memory()
    assert isinstance(memory, ReplayMemory)

    _play_memory(memory, capacity)


def test_proportional_memory():
    capacity = 10

    conf = PriorityExperienceReplayConfig()
    conf.memory.capacity = capacity
    conf.memory.set_proportional_memory()

    memory = conf.memory.create_memory()
    assert isinstance(memory, ProportionalMemory)

    _play_memory(memory, capacity)


def test_rankbase_memory():
    capacity = 10

    conf = PriorityExperienceReplayConfig()
    conf.memory.capacity = capacity
    conf.memory.set_rankbase_memory()

    memory = conf.memory.create_memory()
    assert isinstance(memory, RankBaseMemory)

    _play_memory(memory, capacity)


def test_rankbase_memory_linear():
    capacity = 10

    conf = PriorityExperienceReplayConfig()
    conf.memory.capacity = capacity
    conf.memory.set_rankbase_memory_linear()

    memory = conf.memory.create_memory()
    assert isinstance(memory, RankBaseMemoryLinear)

    _play_memory(memory, capacity)


def test_best_episode_memory():
    capacity = 10

    conf = PriorityExperienceReplayConfig()
    conf.memory.capacity = capacity
    conf.memory.set_proportional_memory()
    conf.memory.set_best_episode_memory()
    assert conf.memory.best_episode_memory is not None

    memory = conf.memory.create_memory()
    assert isinstance(memory, BestEpisodeMemory)

    _play_memory(memory, capacity)


def test_demo_memory():
    capacity = 10

    conf = PriorityExperienceReplayConfig()
    conf.memory.capacity = capacity
    conf.memory.set_proportional_memory()
    conf.memory.set_demo_memory()
    assert conf.memory.demo_memory is not None

    memory = conf.memory.create_memory()
    assert isinstance(memory, DemoMemory)

    _play_memory(memory, capacity)
