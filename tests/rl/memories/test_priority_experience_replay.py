from srl.rl.memories.experience_replay_buffer import ExperienceReplayBuffer, ExperienceReplayBufferConfig


def test_memory():
    capacity = 10

    config = ExperienceReplayBufferConfig()
    config.memory.capacity = capacity

    # ---

    memory = ExperienceReplayBuffer(config)
    assert memory.length() == 0

    # add
    for i in range(100):
        memory.add((i, i, i, i))
    assert memory.length() == capacity

    # sample
    batchs = memory.sample(5)
    assert len(batchs) == 5

    # sample over
    batchs = memory.sample(20)
    assert len(batchs) == 10

    # restore/backup
    dat = memory.backup(compress=True)
    memory2 = ExperienceReplayBuffer(config)
    memory2.restore(dat)
    assert memory2.length() == 10

    # restore over
    config.memory.capacity = 20
    memory2 = ExperienceReplayBuffer(config)
    memory2.restore(dat)
    assert memory2.length() == 10
    memory2.add((11, 11, 11, 11))
    assert memory2.length() == 11
    assert memory2.memory[10] == (11, 11, 11, 11)

    # restore min
    config.memory.capacity = 5
    memory2 = ExperienceReplayBuffer(config)
    memory2.restore(dat)
    assert memory2.length() == 5
    assert memory2.memory[4] == (99, 99, 99, 99)
    memory2.add((11, 11, 11, 11))
    assert memory2.length() == 5
    assert memory2.memory[0] == (11, 11, 11, 11)
