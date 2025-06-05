import pytest

from srl.rl.memories.episode_replay_buffer import EpisodeReplayBuffer, EpisodeReplayBufferConfig


@pytest.mark.parametrize("compress", [False, True])
def test_episode_replay_buffer(compress):
    capacity = 100
    warmup_size = 5
    batch_size = 5
    prefix_size = 1
    suffix_size = 2
    skip_head = 3
    skip_tail = 4

    memory = EpisodeReplayBuffer(
        EpisodeReplayBufferConfig(capacity, warmup_size, compress),
        batch_size,
        prefix_size,
        suffix_size,
        skip_head,
        skip_tail,
    )
    assert memory.length() == 0
    assert memory.batch_length == prefix_size + 1 + suffix_size

    # --- warmup前のsampleはNone
    batches = memory.sample()
    assert batches is None

    # --- warmup
    memory.add([{"obs": i, "action": i * 2} for i in range(15)])
    assert memory.length() == 15 - skip_head - skip_tail - (prefix_size + suffix_size)
    for i in range(10):
        memory.add([{"obs": i, "action": i * 2} for i in range(20)])
    print(memory.length())
    assert memory.length() <= capacity

    memory.call_restore(memory.call_backup())
    assert memory.length() == capacity
    assert memory.length() > warmup_size

    # --- loop
    dat = []
    for i in range(1000):
        batches = memory.sample()
        assert batches is not None
        assert len(batches) == batch_size
        assert memory.length() == capacity
        for b in batches:
            assert len(b) == prefix_size + 1 + suffix_size
            dat += [b2["obs"] for b2 in b]
    dat = list(sorted(set(dat)))
    true_data = list(range(skip_head, 20 - skip_tail))
    print(dat)
    print(true_data)
    assert dat == true_data


@pytest.mark.parametrize("compress", [False, True])
def test_episode_sequential_replay_buffer(compress):
    capacity = 100
    warmup_size = 5
    batch_size = 2
    prefix_size = 1
    suffix_size = 2
    skip_head = 3
    skip_tail = 4

    memory = EpisodeReplayBuffer(
        EpisodeReplayBufferConfig(capacity, warmup_size, compress),
        batch_size,
        prefix_size,
        suffix_size,
        skip_head,
        skip_tail,
    )

    # --- warmup
    for i in range(100):
        memory.add([{"obs": i, "action": i * 2} for i in range(12)])
    print(memory.length())
    assert memory.length() <= capacity

    memory.call_restore(memory.call_backup())
    assert memory.length() == capacity
    assert memory.length() > warmup_size

    # --- 1step
    batches = memory.sample_sequential()
    assert batches is not None
    assert len(batches) == batch_size
    assert memory.length() == capacity
    for b in batches:
        print(b)
        assert len(b) == prefix_size + 1 + suffix_size
        assert b[0]["obs"] == 3
        assert b[1]["obs"] == 4
        assert b[2]["obs"] == 5
        assert b[3]["obs"] == 6

    # --- 2step
    batches = memory.sample_sequential()
    assert batches is not None
    assert len(batches) == batch_size
    assert memory.length() == capacity
    for b in batches:
        print(b)
        assert len(b) == prefix_size + 1 + suffix_size
        assert b[0]["obs"] == 7
        assert b[1]["obs"] == 3
        assert b[2]["obs"] == 4
        assert b[3]["obs"] == 5
