import random
import time

import numpy as np
from tqdm import tqdm

from srl.rl.memories.priority_memories.proportional_memory import ProportionalMemory
from srl.rl.memories.priority_memories.rankbased_memory import RankBasedMemory
from srl.rl.memories.priority_memories.rankbased_memory_linear import RankBasedMemoryLinear
from srl.rl.memories.priority_memories.replay_buffer import ReplayBuffer


def speed_test():
    capacity = 1_000_000

    memories = [
        ReplayBuffer(capacity),
        ProportionalMemory(capacity, 0.8, 0.4, 1000, has_duplicate=True),
        RankBasedMemory(capacity, 0.8, 0.4, 1000),
        RankBasedMemoryLinear(capacity, 0.8, 0.4, 1000),
    ]

    for memory in memories:
        _speed_test(memory)


def _speed_test(memory):
    t0 = time.time()

    warmup_size = 100_000
    batch_size = 64
    epochs = 5_000

    # warmup
    step = 0
    for _ in tqdm(range(warmup_size)):
        r = random.random()
        memory.add((step, step, step, step), r)
        step += 1

    for _ in tqdm(range(epochs)):
        # add
        r = random.random()
        memory.add((step, step, step, step), r)
        step += 1

        # sample
        (batches, weights, update_args) = memory.sample(batch_size, step)
        assert len(batches) == batch_size
        assert len(weights) == batch_size

        # update priority
        memory.update(update_args, [random.random() for _ in range(batch_size)])

    print("{}: {}s".format(memory.__class__.__name__, time.time() - t0))


def batch_speed_test():
    capacity = 1_000_000
    try_times = 10_000
    batch_size = 512

    buffer_arr = [((i, i, i), i, i, (i, i, i), True) for i in range(capacity)]
    buffer_dict = [
        {
            "states": (i, i, i),
            "actions": i,
            "rewards": i,
            "next_states": (i, i, i),
            "dones": True,
        }
        for i in range(capacity)
    ]

    # --- for array
    t0 = time.time()
    for _ in tqdm(range(try_times)):
        batches = random.sample(buffer_arr, batch_size)
        states, actions, rewards, next_states, dones = zip(*batches)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
    print("for array: {}s".format(time.time() - t0))

    # --- for asarray
    t0 = time.time()
    for _ in tqdm(range(try_times)):
        batches = random.sample(buffer_arr, batch_size)
        states, actions, rewards, next_states, dones = zip(*batches)
        states = np.asarray(states)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        next_states = np.asarray(next_states)
        dones = np.asarray(dones)
    print("for asarray: {}s".format(time.time() - t0))

    # --- for array one line
    t0 = time.time()
    for _ in tqdm(range(try_times)):
        batches = random.sample(buffer_arr, batch_size)
        states = np.array([b[0] for b in batches])
        actions = np.array([b[1] for b in batches])
        rewards = np.array([b[2] for b in batches])
        next_states = np.array([b[3] for b in batches])
        dones = np.array([b[4] for b in batches])
    print("for array one line: {}s".format(time.time() - t0))

    # --- for dict
    t0 = time.time()
    for _ in tqdm(range(try_times)):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for batch in random.sample(buffer_dict, batch_size):
            states.append(batch["states"])
            actions.append(batch["actions"])
            rewards.append(batch["rewards"])
            next_states.append(batch["next_states"])
            dones.append(batch["dones"])
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
    print("for dict: {}s".format(time.time() - t0))

    # --- for dict one line
    t0 = time.time()
    for _ in tqdm(range(try_times)):
        batches = random.sample(buffer_dict, batch_size)
        states = np.array([b["states"] for b in batches])
        actions = np.array([b["actions"] for b in batches])
        rewards = np.array([b["rewards"] for b in batches])
        next_states = np.array([b["next_states"] for b in batches])
        dones = np.array([b["dones"] for b in batches])
    print("for dict one line: {}s".format(time.time() - t0))


if __name__ == "__main__":
    speed_test()
    # batch_speed_test()
