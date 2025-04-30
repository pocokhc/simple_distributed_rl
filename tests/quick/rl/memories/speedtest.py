import random
import time

from tqdm import tqdm

from srl.rl.memories.priority_memories.bin import load_native_module
from srl.rl.memories.priority_memories.proportional_memory import ProportionalMemory
from srl.rl.memories.priority_memories.rankbased_memory import RankBasedMemory
from srl.rl.memories.priority_memories.rankbased_memory_linear import RankBasedMemoryLinear
from srl.rl.memories.priority_memories.replay_buffer import ReplayBuffer

proportional_memory_cpp = load_native_module("proportional_memory_cpp")


def speed_test():
    capacity = 1_000_000

    memories = [
        ReplayBuffer(capacity),
        ProportionalMemory(capacity, 0.8, 0.4, 1000, has_duplicate=True),
        proportional_memory_cpp.ProportionalMemory(capacity, 0.8, 0.4, 1000, has_duplicate=True),
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


if __name__ == "__main__":
    speed_test()
