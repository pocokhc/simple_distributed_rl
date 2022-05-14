import random
import time

from srl.rl.memory.proportional_memory import ProportionalMemory
from srl.rl.memory.rankbase_memory import RankBaseMemory
from srl.rl.memory.replay_memory import ReplayMemory
from tqdm import tqdm


def speed_test():
    capacity = 100_000_000

    memories = [
        ReplayMemory(capacity),
        ProportionalMemory(capacity, 0.8, 1, 10),
        RankBaseMemory(capacity, 0.8, 1, 10),
    ]

    for memory in memories:
        _speed_test(memory)


def _speed_test(memory):
    t0 = time.time()

    warmup_size = 10_000
    batch_size = 64
    epochs = 10_000

    # warmup
    step = 0
    for _ in range(warmup_size + batch_size):
        r = random.random()
        memory.add((step, step, step, step), r)
        step += 1

    for _ in tqdm(range(epochs)):

        # add
        r = random.random()
        memory.add((step, step, step, step), r)
        step += 1

        # sample
        (indices, batchs, weights) = memory.sample(batch_size, step)
        assert len(indices) == batch_size
        assert len(batchs) == batch_size
        assert len(weights) == batch_size

        # 重複がないこと
        li_uniq = list(set(batchs))
        assert len(li_uniq) == batch_size

        # update priority
        memory.update(indices, batchs, [random.random() for _ in range(batch_size)])

    print("{}: {}s".format(memory.__class__.__name__, time.time() - t0))


if __name__ == "__main__":
    speed_test()
