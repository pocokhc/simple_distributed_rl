import random
import time

from srl import rl
from srl.rl.memory import registory
from tqdm import tqdm


def speed_test():
    capacity = 100_000_000

    memorys = [
        rl.memory.replay_memory.Config(capacity=capacity),
        rl.memory.proportional_memory.Config(capacity=capacity, alpha=0.8, beta_initial=1, beta_steps=10),
        rl.memory.rankbase_memory.Config(capacity=capacity, alpha=0.8, beta_initial=1, beta_steps=10),
    ]

    for memory_config in memorys:
        memory = registory.make(memory_config)
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
        (indexes, batchs, weights) = memory.sample(batch_size, step)
        assert len(indexes) == batch_size
        assert len(batchs) == batch_size
        assert len(weights) == batch_size

        # 重複がないこと
        li_uniq = list(set(batchs))
        assert len(li_uniq) == batch_size

        # update priority
        memory.update(indexes, batchs, [random.random() for _ in range(batch_size)])

    print("{}: {}s".format(memory.__class__.__name__, time.time() - t0))


if __name__ == "__main__":
    speed_test()
