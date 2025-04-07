import bisect
import logging
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .imemory import IPriorityMemory

logger = logging.getLogger(__name__)


def rank_sum(k, a):
    return k * (2 + (k - 1) * a) / 2


def rank_sum_inverse(k, a):
    if a == 0:
        return k
    t = a - 2 + np.sqrt((2 - a) ** 2 + 8 * a * k)
    return t / (2 * a)


@dataclass
class RankBasedMemoryLinear(IPriorityMemory):
    capacity: int = 100_000
    alpha: float = 1.0
    beta_initial: float = 0.4
    beta_steps: int = 1_000_000
    dtype: type = np.float32

    def __post_init__(self):
        self.clear()

    def clear(self):
        self.memory = []
        self.max_priority: float = 1.0

    def length(self) -> int:
        return len(self.memory)

    def add(self, batch, priority: Optional[float] = None):
        if priority is None:
            priority = self.max_priority
        if self.max_priority < priority:
            self.max_priority = priority

        if len(self.memory) >= self.capacity:
            self.memory.pop(0)

        bisect.insort(self.memory, (priority, batch))

    def sample(self, batch_size: int, step: int):
        # βは最初は低く、学習終わりに1にする。
        beta = self.beta_initial + (1 - self.beta_initial) * step / self.beta_steps
        if beta > 1:
            beta = 1

        memory_size = len(self.memory)
        total = rank_sum(memory_size, self.alpha)

        # 重複がないように作成
        idx_list = []
        for i in range(batch_size):
            for _ in range(999):  # for safety
                r = random.random() * total
                idx = int(rank_sum_inverse(r, self.alpha))
                if idx in idx_list:
                    continue
                break
            idx_list.append(idx)
        idx_list.sort(reverse=True)
        idx_list = np.array(idx_list)

        # IS: w = (N * p)^-1
        r2 = rank_sum(idx_list + 1, self.alpha)
        r1 = rank_sum(idx_list, self.alpha)
        prob = (r2 - r1) / total
        weights = (memory_size * prob) ** (-beta)
        weights = weights / weights.max()

        batches = [self.memory[i][1] for i in idx_list]
        for i in idx_list:  # idxがずれないように逆順
            del self.memory[i]

        return batches, weights.astype(self.dtype), batches

    def update(self, batches, priorities: np.ndarray) -> None:
        for b, p in zip(batches, priorities):
            self.add(b, p)

    def backup(self):
        return [
            self.capacity,
            [d[:] for d in self.memory],
            self.max_priority,
        ]

    def restore(self, data):
        if self.capacity != data[0]:
            logger.warning("Capacity mismatch: expected %d, but got %d", self.capacity, data[0])
        self.memory = [d[:] for d in data[1]]
        self.max_priority = data[2]
