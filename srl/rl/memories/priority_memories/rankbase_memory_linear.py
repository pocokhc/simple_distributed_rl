import bisect
import math
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from .imemory import IPriorityMemory


def rank_sum(k, a):
    return k * (2 + (k - 1) * a) / 2


def rank_sum_inverse(k, a):
    if a == 0:
        return k
    t = a - 2 + math.sqrt((2 - a) ** 2 + 8 * a * k)
    return t / (2 * a)


class _bisect_wrapper:
    def __init__(self, priority, batch):
        self.priority = priority
        self.batch = batch

    def __lt__(self, o):  # a<b
        return self.priority < o.priority


# TODO: 赤黒木


@dataclass
class RankBaseMemoryLinear(IPriorityMemory):
    capacity: int = 100_000
    alpha: float = 1.0
    beta_initial: float = 0.4
    beta_steps: int = 1_000_000

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

        bisect.insort(self.memory, _bisect_wrapper(priority, batch))

    def sample(self, batch_size: int, step: int) -> Tuple[List[int], List[Any], np.ndarray]:
        batchs = []
        weights = np.ones(batch_size, dtype=np.float32)

        # βは最初は低く、学習終わりに1にする。
        beta = self.beta_initial + (1 - self.beta_initial) * step / self.beta_steps
        if beta > 1:
            beta = 1

        # 合計値をだす
        memory_size = len(self.memory)
        total = rank_sum(memory_size, self.alpha)

        # index_list
        index_list = []
        for _ in range(batch_size):
            for _ in range(9999):  # for safety
                r = random.random() * total
                index = rank_sum_inverse(r, self.alpha)
                index = int(index)  # 整数にする(切り捨て)
                if index not in index_list:
                    index_list.append(index)
                    break

        index_list.sort(reverse=True)
        for i, index in enumerate(index_list):
            o = self.memory.pop(index)  # 後ろから取得するのでindexに変化なし
            batchs.append(o.batch)

            # 重点サンプリングを計算 w = (N * p)^-1
            r1 = rank_sum(index + 1, self.alpha)
            r2 = rank_sum(index, self.alpha)
            prob = (r1 - r2) / total
            weights[i] = (memory_size * prob) ** (-beta)

        weights = weights / weights.max()

        return index_list, batchs, weights

    def update(self, indices: List[int], batchs: List[Any], priorities: np.ndarray) -> None:
        for i in range(len(batchs)):
            priority = float(priorities[i])
            if self.max_priority < priority:
                self.max_priority = priority
            bisect.insort(self.memory, _bisect_wrapper(priority, batchs[i]))

    def backup(self):
        return [
            [(d.priority, d.batch) for d in self.memory],
            self.max_priority,
        ]

    def restore(self, data):
        self.memory = []
        for d in data[0]:
            self.memory.append(_bisect_wrapper(d[0], d[1]))
        self.max_priority = data[1]
